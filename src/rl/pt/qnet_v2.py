"""
EdgeQNet — edge-score Q-network for quantum routing.

Input per (request r at node curr, candidate neighbor nbr):
  [deg_curr, dens_curr, bfs_curr (3),
   deg_dst,  dens_dst,  0        (3),
   deg_nbr,  dens_nbr,  bfs_nbr  (3),
   fid_uv, fid_so_far, hops_frac (3)]
  Total: STATE_DIM = 12

Replaces the previous 99-dim GAT-embedding input.  Simple hand-crafted features
carry real routing signal (BFS distance, link capacity, congestion) and are
derived fresh each timeslot — no frozen/random encoder weights.

Output: scalar Q-value per (request, candidate) pair.

QMixerV2: monotonic mixing for QMIX (Hive variant).
  req_feat_dim = QMIX_REQ_DIM = 4   (bfs_curr, bfs_nbr, deg_curr, deg_dst)
  global_state_dim = QMIX_GLOBAL_DIM = 4  (mean_deg, max_deg, mean_fid, req_load)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM      = 12   # edge-state vector length (see module docstring)
QMIX_REQ_DIM   = 4    # per-request feature vector for QMixerV2
QMIX_GLOBAL_DIM= 4    # global state vector for QMixerV2

# Alias kept so any code that imported EMB_DIM still compiles.
EMB_DIM = STATE_DIM


class EdgeQNet(nn.Module):
    """
    MLP scoring a single (request-at-curr, candidate-nbr) edge.

    Batched call: (B, STATE_DIM) → (B, 1) Q-values.
    """

    def __init__(self, state_dim: int = STATE_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, STATE_DIM) → (B, 1)"""
        return self.net(x)

    def score_candidates(self, state_vecs: torch.Tensor) -> torch.Tensor:
        """state_vecs: (K, STATE_DIM) → (K,) Q-values"""
        return self(state_vecs).squeeze(1)


class QMixerV2(nn.Module):
    """
    Monotonic value mixer (QMIX) with variable-length agent Q-values.

    Instead of padding to MAX_REQUESTS, produces one weight per active
    request using a shared per-request score network conditioned on the
    global state.  Monotonicity is enforced with softplus weights.
    """

    def __init__(self, global_state_dim: int = QMIX_GLOBAL_DIM,
                 req_feat_dim: int = QMIX_REQ_DIM,
                 hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.req_proj  = nn.Linear(req_feat_dim, hidden)
        self.hyper_b1  = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, qs: torch.Tensor, mask: torch.Tensor,
                req_feats: torch.Tensor,
                global_state: torch.Tensor) -> torch.Tensor:
        """
        qs          : (B, R) chosen Q-values (padded to R with 0)
        mask        : (B, R) float — 1 for real request, 0 for pad
        req_feats   : (B, R, QMIX_REQ_DIM) per-request features
        global_state: (B, QMIX_GLOBAL_DIM) compact global state

        Returns     : (B, 1) joint Q_tot
        """
        gs_emb = self.hyper_w1(global_state)         # (B, hidden)
        rq_emb = self.req_proj(req_feats)             # (B, R, hidden)
        w = torch.einsum('bh,brh->br', gs_emb, rq_emb)  # (B, R)
        w = F.softplus(w) * mask                      # non-neg, zero pads
        b = self.hyper_b1(global_state)               # (B, 1)
        return (w * qs).sum(dim=1, keepdim=True) + b  # (B, 1)
