"""
EdgeQNet — edge-score Q-network for quantum routing.

For each (request, candidate_neighbor) pair, predicts Q(route via this edge).

Input per (request r at node u, candidate neighbor v):
  [emb_u (D), emb_dst (D), emb_v (D), fid_uv (1), fid_so_far (1), hops_frac (1)]
  Total: 3*D + 3

Output: scalar Q-value

Parameters: ~30k total (D=32 → 3*32+3=99 → 64 → 32 → 1)

Also provides QMixerV2: monotonic value mixing for QMIX (Hive variant).
Uses request-wise learned weights derived from a shared global-state MLP,
so the batch dimension is always the actual number of active requests
(no padding to a fixed MAX_REQUESTS constant).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

EMB_DIM = 32   # must match QuantumGAT OUT_DIM


class EdgeQNet(nn.Module):
    """
    MLP that scores a single (request-at-u, candidate-v) edge.

    Batched call: pass a (B, 3*D+3) tensor, get (B, 1) Q-values back.
    Caller constructs the batch by iterating over (request, neighbor) pairs.
    """

    def __init__(self, emb_dim: int = EMB_DIM):
        super().__init__()
        in_dim = 3 * emb_dim + 3   # emb_u, emb_dst, emb_v, fid_uv, fid_so_far, hops_frac
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3*D+3) → (B, 1)"""
        return self.net(x)

    def score_candidates(self, emb_u: torch.Tensor, emb_dst: torch.Tensor,
                         H: torch.Tensor,
                         neighbors: list[int],
                         fid_uv: list[float],
                         fid_so_far: float,
                         hops_frac: float) -> torch.Tensor:
        """
        Score all candidate neighbors for one request.

        emb_u    : (D,) embedding of current node
        emb_dst  : (D,) embedding of destination node
        H        : (N, D) all node embeddings (for neighbor lookup)
        neighbors: list of K neighbor node ids
        fid_uv   : list of K edge fidelities
        fid_so_far: Werner fidelity accumulated so far
        hops_frac : hops_used / W  (normalised)

        Returns: (K,) Q-values
        """
        K = len(neighbors)
        if K == 0:
            return torch.empty(0)

        emb_v = H[neighbors]                                  # (K, D)
        eu    = emb_u.unsqueeze(0).expand(K, -1)              # (K, D)
        ed    = emb_dst.unsqueeze(0).expand(K, -1)            # (K, D)

        fid_t = torch.tensor(fid_uv, dtype=torch.float32).unsqueeze(1)     # (K, 1)
        sf_t  = torch.full((K, 1), fid_so_far, dtype=torch.float32)        # (K, 1)
        hp_t  = torch.full((K, 1), hops_frac,  dtype=torch.float32)        # (K, 1)

        x = torch.cat([eu, ed, emb_v, fid_t, sf_t, hp_t], dim=1)          # (K, 3D+3)
        return self(x).squeeze(1)                                           # (K,)


class QMixerV2(nn.Module):
    """
    Monotonic value mixer (QMIX) with variable-length agent Q-values.

    Instead of padding to MAX_REQUESTS, we produce one weight per active
    request using a shared per-request score network conditioned on the
    global state. Monotonicity is enforced with softplus weights.

    global_state_dim: dimension of the compact global state vector.
                      Trainer passes concat(mean_pool(H), max_pool(H)) = 64 dims.
                      This keeps the hypernetwork small (~10k params total).
    """

    def __init__(self, global_state_dim: int = 64, hidden: int = 32):
        super().__init__()
        self.hidden = hidden
        # Hypernetwork: global_state → weight for each request
        # Shared MLP; applied once per timeslot, not once per request.
        self.hyper_w1 = nn.Sequential(
            nn.Linear(global_state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),   # will produce per-slot weights via outer product
        )
        # Per-request context: emb_u + emb_dst → slot embedding (D → hidden)
        self.req_proj  = nn.Linear(2 * EMB_DIM, hidden)

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
        mask        : (B, R) float, 1 = active request, 0 = pad
        req_feats   : (B, R, 2*D) concat(emb_u, emb_dst) per request
        global_state: (B, G) compact global state (mean+max pooled H, 64 dims)

        Returns     : (B, 1) joint Q_tot
        """
        B, R = qs.shape

        # (B, hidden) → used as key to compute weight per request slot
        gs_emb  = self.hyper_w1(global_state)            # (B, hidden)

        # (B, R, hidden)
        rq_emb  = self.req_proj(req_feats)                # (B, R, hidden)

        # Dot product between gs_emb and each request slot → (B, R) weights
        w = torch.einsum('bh,brh->br', gs_emb, rq_emb)   # (B, R)
        w = F.softplus(w)                                 # non-negative (monotonicity)
        w = w * mask                                      # zero out padding

        b = self.hyper_b1(global_state)                  # (B, 1)

        q_tot = (w * qs).sum(dim=1, keepdim=True) + b    # (B, 1)
        return q_tot
