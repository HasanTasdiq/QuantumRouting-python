"""
EdgeQNet — edge-score Q-network for quantum routing.

Input per (request r at node curr, candidate neighbor nbr):
  [deg_curr, dens_curr, bfs_curr (3),
   deg_dst,  dens_dst,  0        (3),
   deg_nbr,  dens_nbr,  bfs_nbr  (3),
   fid_uv, fid_so_far, hops_frac (3)]
  Total: STATE_DIM = 29

Replaces the previous 99-dim GAT-embedding input.  Simple hand-crafted features
carry real routing signal (BFS distance, link capacity, congestion) and are
derived fresh each timeslot — no frozen/random encoder weights.

Output: scalar Q-value per (request, candidate) pair.

MAPPOActorCritic: centralized-training actor-critic for Hive.
  Actor scores a variable candidate-neighbor set.
  Critic sees compact global state, pooled candidate/request features, and a
  learned topology embedding from the active entanglement graph.
"""
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

STATE_DIM       = 29   # edge-state vector length (see local_trainer_v2._edge_state)
MAPPO_BASE_GLOBAL_DIM = 16
MAPPO_PENDING_DIM = 12
MAPPO_GLOBAL_DIM= MAPPO_BASE_GLOBAL_DIM + MAPPO_PENDING_DIM
GNN_NODE_DIM    = 8    # per-node topology features (see _node_features)
GNN_EMB_DIM     = 64
MAPPO_ACTOR_DIM = STATE_DIM + 3 * GNN_EMB_DIM
MAPPO_CRITIC_DIM= MAPPO_GLOBAL_DIM + 2 * STATE_DIM + 5 * GNN_EMB_DIM
MAPPO_CF_DIM    = MAPPO_CRITIC_DIM + STATE_DIM + GNN_EMB_DIM

# Compatibility aliases for older imports and checkpoints.
QMIX_REQ_DIM    = 4
QMIX_GLOBAL_DIM = MAPPO_GLOBAL_DIM

# Alias kept so any code that imported EMB_DIM still compiles.
EMB_DIM = STATE_DIM


class EdgeQNet(nn.Module):
    """
    MLP scoring a single (request-at-curr, candidate-nbr) edge.

    Batched call: (B, STATE_DIM) → (B, 1) Q-values.
    """

    def __init__(self, state_dim: int = STATE_DIM, arch: str | None = None):
        super().__init__()
        self.arch = (arch or os.environ.get("DQN_ARCH", "deep")).strip().lower()
        if self.arch == "linear":
            self.net = nn.Linear(state_dim, 1)
        elif self.arch == "tiny":
            self.net = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )
        else:
            self.arch = "deep"
            self.net = nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
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


class GraphSAGELayer(nn.Module):
    """Mean-aggregation message passing layer for the active topology graph."""

    def __init__(self, hidden: int):
        super().__init__()
        self.update = nn.Linear(2 * hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.numel() == 0:
            agg = torch.zeros_like(h)
        else:
            src = edge_index[0].long()
            dst = edge_index[1].long()
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, h[src])
            deg = torch.zeros(h.size(0), device=h.device, dtype=h.dtype)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
            agg = agg / deg.clamp_min(1.0).unsqueeze(1)

        out = self.update(torch.cat([h, agg], dim=1))
        return F.relu(self.norm(out)) + h


class TopologyGNN(nn.Module):
    """
    Lightweight GraphSAGE encoder over the currently entangled graph.

    node_features: (N, GNN_NODE_DIM)
    edge_index   : (2, E) directed active edges
    returns      : (N, GNN_EMB_DIM)
    """

    def __init__(self, node_dim: int = GNN_NODE_DIM,
                 hidden: int = GNN_EMB_DIM,
                 layers: int = 3):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(node_dim, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(GraphSAGELayer(hidden) for _ in range(layers))
        self.output = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )

    def forward(self, node_features: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        h = self.input(node_features)
        for layer in self.layers:
            h = layer(h, edge_index)
        return self.output(h)


class MAPPOActorCritic(nn.Module):
    """
    Variable-action actor plus centralized critic for QuRA-Hive.

    Actor input:
      candidate edge-state rows plus topology embeddings for
      (current node, destination node, candidate neighbor).

    Critic input:
      [global_state, mean/max(candidate_states), graph mean/max,
       curr embedding, dst embedding, mean candidate-neighbor embedding].

    Counterfactual head input:
      critic context plus the selected candidate row and its neighbor
      embedding. This auxiliary head predicts whether the chosen next hop was
      better than the alternatives under the current contested frontier.
    """

    def __init__(self,
                 state_dim: int = STATE_DIM,
                 global_dim: int = MAPPO_GLOBAL_DIM,
                 hidden: int = 128):
        super().__init__()
        self.global_dim = global_dim
        self.state_dim = state_dim
        self.gnn = TopologyGNN()
        self.actor = nn.Sequential(
            nn.Linear(MAPPO_ACTOR_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
        self.critic = nn.Sequential(
            nn.Linear(MAPPO_CRITIC_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.counterfactual = nn.Sequential(
            nn.Linear(MAPPO_CF_DIM, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def encode_graph(self, node_features: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
        return self.gnn(node_features, edge_index)

    def actor_input(self, candidate_states: torch.Tensor,
                    node_embeddings: torch.Tensor,
                    curr_id: int,
                    dst_id: int,
                    candidate_ids: torch.Tensor) -> torch.Tensor:
        k = candidate_states.size(0)
        curr_emb = node_embeddings[int(curr_id)].unsqueeze(0).expand(k, -1)
        dst_emb = node_embeddings[int(dst_id)].unsqueeze(0).expand(k, -1)
        nbr_emb = node_embeddings[candidate_ids.long()]
        return torch.cat([candidate_states, curr_emb, dst_emb, nbr_emb], dim=1)

    def logits(self, candidate_states: torch.Tensor,
               node_features: torch.Tensor | None = None,
               edge_index: torch.Tensor | None = None,
               curr_id: int | None = None,
               dst_id: int | None = None,
               candidate_ids: torch.Tensor | None = None) -> torch.Tensor:
        """candidate_states: (K, STATE_DIM) -> (K,) logits."""
        if node_features is None:
            zeros = candidate_states.new_zeros(candidate_states.size(0),
                                               3 * GNN_EMB_DIM)
            return self.actor(torch.cat([candidate_states, zeros], dim=1)).squeeze(-1)
        node_embeddings = self.encode_graph(node_features, edge_index)
        return self.logits_from_embeddings(candidate_states, node_embeddings,
                                           int(curr_id), int(dst_id),
                                           candidate_ids)

    def logits_from_embeddings(self, candidate_states: torch.Tensor,
                               node_embeddings: torch.Tensor,
                               curr_id: int,
                               dst_id: int,
                               candidate_ids: torch.Tensor) -> torch.Tensor:
        """Score candidates using a precomputed topology embedding."""
        x = self.actor_input(candidate_states, node_embeddings,
                             int(curr_id), int(dst_id), candidate_ids)
        return self.actor(x).squeeze(-1)

    def critic_input(self, candidate_states: torch.Tensor,
                     global_state: torch.Tensor,
                     node_features: torch.Tensor | None = None,
                     edge_index: torch.Tensor | None = None,
                     curr_id: int | None = None,
                     dst_id: int | None = None,
                     candidate_ids: torch.Tensor | None = None) -> torch.Tensor:
        """
        Unbatched candidate set for one request decision.
        """
        pooled_mean = candidate_states.mean(dim=0)
        pooled_max = candidate_states.max(dim=0).values
        if node_features is None:
            topo = candidate_states.new_zeros(5 * GNN_EMB_DIM)
        else:
            node_embeddings = self.encode_graph(node_features, edge_index)
            topo = self.critic_topology_from_embeddings(
                node_embeddings, int(curr_id), int(dst_id), candidate_ids)

        return torch.cat([global_state, pooled_mean, pooled_max, topo], dim=0)

    def critic_topology_from_embeddings(self, node_embeddings: torch.Tensor,
                                        curr_id: int,
                                        dst_id: int,
                                        candidate_ids: torch.Tensor) -> torch.Tensor:
        graph_mean = node_embeddings.mean(dim=0)
        graph_max = node_embeddings.max(dim=0).values
        curr_emb = node_embeddings[int(curr_id)]
        dst_emb = node_embeddings[int(dst_id)]
        nbr_mean = node_embeddings[candidate_ids.long()].mean(dim=0)
        return torch.cat([graph_mean, graph_max, curr_emb, dst_emb, nbr_mean])

    def critic_input_from_embeddings(self, candidate_states: torch.Tensor,
                                     global_state: torch.Tensor,
                                     node_embeddings: torch.Tensor,
                                     curr_id: int,
                                     dst_id: int,
                                     candidate_ids: torch.Tensor) -> torch.Tensor:
        pooled_mean = candidate_states.mean(dim=0)
        pooled_max = candidate_states.max(dim=0).values
        topo = self.critic_topology_from_embeddings(
            node_embeddings, curr_id, dst_id, candidate_ids)
        return torch.cat([global_state, pooled_mean, pooled_max, topo], dim=0)

    def counterfactual_input_from_embeddings(self,
                                             candidate_states: torch.Tensor,
                                             global_state: torch.Tensor,
                                             node_embeddings: torch.Tensor,
                                             curr_id: int,
                                             dst_id: int,
                                             candidate_ids: torch.Tensor,
                                             action_idx: int) -> torch.Tensor:
        critic_ctx = self.critic_input_from_embeddings(
            candidate_states, global_state, node_embeddings,
            curr_id, dst_id, candidate_ids)
        chosen_state = candidate_states[int(action_idx)]
        chosen_nbr = node_embeddings[candidate_ids.long()[int(action_idx)]]
        return torch.cat([critic_ctx, chosen_state, chosen_nbr], dim=0)

    def value(self, candidate_states: torch.Tensor,
              global_state: torch.Tensor,
              node_features: torch.Tensor | None = None,
              edge_index: torch.Tensor | None = None,
              curr_id: int | None = None,
              dst_id: int | None = None,
              candidate_ids: torch.Tensor | None = None) -> torch.Tensor:
        x = self.critic_input(candidate_states, global_state, node_features,
                              edge_index, curr_id, dst_id, candidate_ids)
        return self.critic(x).squeeze(-1)

    def value_from_embeddings(self, candidate_states: torch.Tensor,
                              global_state: torch.Tensor,
                              node_embeddings: torch.Tensor,
                              curr_id: int,
                              dst_id: int,
                              candidate_ids: torch.Tensor) -> torch.Tensor:
        x = self.critic_input_from_embeddings(
            candidate_states, global_state, node_embeddings,
            int(curr_id), int(dst_id), candidate_ids)
        return self.critic(x).squeeze(-1)

    def counterfactual_from_embeddings(self,
                                       candidate_states: torch.Tensor,
                                       global_state: torch.Tensor,
                                       node_embeddings: torch.Tensor,
                                       curr_id: int,
                                       dst_id: int,
                                       candidate_ids: torch.Tensor,
                                       action_idx: int) -> torch.Tensor:
        x = self.counterfactual_input_from_embeddings(
            candidate_states, global_state, node_embeddings,
            int(curr_id), int(dst_id), candidate_ids, int(action_idx))
        return self.counterfactual(x).squeeze(-1)
