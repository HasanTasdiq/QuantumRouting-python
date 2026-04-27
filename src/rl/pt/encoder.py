"""
QuantumGAT — 2-layer Graph Attention Network for entanglement graphs.

No external GNN libraries. Dense-matrix message passing is efficient for
N=100 nodes (the whole adjacency fits in ~40 KB of float32).

Input per timeslot:
  node_feats : (N, node_dim)  — per-node features
  adj_fid    : (N, N) float32 — fidelity of best available entangled pair
               (0 if no pair)
  adj_cnt    : (N, N) float32 — number of available entangled pairs (capacity)

Output:
  H : (N, out_dim) float32 — contextual node embeddings
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

NODE_DIM = 4    # degree, request_density, fid_mean_out, fid_mean_in
EDGE_DIM = 2    # fidelity, capacity (normalised)
HIDDEN   = 32
OUT_DIM  = 32
HEADS    = 4    # multi-head attention, heads concatenated at layer 1


class GATLayerDense(nn.Module):
    """
    Single dense GAT layer.

    H_in  : (N, in_dim)
    A_mask: (N, N) bool — True where an edge exists (attend only over neighbors)
    E     : (N, N, edge_dim) or None — per-edge features

    H_out : (N, heads * out_per_head)  if concat=True
            (N, out_per_head)          if concat=False  (last layer)
    """

    def __init__(self, in_dim: int, out_per_head: int, heads: int,
                 edge_dim: int = 0, concat: bool = True):
        super().__init__()
        self.heads        = heads
        self.out_per_head = out_per_head
        self.concat       = concat

        self.W  = nn.Linear(in_dim, heads * out_per_head, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, out_per_head))
        self.a_dst = nn.Parameter(torch.empty(heads, out_per_head))
        nn.init.xavier_uniform_(self.a_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.a_dst.unsqueeze(0))

        if edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, heads, bias=False)
        else:
            self.edge_proj = None

        self.bias = nn.Parameter(torch.zeros(heads * out_per_head
                                             if concat else out_per_head))

    def forward(self, H: torch.Tensor, A_mask: torch.Tensor,
                E: torch.Tensor | None = None) -> torch.Tensor:
        N = H.size(0)
        # (N, heads * out_per_head) → (N, heads, out_per_head)
        Wh = self.W(H).view(N, self.heads, self.out_per_head)

        # Attention score: e_ij = LeakyReLU(a_src·h_i + a_dst·h_j)
        # src contrib: (N, heads), broadcast to (N, N, heads)
        e_src = (Wh * self.a_src).sum(-1)          # (N, heads)
        e_dst = (Wh * self.a_dst).sum(-1)          # (N, heads)
        # (N, 1, heads) + (1, N, heads) → (N, N, heads)
        e = e_src.unsqueeze(1) + e_dst.unsqueeze(0)

        if E is not None and self.edge_proj is not None:
            # E: (N, N, edge_dim) → (N, N, heads)
            e = e + self.edge_proj(E)

        e = F.leaky_relu(e, negative_slope=0.2)

        # Mask out non-edges
        mask = A_mask.unsqueeze(-1).expand_as(e)   # (N, N, heads)
        e = e.masked_fill(~mask, -1e9)

        alpha = F.softmax(e, dim=1)                 # (N, N, heads) softmax over src

        # Aggregate: out_i = Σ_j α_ij * Wh_j
        # (N, N, heads, 1) * (N, N, heads, out_per_head) → sum over N (dim=1)
        # Efficient: alpha @ Wh across head dimension
        # alpha: (N, N, heads), Wh: (N, heads, out_per_head)
        # → (N, heads, out_per_head)
        out = torch.einsum('ijh,jhd->ihd', alpha, Wh)   # (N, heads, out_per_head)

        if self.concat:
            out = out.reshape(N, self.heads * self.out_per_head)
        else:
            out = out.mean(dim=1)   # average heads for final layer

        return F.elu(out + self.bias)


class QuantumGAT(nn.Module):
    """
    2-layer GAT for quantum entanglement graphs.

    Layer 1: NODE_DIM → HEADS * (HIDDEN // HEADS)  with edge features
    Layer 2: HIDDEN   → OUT_DIM                    no edge features (concat=False)

    Total params ≈ 5k (small by design — state encoder should not dominate).
    """

    def __init__(self, node_dim: int = NODE_DIM, edge_dim: int = EDGE_DIM,
                 hidden: int = HIDDEN, out_dim: int = OUT_DIM, heads: int = HEADS):
        super().__init__()
        assert hidden % heads == 0
        head_dim = hidden // heads

        self.layer1 = GATLayerDense(node_dim, head_dim, heads, edge_dim=edge_dim, concat=True)
        self.layer2 = GATLayerDense(hidden, out_dim, heads, edge_dim=0, concat=False)

    def forward(self, node_feats: torch.Tensor,
                adj_fid: torch.Tensor,
                adj_cnt: torch.Tensor) -> torch.Tensor:
        """
        node_feats : (N, node_dim)
        adj_fid    : (N, N) fidelity (0 = no edge)
        adj_cnt    : (N, N) normalised capacity
        Returns    : (N, out_dim)
        """
        A_mask = (adj_fid > 0)                             # (N, N) bool
        E      = torch.stack([adj_fid, adj_cnt], dim=-1)   # (N, N, 2)

        H = self.layer1(node_feats, A_mask, E)             # (N, HIDDEN)

        # For layer 2, drop edge features (already encoded in H)
        # Use same adjacency (structural)
        H = self.layer2(H, A_mask, None)                   # (N, out_dim)
        return H


def build_graph_tensors(ent_matrix: 'np.ndarray',
                        dist_matrix: 'np.ndarray',
                        request_density: 'np.ndarray',
                        N: int) -> tuple:
    """
    Convert environment arrays to tensors for QuantumGAT.

    ent_matrix     : (N, N) int  — number of entangled pairs per link
    dist_matrix    : (N, N) float — fidelity of available links (0 if none)
    request_density: (N,) float  — number of in-flight requests currently at each node
    N              : number of nodes

    Returns (node_feats, adj_fid, adj_cnt) as float32 tensors on CPU.
    """
    import numpy as np

    # Node features: [out_degree, in_req_density, mean_fid_out, mean_fid_in]
    deg      = (ent_matrix > 0).sum(axis=1).astype(np.float32) / max(N - 1, 1)
    req_dens = request_density.astype(np.float32) / max(request_density.max(), 1.0)

    # Mean fidelity of outgoing available links
    fid_sum  = dist_matrix.sum(axis=1).astype(np.float32)
    fid_cnt  = (dist_matrix > 0).sum(axis=1).astype(np.float32)
    fid_mean = np.where(fid_cnt > 0, fid_sum / np.maximum(fid_cnt, 1), 0.0)

    # Symmetric: mean incoming fidelity (same for undirected graphs)
    fid_mean_in = dist_matrix.sum(axis=0).astype(np.float32)
    fid_cnt_in  = (dist_matrix > 0).sum(axis=0).astype(np.float32)
    fid_mean_in = np.where(fid_cnt_in > 0, fid_mean_in / np.maximum(fid_cnt_in, 1), 0.0)

    node_feats = np.stack([deg, req_dens, fid_mean, fid_mean_in], axis=1)  # (N, 4)

    max_cnt    = float(ent_matrix.max()) if ent_matrix.max() > 0 else 1.0
    adj_cnt    = ent_matrix.astype(np.float32) / max_cnt

    return (
        torch.tensor(node_feats, dtype=torch.float32),
        torch.tensor(dist_matrix, dtype=torch.float32),
        torch.tensor(adj_cnt,     dtype=torch.float32),
    )
