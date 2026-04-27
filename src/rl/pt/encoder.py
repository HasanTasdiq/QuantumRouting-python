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
