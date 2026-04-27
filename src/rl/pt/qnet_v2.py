"""
EdgeQNet — edge-score Q-network for quantum routing.

For each (request, candidate_neighbor) pair, predicts Q(route via this edge).

Input per (request r at node u, candidate neighbor v):
  [emb_u (D), emb_dst (D), emb_v (D), fid_uv (1), fid_so_far (1), hops_frac (1)]
  Total: 3*D + 3

Output: scalar Q-value

Parameters: ~30k total (D=32 → 3*32+3=99 → 64 → 32 → 1)
"""
