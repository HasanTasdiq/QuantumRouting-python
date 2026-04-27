"""
QuRA-v2 local trainer — all four variants share one architecture.

Variants (controlled by `variant` param):
  seq   — sequential single-request routing, no coordination
  flock — parallel routing, no conflict resolution
  guard — parallel routing + greedy b-matching conflict resolution
  hive  — guard + QMIX joint training

Key fixes vs v1:
  - State: graph-structured (node embeddings from GAT), NOT 20228-dim flat
  - Action: relative neighbor index, not absolute node index
  - TTL: W timeslots (default 75), not 1
  - F_min gate: Werner fidelity < F_min counts as failure
  - Gradient flow: encoder/q-net/mixer all in optimizer, no @torch.no_grad wrapping
"""
