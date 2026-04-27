"""
EBSPA — Entanglement-fidelity-aware Best-path Shortest Path Algorithm.

Dijkstra on edge weights -log(F_uv) over the per-timeslot entanglement graph.
This maximises product fidelity (Werner-swap) along the path.

Used as a deterministic strong baseline above ShortestPath (hop count)
but below RL methods.
"""
