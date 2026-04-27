"""
Unified CSV metrics for all QuRA-v2 algorithms.

Schema per row:
  timeslot, algo, load, requests_in, served, served_F_geq_Fmin,
  mean_F_delivered, conflicts, wall_ms

F_min gate is applied identically to all algorithms (QuRA variants, RELiQ,
EBSPA, ShortestPath) so throughput numbers are comparable.
"""
