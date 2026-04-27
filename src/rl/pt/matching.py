"""
Greedy max-weight b-matching for Guard and Hive variants.

Given a list of (ridx, neighbor, q_score) triples and per-link capacities
(number of available entangled pairs), greedily assigns requests to links
in descending Q-score order, respecting capacity.

Used by Guard to resolve conflicts after parallel Q-based hop selection,
and by Hive (which also trains with QMIX).
"""
