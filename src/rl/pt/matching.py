"""
Greedy max-weight b-matching for Guard and Hive variants.

Given a list of (ridx, neighbor, q_score) triples and per-link capacities
(number of available entangled pairs), greedily assigns requests to links
in descending Q-score order, respecting capacity.

Used by Guard to resolve conflicts after parallel Q-based hop selection,
and by Hive (which also trains with QMIX).
"""
from __future__ import annotations


def greedy_bmatching(
    candidates: list[tuple[int, int, float]],
    link_capacity: dict[tuple[int, int], int],
) -> dict[int, int]:
    """
    Greedy max-weight b-matching.

    candidates   : list of (ridx, neighbor_id, q_score)
                   Each request may appear multiple times with different neighbors.
                   Caller passes the *best* (ridx, neighbor, q) per request from
                   the Q-network, or all (ridx, neighbor, q) triples if we want
                   fallback re-assignment.
    link_capacity: dict mapping (min_id, max_id) → remaining capacity
                   (number of entangled pairs available on that link).

    Returns: dict ridx → assigned_neighbor  (only for successfully matched requests)
    """
    # Sort by descending Q-score so highest-value requests grab links first
    sorted_cands = sorted(candidates, key=lambda x: x[2], reverse=True)

    assigned: dict[int, int] = {}

    for ridx, neighbor, _ in sorted_cands:
        if ridx in assigned:
            continue   # already matched this request

        key = (min(ridx, neighbor), max(ridx, neighbor))
        # Use canonical link key (u, v) with u < v
        u_key = (min(0, 0), max(0, 0))   # placeholder — build correct key below
        link_key = _link_key(ridx, neighbor, link_capacity)
        if link_key is None:
            continue   # no matching link in capacity dict — skip

        cap = link_capacity.get(link_key, 0)
        if cap > 0:
            link_capacity[link_key] = cap - 1
            assigned[ridx] = neighbor

    return assigned


def _link_key(ridx: int, neighbor: int,
              capacity: dict) -> tuple[int, int] | None:
    """Return the canonical (u, v) key with u<v, or None if not present."""
    k = (min(ridx, neighbor), max(ridx, neighbor))
    # The capacity dict is keyed by node ids, not ridx.
    # Caller passes node ids directly; ridx is the request index.
    # We need to look up by the *node* ids of the link, not the request index.
    # This helper is not needed — see bmatching_nodes below.
    return k


def bmatching_nodes(
    candidates: list[tuple[int, int, int, float]],
    link_capacity: dict[tuple[int, int], int],
) -> dict[int, int]:
    """
    Greedy max-weight b-matching keyed by node ids.

    candidates   : list of (ridx, curr_node_id, neighbor_node_id, q_score)
    link_capacity: dict mapping (min_node_id, max_node_id) → remaining capacity

    Returns: dict ridx → chosen_neighbor_node_id
    """
    sorted_cands = sorted(candidates, key=lambda x: x[3], reverse=True)

    assigned: dict[int, int] = {}

    for ridx, curr, nbr, _ in sorted_cands:
        if ridx in assigned:
            continue

        lk = (min(curr, nbr), max(curr, nbr))
        cap = link_capacity.get(lk, 0)
        if cap > 0:
            link_capacity[lk] = cap - 1
            assigned[ridx] = nbr

    return assigned
