"""
EBSPA — Entanglement-fidelity-aware Best-path Shortest Path Algorithm.

Dijkstra on edge weights -log(F_uv) over the per-timeslot entanglement graph.
This maximises product fidelity (Werner-swap) along the path.

Used as a deterministic strong baseline above ShortestPath (hop count)
but below RL methods.
"""
from __future__ import annotations
import os
import sys
import numpy as np

_here     = os.path.dirname(os.path.abspath(__file__))
_src_dir  = os.path.normpath(os.path.join(_here, '../..'))
_algo_dir = os.path.join(_src_dir, 'quantum', 'algorithm')
for _p in [_algo_dir, _src_dir, os.path.join(_src_dir, 'quantum')]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AlgorithmBase import AlgorithmBase

TTL_W = int(os.environ.get("TTL_W", "75"))
F_MIN = float(os.environ.get("F_MIN", "0.7"))
SIZE  = int(os.environ.get("SIZE", "100"))


def _werner_swap(f1: float, f2: float) -> float:
    return f1 * f2 + (1.0 - f1) * (1.0 - f2) / 3.0


class EBSPA(AlgorithmBase):
    """
    Entanglement-fidelity Best-path Shortest Path Algorithm.

    Runs Dijkstra on -log(fidelity) edge weights to find the max-fidelity path
    through the per-timeslot entanglement graph.  F_min gate applied.
    """

    def __init__(self, topo, param=None, name='EBSPA'):
        super().__init__(topo)
        self.name         = name
        self.requests     = []
        self.totalRequest = 0
        self.totalWaiting = 0
        self.totalQubits  = 0
        self.requestState = []

    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))
        self.srcDstPairs = []
        self.requestState = []
        for idx, (src, dst, _) in enumerate(self.requests):
            self.requestState.append([src, dst, src.id, set({src.id}), idx, False])

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaiting += len(self.requests)
        self.result.idleTime += len(self.requests)
        self.result.numOfTimeslot += 1
        if self.requests:
            _assignable = True
            while _assignable:
                _assignable = False
                for link in self.topo.links:
                    if link.assignable():
                        _assignable = True
                        if np.random.random() > 0.5:
                            link.assignQubits()
                            self.totalQubits += 2

    def _build_matrices(self):
        ent  = np.zeros((SIZE, SIZE), dtype=np.float32)
        dist = np.zeros((SIZE, SIZE), dtype=np.float32)
        cap  = {}
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                i, j  = link.n1.id, link.n2.id
                ent[i][j] += 1;  ent[j][i] += 1
                dist[i][j] = link.fidelity;  dist[j][i] = link.fidelity
                lk = (min(i, j), max(i, j))
                cap[lk] = cap.get(lk, 0) + 1
        return ent, dist, cap

    def p4(self):
        ent, dist, cap = self._build_matrices()
        avail_cap = dict(cap)

        success_req  = 0
        total_fid    = 0.0
        success_fid  = 0

        for rs in self.requestState:
            if rs[5]:
                continue
            src_id = int(rs[0].id)
            dst_id = int(rs[1].id)

            # Dijkstra on -log(fidelity) weights
            path, path_fid = self._dijkstra_fid(src_id, dst_id, dist, avail_cap)
            if path is None:
                continue

            # Consume capacity
            for u, v in zip(path[:-1], path[1:]):
                lk = (min(u, v), max(u, v))
                avail_cap[lk] = max(0, avail_cap.get(lk, 0) - 1)

            rs[5] = True
            if path_fid >= F_MIN:
                success_req += 1
                success_fid += 1
                total_fid   += path_fid

        self.requests     = []
        self.requestState = []

        self.result.successfulRequest += success_req
        self.result.successfulRequestPerRound.append(success_req)
        self.result.entanglementPerRound.append(success_req)
        avg_fid = total_fid / max(success_fid, 1)
        self.result.fidelityPerRound.append(avg_fid)
        self.result.rewardPerRound.append(float(success_req))
        self.printResult()
        return self.result

    @staticmethod
    def _dijkstra_fid(src: int, dst: int,
                      fid_matrix: np.ndarray,
                      avail_cap: dict) -> tuple:
        """
        Dijkstra minimising -log(fidelity) (maximises path fidelity product).
        Returns (path_list, product_fidelity) or (None, 0) if unreachable.
        """
        import heapq
        # heap: (neg_log_fid, current_node, path)
        INF = float('inf')
        dist_heap: list = [(0.0, src, [src])]
        best = {src: 0.0}

        while dist_heap:
            cost, node, path = heapq.heappop(dist_heap)
            if node == dst:
                prod_fid = float(np.exp(-cost)) if cost < INF else 0.0
                return path, prod_fid
            if cost > best.get(node, INF):
                continue
            row = fid_matrix[node]
            nbrs = np.nonzero(row)[0]
            for nbr in nbrs:
                lk = (min(node, int(nbr)), max(node, int(nbr)))
                if avail_cap.get(lk, 0) < 1:
                    continue
                if int(nbr) in path:
                    continue
                f = float(row[nbr])
                if f <= 0:
                    continue
                new_cost = cost - np.log(f)
                nbr_i    = int(nbr)
                if new_cost < best.get(nbr_i, INF):
                    best[nbr_i] = new_cost
                    heapq.heappush(dist_heap, (new_cost, nbr_i, path + [nbr_i]))

        return None, 0.0

    def printResult(self):
        self.topo.clearAllEntanglements()
        if self.totalRequest > 0:
            self.result.waitingTime = self.totalWaiting / self.totalRequest
            self.result.usedQubits  = self.totalQubits  / self.totalRequest
        self.result.remainRequestPerRound.append(len(self.requests))
