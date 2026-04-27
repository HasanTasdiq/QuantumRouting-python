"""
QuRA-v2 local trainer — all four variants share one architecture.

Variants (controlled by `variant` param):
  seq   — sequential single-request routing, no coordination
  flock — parallel routing, no conflict resolution
  guard — parallel routing + greedy b-matching conflict resolution
  hive  — guard + QMIX joint training

Fixes vs previous version:
  - n-step buffer: per-request push_sequence, no cross-request contamination
  - Bellman target: next_state = argmax_Q candidate edge, not first neighbor
  - Encoder removed: edge states use hand-crafted features (degree, BFS,
    request density) — no frozen/random GAT embeddings
  - Drop transitions: R_TTL attached to last real hop, not zero-vector phantom
  - flush_episode called automatically via push_sequence
"""
from __future__ import annotations

import os
import sys
import random as _rng
import time

import numpy as np
import torch

# ── Path setup ────────────────────────────────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_src_dir  = os.path.normpath(os.path.join(_here, '../..'))
_algo_dir = os.path.join(_src_dir, 'quantum', 'algorithm')
for _p in [_algo_dir, _src_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AlgorithmBase import AlgorithmBase
from .agent_v2  import DQRLAgentV2
from .matching  import bmatching_nodes
from .replay_v2 import STEP_BETWEEN_TRAIN, MIN_REPLAY, TRAINING_MODE, GAMMA
from .qnet_v2   import STATE_DIM, QMIX_REQ_DIM, QMIX_GLOBAL_DIM

# ── Constants ─────────────────────────────────────────────────────────────────
SIZE           = int(os.environ.get("SIZE", "100"))
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"
MODEL_DIR      = os.environ.get("MODEL_DIR", "/tmp/qrouting_model")

TTL_W = int(os.environ.get("TTL_W", "75"))
F_MIN = float(os.environ.get("F_MIN", "0.7"))

_SHAPING_COEFF = float(os.environ.get("SHAPING_COEFF", "0.1"))
_MAX_HOPS      = float(SIZE)

R_SUCCESS   = 10.0
R_FAIL_FMIN = -2.0
R_HOP       = -0.01
R_TTL       = -0.05

_ttime_env = int(os.environ.get("TTIME", "10000"))
LOG_EVERY  = int(os.environ.get("LOG_EVERY", str(max(1, _ttime_env // 10))))


def _werner_swap(f1: float, f2: float) -> float:
    return f1 * f2 + (1.0 - f1) * (1.0 - f2) / 3.0


def _epsilon(ts: int) -> float:
    """Linear ε decay: warmup (ε=1) → decay (1→0) → exploit (ε=0)."""
    if INFERENCE_MODE:
        return 0.0
    if TRAINING_MODE == "paper":
        s, e = 4000, 16000
    elif TRAINING_MODE == "mid":
        s, e = 1600, 6400
    else:   # smoke
        s, e = 400, 1600
    if ts < s:
        return 1.0
    if ts >= e:
        return 0.0
    return max(0.0, 1.0 - (ts - s) / (e - s))


class QuRA_Local_v2(AlgorithmBase):
    """
    Single class implementing all four QuRA-v2 variants.

    variant : 'seq' | 'flock' | 'guard' | 'hive'
    """

    def __init__(self, topo, param=None, name='QuRA_v2', variant: str = 'guard'):
        super().__init__(topo)
        self.name     = name
        self.variant  = variant
        self.use_qmix = (variant == 'hive')

        self.requests      = []
        self.requestState  = []
        self.totalRequest  = 0
        self.totalWaiting  = 0
        self.totalQubits   = 0
        self._push_ctr     = 0
        self._loss_log: list[float] = []

        self._suppress_base_log = True
        self.agent = DQRLAgentV2(pid=0, num_nodes=SIZE, use_qmix=self.use_qmix)
        self._model_path = os.path.join(
            MODEL_DIR,
            f"{name.lower().replace(' ', '_').replace('-', '_')}.pt.gz")

        if INFERENCE_MODE:
            self._load_weights()

    # ── Weight I/O ────────────────────────────────────────────────────────────

    def _save_weights(self) -> None:
        self.agent.save_weights(self._model_path)
        print(f"[{self.name}] weights → {self._model_path}")

    def _load_weights(self) -> bool:
        ok = self.agent.load_weights(self._model_path)
        if ok:
            print(f"[{self.name}] weights loaded from {self._model_path}")
        else:
            print(f"[{self.name}] no checkpoint — random policy")
        return ok

    # ── AlgorithmBase callbacks ───────────────────────────────────────────────

    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))
        self.srcDstPairs = []

        _max_q = max(self.topo.numOfRequestPerRound * 5, 50)
        if len(self.requests) > _max_q:
            self.requests = self.requests[-_max_q:]

        self.requestState = []
        for idx, (src, dst, _) in enumerate(self.requests):
            self.requestState.append(
                [src, dst, src.id, frozenset({src.id}), idx, False, 1.0])

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaiting += len(self.requests)
        self.result.idleTime += len(self.requests)
        self.result.numOfTimeslot += 1
        if self.requests:
            self._randPFT()

    def _randPFT(self):
        assignable = True
        while assignable:
            assignable = False
            for link in self.topo.links:
                if link.assignable():
                    assignable = True
                    if np.random.random() > 0.5:
                        link.assignQubits()
                        self.totalQubits += 2

    # ── Graph helpers ─────────────────────────────────────────────────────────

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

    def _req_density(self) -> np.ndarray:
        dens = np.zeros(SIZE, dtype=np.float32)
        for rs in self.requestState:
            if not rs[5]:
                dens[int(rs[2])] += 1.0
        return dens

    def _compute_global_state(self, ent: np.ndarray,
                               dist: np.ndarray,
                               req_dens: np.ndarray) -> np.ndarray:
        """4-dim global state for QMIX: [mean_deg, max_deg, mean_fid, req_load]."""
        degrees  = np.sum(ent > 0, axis=1)
        link_fids = dist[ent > 0]
        return np.array([
            float(degrees.mean()) / max(SIZE, 1),
            float(degrees.max())  / max(SIZE, 1),
            float(link_fids.mean()) if link_fids.size > 0 else 0.0,
            float(req_dens.sum())  / max(SIZE, 1),
        ], dtype=np.float32)

    # ── Edge-score state vector ───────────────────────────────────────────────

    @staticmethod
    def _edge_state(ent: np.ndarray, dist: np.ndarray,
                    req_dens: np.ndarray, bfs_dists: dict,
                    curr: int, dst: int, nbr: int,
                    fid_uv: float, fid_so_far: float,
                    hops_used: int) -> np.ndarray:
        """
        Build STATE_DIM=12 edge-state vector.

        3 nodes × (norm_degree, norm_req_density, norm_bfs_to_dst) + fid_uv + fid_so_far + hops_frac

        BFS distance uses the graph at the START of the timeslot (precomputed).
        Degree and req_density are the live values (updated as links are consumed).
        """
        bfs_d      = bfs_dists.get(dst, {})
        total_dens = max(float(req_dens.sum()), 1.0)

        def node_feat(nid: int) -> list:
            deg  = float(np.sum(ent[nid] > 0)) / max(SIZE, 1)
            dens = float(req_dens[nid]) / total_dens
            bfs  = float(bfs_d.get(nid, SIZE)) / SIZE
            return [deg, dens, bfs]

        return np.array(
            node_feat(curr) + node_feat(dst) + node_feat(nbr) +
            [fid_uv, fid_so_far, min(float(hops_used) / TTL_W, 1.0)],
            dtype=np.float32
        )

    # ── Potential-based shaping ───────────────────────────────────────────────

    @staticmethod
    def _precompute_bfs(G_nx, dsts: set) -> dict:
        import networkx as nx
        result = {}
        for d in dsts:
            try:
                result[d] = nx.single_source_shortest_path_length(G_nx, d)
            except Exception:
                result[d] = {}
        return result

    @staticmethod
    def _phi_cache(bfs: dict, curr: int, dst: int) -> float:
        if curr == dst:
            return 0.0
        d = bfs.get(dst, {}).get(curr, _MAX_HOPS)
        return -d / _MAX_HOPS

    # ── p4 routing loop ───────────────────────────────────────────────────────

    def p4(self):
        t0 = time.perf_counter()

        if not self.requestState:
            for lst in (self.result.successfulRequestPerRound,
                        self.result.entanglementPerRound,
                        self.result.fidelityPerRound,
                        self.result.rewardPerRound):
                lst.append(0)
            self.printResult()
            return self.result

        ent, dist, cap = self._build_matrices()
        req_dens        = self._req_density()

        gs_vec = self._compute_global_state(ent, dist, req_dens) if self.use_qmix else None

        eps         = _epsilon(self.timeSlot)
        success_req = 0
        success_fid = 0
        total_fid   = 0.0

        # Per-request transition lists: ridx → [(s, a, r, ns, done), ...]
        req_transitions: dict[int, list] = {
            ridx: [] for ridx in range(len(self.requestState))
        }
        qmix_states: list = []   # flat list of (s, a, r, ns, rf_vec) for Hive

        avail_cap = dict(cap)

        import networkx as nx
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(SIZE))
        for (u, v), c in cap.items():
            if c > 0:
                G_nx.add_edge(u, v)
        unique_dsts = {int(rs[1].id) for rs in self.requestState if not rs[5]}
        bfs_dists   = self._precompute_bfs(G_nx, unique_dsts)

        MAX_HOPS_PER_REQ = min(15, TTL_W)
        hop_counts = [0] * len(self.requestState)

        for _hop in range(MAX_HOPS_PER_REQ):
            pending = []
            for ridx, rs in enumerate(self.requestState):
                if rs[5]:
                    continue
                curr    = int(rs[2])
                dst_id  = rs[1].id
                visited = rs[3]
                nbrs    = [n for n in np.nonzero(ent[curr])[0]
                           if n not in visited and n != curr]
                if nbrs:
                    pending.append((ridx, rs, curr, dst_id, visited, nbrs))

            if not pending:
                break

            if self.variant == 'seq':
                chosen = self._route_seq(
                    pending, eps, ent, dist, req_dens, bfs_dists, hop_counts, avail_cap)
            else:
                chosen = self._route_parallel(
                    pending, eps, ent, dist, req_dens, bfs_dists, hop_counts, avail_cap,
                    deconflict=(self.variant in ('guard', 'hive')))

            for ridx, nbr_id, state_v in chosen:
                rs     = self.requestState[ridx]
                curr   = int(rs[2])
                dst_id = rs[1].id

                lk     = (min(curr, nbr_id), max(curr, nbr_id))
                f_hop  = float(dist[curr][nbr_id])
                f_old  = float(rs[6])
                f_new  = _werner_swap(f_old, f_hop)
                rs[6]  = f_new

                ent[curr][nbr_id]  = max(0.0, ent[curr][nbr_id]  - 1)
                ent[nbr_id][curr]  = max(0.0, ent[nbr_id][curr]  - 1)
                avail_cap[lk]      = max(0, avail_cap.get(lk, 0) - 1)

                rs[2]           = nbr_id
                rs[3]           = rs[3] | {nbr_id}
                hop_counts[ridx]+= 1
                done            = (nbr_id == dst_id)

                phi_s   = self._phi_cache(bfs_dists, curr,   dst_id)
                phi_ns  = self._phi_cache(bfs_dists, nbr_id, dst_id)
                shaping = _SHAPING_COEFF * (GAMMA * phi_ns - phi_s)

                if done:
                    rs[5] = True
                    if f_new >= F_MIN:
                        raw_r = R_SUCCESS
                        success_req += 1
                        success_fid += 1
                        total_fid   += f_new
                    else:
                        raw_r = R_FAIL_FMIN
                else:
                    raw_r = R_HOP

                reward = raw_r + shaping

                # ── Bellman target: next_state = argmax Q candidate ───────────
                next_nbrs = [n for n in np.nonzero(ent[nbr_id])[0]
                             if n not in rs[3] and n != nbr_id]
                if done or not next_nbrs:
                    next_state_v = state_v   # terminal: done flag zeroes bootstrap
                elif len(next_nbrs) == 1:
                    next_state_v = self._edge_state(
                        ent, dist, req_dens, bfs_dists, nbr_id, dst_id,
                        next_nbrs[0], float(dist[nbr_id][next_nbrs[0]]),
                        f_new, hop_counts[ridx])
                else:
                    nxt_vecs = np.stack([
                        self._edge_state(
                            ent, dist, req_dens, bfs_dists, nbr_id, dst_id, n,
                            float(dist[nbr_id][n]), f_new, hop_counts[ridx])
                        for n in next_nbrs
                    ])
                    best_idx     = int(np.argmax(self.agent.score_neighbors_v(nxt_vecs)))
                    next_state_v = nxt_vecs[best_idx]

                req_transitions[ridx].append((state_v, 0, reward, next_state_v, done))

                if self.use_qmix:
                    rf_vec = np.array([
                        bfs_dists.get(dst_id, {}).get(curr,   SIZE) / SIZE,
                        bfs_dists.get(dst_id, {}).get(nbr_id, SIZE) / SIZE,
                        float(np.sum(ent[curr]   > 0)) / SIZE,
                        float(np.sum(ent[dst_id] > 0)) / SIZE,
                    ], dtype=np.float32)
                    qmix_states.append((state_v, 0, reward, next_state_v, rf_vec))

        # ── Training ──────────────────────────────────────────────────────────
        if not INFERENCE_MODE:
            for ridx, t_list in req_transitions.items():
                if not t_list:
                    continue   # request never got a hop — skip
                rs = self.requestState[ridx]
                if not rs[5]:
                    # Request dropped: attach R_TTL to last real transition
                    s, a, r, ns, _ = t_list[-1]
                    t_list[-1] = (s, a, r + R_TTL, ns, True)
                self.agent.single_replay.push_sequence(t_list)
                self._push_ctr += len(t_list)

            if self.use_qmix and qmix_states:
                ep_states  = [t[0] for t in qmix_states]
                ep_acts    = [t[1] for t in qmix_states]
                ep_rews    = [t[2] for t in qmix_states]
                ep_nstates = [t[3] for t in qmix_states]
                ep_rf      = [t[4] for t in qmix_states]
                self.agent.qmix_replay.push(
                    ep_states, ep_acts, ep_rews, ep_nstates,
                    ep_rf, gs_vec, gs_vec, done=False)

            while self._push_ctr >= STEP_BETWEEN_TRAIN:
                self._push_ctr -= STEP_BETWEEN_TRAIN
                if self.use_qmix:
                    loss = self.agent.train_qmix()
                else:
                    loss = self.agent.train_dqn()
                if loss is not None:
                    self._loss_log.append(loss)

        # ── Cleanup: drop all remaining requests (no carryover) ───────────────
        self.requests     = []
        self.requestState = []

        wall_ms = (time.perf_counter() - t0) * 1000.0
        if self.timeSlot % LOG_EVERY == 0:
            avg_loss_str = ""
            if self._loss_log:
                avg_loss = float(np.mean(self._loss_log[-50:]))
                avg_loss_str = f"  loss={avg_loss:.5f}  eps={eps:.3f}"
            print(f"[{self.name}] ts={self.timeSlot:5d}"
                  f"  succ={success_req:3d}"
                  f"{avg_loss_str}"
                  f"  wall={wall_ms:.1f}ms")

        self.result.successfulRequest  += success_req
        self.result.successfulRequestPerRound.append(success_req)
        self.result.entanglementPerRound.append(success_req)
        avg_fid = total_fid / max(success_fid, 1)
        self.result.fidelityPerRound.append(avg_fid)
        self.result.rewardPerRound.append(float(success_req))

        self.printResult()
        return self.result

    # ── Routing strategies ────────────────────────────────────────────────────

    def _route_seq(self, pending, eps, ent, dist, req_dens, bfs_dists,
                   hop_counts, avail_cap):
        """
        Sequential: each request consumes its link before the next decides.
        Returns list of (ridx, chosen_nbr_id, state_v).
        """
        chosen    = []
        local_cap = dict(avail_cap)

        for ridx, rs, curr, dst_id, visited, nbrs in pending:
            avail_nbrs = [n for n in nbrs
                          if local_cap.get((min(curr, n), max(curr, n)), 0) > 0]
            if not avail_nbrs:
                continue
            nbr, state_v = self._pick_action(
                ent, dist, req_dens, bfs_dists,
                curr, dst_id, avail_nbrs, rs, hop_counts[ridx], eps)
            lk = (min(curr, nbr), max(curr, nbr))
            local_cap[lk] = max(0, local_cap.get(lk, 0) - 1)
            chosen.append((ridx, nbr, state_v))

        avail_cap.update(local_cap)
        return chosen

    def _route_parallel(self, pending, eps, ent, dist, req_dens, bfs_dists,
                         hop_counts, avail_cap, deconflict: bool):
        """
        Parallel: all requests choose simultaneously; optional b-matching deconflict.
        Returns list of (ridx, chosen_nbr_id, state_v).
        """
        raw = []
        for ridx, rs, curr, dst_id, visited, nbrs in pending:
            nbr, state_v = self._pick_action(
                ent, dist, req_dens, bfs_dists,
                curr, dst_id, nbrs, rs, hop_counts[ridx], eps)
            raw.append((ridx, curr, nbr, state_v))

        if not deconflict:
            chosen    = []
            local_cap = dict(avail_cap)
            for ridx, curr, nbr, state_v in raw:
                lk = (min(curr, nbr), max(curr, nbr))
                if local_cap.get(lk, 0) > 0:
                    local_cap[lk] -= 1
                    chosen.append((ridx, nbr, state_v))
            avail_cap.update(local_cap)
            return chosen

        # Guard / Hive: greedy b-matching
        if not raw:
            return []

        # Use each request's state_v (already computed for chosen nbr) as score input
        batch_rows = [state_v for (_, _, _, state_v) in raw]
        batch_t    = torch.tensor(np.stack(batch_rows), dtype=torch.float32)
        self.agent.qnet.eval()
        with torch.no_grad():
            scores_np = self.agent.qnet.net(batch_t).squeeze(1).numpy()

        candidates = [(ridx, curr, nbr, float(scores_np[i]))
                      for i, (ridx, curr, nbr, _) in enumerate(raw)]
        matched    = bmatching_nodes(candidates, dict(avail_cap))

        for ridx, curr, nbr, _ in candidates:
            if ridx in matched:
                lk = (min(curr, nbr), max(curr, nbr))
                avail_cap[lk] = max(0, avail_cap.get(lk, 0) - 1)

        raw_dict = {ridx: (curr, nbr, state_v) for ridx, curr, nbr, state_v in raw}
        chosen   = []
        for ridx, nbr_id in matched.items():
            _, _, state_v = raw_dict[ridx]
            chosen.append((ridx, nbr_id, state_v))

        return chosen

    def _pick_action(self, ent, dist, req_dens, bfs_dists,
                     curr: int, dst_id: int, nbrs: list,
                     rs, hops_used: int, eps: float):
        """
        Pick a neighbor via ε-greedy Q-scoring.
        Returns (nbr_id, state_v).
        """
        if dst_id in nbrs:
            nbr = dst_id
        elif _rng.random() < eps:
            nbr = _rng.choice(nbrs)
        else:
            state_vecs = np.stack([
                self._edge_state(
                    ent, dist, req_dens, bfs_dists,
                    curr, dst_id, n, float(dist[curr][n]),
                    float(rs[6]), hops_used)
                for n in nbrs
            ])
            scores  = self.agent.score_neighbors_v(state_vecs)
            best_i  = int(np.argmax(scores))
            nbr     = nbrs[best_i]
            state_v = state_vecs[best_i]
            return nbr, state_v

        state_v = self._edge_state(
            ent, dist, req_dens, bfs_dists,
            curr, dst_id, nbr, float(dist[curr][nbr]),
            float(rs[6]), hops_used)
        return nbr, state_v

    # ── AlgorithmBase bookkeeping ─────────────────────────────────────────────

    def printResult(self):
        self.topo.clearAllEntanglements()
        if self.totalRequest > 0:
            self.result.waitingTime = self.totalWaiting / self.totalRequest
            self.result.usedQubits  = self.totalQubits  / self.totalRequest
        self.result.remainRequestPerRound.append(len(self.requests))


# ── Named variants ─────────────────────────────────────────────────────────────

class QuRA_DQRL_DIST(QuRA_Local_v2):
    def __init__(self, topo, param=None, name='QuRA_Seq_DIST'):
        super().__init__(topo, param, name, variant='seq')


class QuRA_Flock_DIST(QuRA_Local_v2):
    def __init__(self, topo, param=None, name='QuRA_Flock_DIST'):
        super().__init__(topo, param, name, variant='flock')


class QuRA_Guard_DIST(QuRA_Local_v2):
    def __init__(self, topo, param=None, name='QuRA_Guard_DIST'):
        super().__init__(topo, param, name, variant='guard')


class QuRA_Hive_DIST(QuRA_Local_v2):
    def __init__(self, topo, param=None, name='QuRA_Hive_DIST'):
        super().__init__(topo, param, name, variant='hive')


# ── ShortestPath baseline (unchanged) ─────────────────────────────────────────

import networkx as nx

class ShortestPath(AlgorithmBase):
    """BFS over per-timeslot entanglement graph. No learning."""

    def __init__(self, topo, param=None, name='ShortestPath'):
        super().__init__(topo)
        self.name             = name
        self.requests         = []
        self.totalRequest     = 0
        self.totalWaiting     = 0
        self.totalQubits      = 0
        self.requestState     = []

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

    def p4(self):
        keep = [i for i, r in enumerate(self.requests)
                if self.timeSlot - r[2] < TTL_W]
        self.requests     = [self.requests[i]     for i in keep]
        self.requestState = [self.requestState[i] for i in keep]

        G = nx.Graph()
        G.add_nodes_from(range(SIZE))
        ent_avail: dict = {}
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                u, v = link.n1.id, link.n2.id
                G.add_edge(u, v)
                lk = (min(u, v), max(u, v))
                ent_avail[lk] = ent_avail.get(lk, 0) + 1

        success_req = 0
        for rs in self.requestState:
            if rs[5]:
                continue
            curr   = int(rs[2])
            dst_id = rs[1].id
            try:
                path = nx.shortest_path(G, curr, dst_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            routed = True
            for nhop in path[1:]:
                lk = (min(curr, nhop), max(curr, nhop))
                if ent_avail.get(lk, 0) < 1:
                    routed = False
                    break
                ent_avail[lk] -= 1
                curr = nhop

            if routed and curr == dst_id:
                rs[5] = True
                success_req += 1

        self.requests     = [r for i, r in enumerate(self.requests)
                             if not self.requestState[i][5]]
        self.requestState = [s for s in self.requestState if not s[5]]

        self.result.successfulRequest += success_req
        self.result.successfulRequestPerRound.append(success_req)
        self.result.entanglementPerRound.append(success_req)
        self.result.fidelityPerRound.append(0)
        self.result.rewardPerRound.append(float(success_req))
        self.printResult()
        return self.result

    def printResult(self):
        self.topo.clearAllEntanglements()
        if self.totalRequest > 0:
            self.result.waitingTime = self.totalWaiting / self.totalRequest
            self.result.usedQubits  = self.totalQubits  / self.totalRequest
        self.result.remainRequestPerRound.append(len(self.requests))
