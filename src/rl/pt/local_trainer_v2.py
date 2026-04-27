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
from __future__ import annotations

import os
import sys
import random as _rng
import time

import numpy as np
import torch

# ── Path setup (mirrors local_trainer.py) ────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))
_src_dir  = os.path.normpath(os.path.join(_here, '../..'))
_algo_dir = os.path.join(_src_dir, 'quantum', 'algorithm')
for _p in [_algo_dir, _src_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AlgorithmBase import AlgorithmBase

from .agent_v2  import DQRLAgentV2, _compact_global, GLOBAL_STATE_DIM
from .encoder   import build_graph_tensors, OUT_DIM
from .matching  import bmatching_nodes
from .replay_v2 import STEP_BETWEEN_TRAIN, MIN_REPLAY, TRAINING_MODE, GAMMA

# ── Constants ─────────────────────────────────────────────────────────────────
SIZE           = int(os.environ.get("SIZE", "100"))
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"
MODEL_DIR      = os.environ.get("MODEL_DIR", "/tmp/qrouting_model")

TTL_W          = int(os.environ.get("TTL_W", "75"))   # routing window W
F_MIN          = float(os.environ.get("F_MIN", "0.7"))

# Potential-based shaping coefficient (Ng et al.)
# Φ(curr, dst) = -dist(curr, dst) / MAX_HOPS
_SHAPING_COEFF = float(os.environ.get("SHAPING_COEFF", "0.5"))
_MAX_HOPS      = float(SIZE)   # conservative upper bound

# Reward magnitudes
R_SUCCESS      = 10.0
R_FAIL_FMIN    = -5.0   # reached dst but fidelity below F_min
R_HOP          = -0.01  # small step cost (encourages shorter paths)
R_TTL          = -1.0   # request timed out

# Training log interval: env-override or ~10 checkpoints across TTIME
_ttime_env = int(os.environ.get("TTIME", "10000"))
LOG_EVERY  = int(os.environ.get("LOG_EVERY", str(max(1, _ttime_env // 10))))


def _werner_swap(f1: float, f2: float) -> float:
    """Werner-state fidelity after a single entanglement swap."""
    return f1 * f2 + (1.0 - f1) * (1.0 - f2) / 3.0


def _epsilon(ts: int) -> float:
    """Linear epsilon decay schedule (same logic as helpers.py but self-contained)."""
    if INFERENCE_MODE:
        return 0.0
    if TRAINING_MODE == "paper":
        s, e = 3000, 8000
    elif TRAINING_MODE == "mid":
        s, e = 200, 1500
    else:   # smoke
        s, e = 10, 40
    if ts < s:
        return 1.0
    if ts >= e:
        return 0.0
    return max(0.0, 1.0 - (ts - s) / (e - s))


class QuRA_Local_v2(AlgorithmBase):
    """
    Single class implementing all four QuRA-v2 variants.

    variant : 'seq' | 'flock' | 'guard' | 'hive'
      seq   — iterate requests one-by-one; greedy hop-by-hop
      flock — all requests choose in parallel; no deconfliction
      guard — flock + greedy b-matching after each parallel hop
      hive  — guard + QMIX joint training
    """

    def __init__(self, topo, param=None, name='QuRA_v2',
                 variant: str = 'guard'):
        super().__init__(topo)
        self.name     = name
        self.variant  = variant
        self.use_qmix = (variant == 'hive')

        self.requests      = []   # list of (src_node, dst_node, arrival_ts)
        self.requestState  = []   # list of [src, dst, curr_id, path_set, idx, done]
        self.totalRequest  = 0
        self.totalWaiting  = 0
        self.totalQubits   = 0
        self._push_ctr     = 0
        self._loss_log: list[float] = []

        self._suppress_base_log = True   # AlgorithmBase.work() skips its per-slot print
        self.agent = DQRLAgentV2(pid=0, num_nodes=SIZE, use_qmix=self.use_qmix)
        self._model_path = os.path.join(
            MODEL_DIR, f"{name.lower().replace(' ', '_').replace('-', '_')}.pt.gz")

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

        # Cap queue: at most 5× per-round load (prevents unbounded growth)
        _max_q = max(self.topo.numOfRequestPerRound * 5, 50)
        if len(self.requests) > _max_q:
            self.requests = self.requests[-_max_q:]

        self.requestState = []
        for idx, (src, dst, _) in enumerate(self.requests):
            self.requestState.append(
                [src, dst, src.id, frozenset({src.id}), idx, False, 1.0])
            # fields: [src_node, dst_node, curr_id, visited, idx, done, fid_accum]

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaiting += len(self.requests)
        self.result.idleTime += len(self.requests)
        # Always count as an active timeslot (avoids /0 in AlgorithmBase)
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
        """Single link-pass → (ent_matrix, dist_matrix, link_capacity_dict)."""
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

    # ── Edge-score state vector ───────────────────────────────────────────────

    @staticmethod
    def _edge_state(H: torch.Tensor, curr: int, dst: int,
                    nbr: int, fid_uv: float, fid_so_far: float,
                    hops_used: int) -> np.ndarray:
        """
        Build 3*D+3 edge-score input vector for one (request, candidate) pair.
        D = OUT_DIM = 32  →  total = 99 dims (< 500 ✓)
        """
        D = OUT_DIM
        x = np.empty(3 * D + 3, dtype=np.float32)
        x[:D]        = H[curr].detach().numpy()
        x[D:2*D]     = H[dst].detach().numpy()
        x[2*D:3*D]   = H[nbr].detach().numpy()
        x[3*D]       = fid_uv
        x[3*D + 1]   = fid_so_far
        x[3*D + 2]   = min(hops_used / TTL_W, 1.0)
        return x

    # ── Potential-based shaping ───────────────────────────────────────────────

    @staticmethod
    def _precompute_bfs(G_nx, dsts: set) -> dict:
        """
        BFS distance from each destination to all reachable nodes.
        Returns {dst: {node: distance}}.  Done once per timeslot.
        """
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
        """Φ(s) = -d(curr, dst)/MAX_HOPS using precomputed BFS distances."""
        if curr == dst:
            return 0.0
        d = bfs.get(dst, {}).get(curr, _MAX_HOPS)
        return -d / _MAX_HOPS

    # ── p4 routing loop ───────────────────────────────────────────────────────

    def p4(self):
        t0 = time.perf_counter()

        # TTL filter: drop requests older than W timeslots
        keep = [i for i, r in enumerate(self.requests)
                if self.timeSlot - r[2] < TTL_W]
        self.requests     = [self.requests[i]     for i in keep]
        self.requestState = [self.requestState[i] for i in keep]

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
        nf, af, ac      = build_graph_tensors(ent, dist, req_dens, SIZE)

        # Encode graph — no_grad during inference; training path keeps grad enabled
        H = self.agent.encode(nf, af, ac, training=False)

        # Global state for QMIX
        gs_vec = _compact_global(H).detach().numpy()   # (2D,)

        eps          = _epsilon(self.timeSlot)
        success_req  = 0
        success_fid  = 0
        total_fid    = 0.0
        transitions  = []   # (state_vec, nbr_rel_idx, reward, next_state_vec, done)
        qmix_req_feats = []  # per-request: (state_vec, action, reward, next_state_vec)
        qmix_rf_vecs   = []  # per-request: concat(emb_curr, emb_dst) for QMixerV2

        # Mutable capacity for Guard/Hive deconfliction (copy of cap)
        avail_cap = dict(cap)

        # Build networkx graph + precompute BFS distances once per timeslot
        import networkx as nx
        G_nx = nx.Graph()
        G_nx.add_nodes_from(range(SIZE))
        for (u, v), c in cap.items():
            if c > 0:
                G_nx.add_edge(u, v)
        unique_dsts = {int(rs[1].id) for rs in self.requestState if not rs[5]}
        _bfs = self._precompute_bfs(G_nx, unique_dsts)

        # ── Routing (all variants share this loop, differ in hop_strategy) ───
        MAX_HOPS_PER_REQ = min(15, TTL_W)

        # Per-request hop state (hop count per active request)
        hop_counts = [0] * len(self.requestState)

        for _hop in range(MAX_HOPS_PER_REQ):
            # Collect active requests with valid moves
            pending = []
            for ridx, rs in enumerate(self.requestState):
                if rs[5]:
                    continue
                curr     = int(rs[2])
                dst_id   = rs[1].id
                visited  = rs[3]
                nbrs     = [n for n in np.nonzero(ent[curr])[0]
                            if n not in visited and n != curr]
                if nbrs:
                    pending.append((ridx, rs, curr, dst_id, visited, nbrs))

            if not pending:
                break

            if self.variant == 'seq':
                chosen = self._route_seq(H, pending, eps, ent, dist, hop_counts, avail_cap)
            else:
                chosen = self._route_parallel(H, pending, eps, ent, dist, hop_counts, avail_cap,
                                              deconflict=(self.variant in ('guard', 'hive')))

            # Apply chosen actions
            for ridx, nbr_id, state_v, fid_uv_val in chosen:
                rs = self.requestState[ridx]
                curr   = int(rs[2])
                dst_id = rs[1].id

                lk = (min(curr, nbr_id), max(curr, nbr_id))

                # Werner-swap fidelity update
                f_hop          = float(dist[curr][nbr_id])
                f_old          = float(rs[6])
                f_new          = _werner_swap(f_old, f_hop)
                rs[6]          = f_new

                # Consume entanglement (update ent and avail_cap)
                ent[curr][nbr_id]  = max(0.0, ent[curr][nbr_id]  - 1)
                ent[nbr_id][curr]  = max(0.0, ent[nbr_id][curr]  - 1)
                avail_cap[lk]      = max(0, avail_cap.get(lk, 0) - 1)

                rs[2]           = nbr_id
                rs[3]           = rs[3] | {nbr_id}
                hop_counts[ridx]+= 1
                done            = (nbr_id == dst_id)

                # Potential-based shaped reward (precomputed BFS distances)
                phi_s   = self._phi_cache(_bfs, curr,   dst_id)
                phi_ns  = self._phi_cache(_bfs, nbr_id, dst_id)
                shaping = _SHAPING_COEFF * (GAMMA * phi_ns - phi_s)

                if done:
                    rs[5] = True
                    if f_new >= F_MIN:
                        raw_r = R_SUCCESS
                        success_req += 1
                        success_fid += 1
                        total_fid   += f_new
                    else:
                        raw_r = R_FAIL_FMIN   # reached dst but fidelity too low
                else:
                    raw_r = R_HOP

                reward = raw_r + shaping

                # Next state for the edge that was taken
                next_nbrs = [n for n in np.nonzero(ent[nbr_id])[0]
                             if n not in rs[3] and n != nbr_id]
                if next_nbrs:
                    next_state_v = self._edge_state(
                        H, nbr_id, dst_id, next_nbrs[0],
                        float(dist[nbr_id][next_nbrs[0]]),
                        float(rs[6]), hop_counts[ridx])
                else:
                    next_state_v = state_v   # terminal / no moves

                transitions.append((state_v, 0, reward, next_state_v, done))
                qmix_req_feats.append((state_v, 0, reward, next_state_v))
                qmix_rf_vecs.append(
                    np.concatenate([H[curr].detach().numpy(),
                                    H[dst_id].detach().numpy()]))

        # TTL expiry: penalise requests that ran out of time this timeslot
        for ridx, rs in enumerate(self.requestState):
            if not rs[5]:
                arr_ts = self.requests[ridx][2]
                if self.timeSlot - arr_ts >= TTL_W - 1:
                    transitions.append((
                        np.zeros(3 * OUT_DIM + 3, dtype=np.float32),
                        0, R_TTL,
                        np.zeros(3 * OUT_DIM + 3, dtype=np.float32),
                        True))

        # ── Training ──────────────────────────────────────────────────────────
        if not INFERENCE_MODE and transitions:
            for (sv, a, r, nsv, d) in transitions:
                self.agent.remember(sv, a, r, nsv, d)
                self._push_ctr += 1

            if self.use_qmix and qmix_req_feats:
                ep_states = [t[0] for t in qmix_req_feats]
                ep_acts   = [t[1] for t in qmix_req_feats]
                ep_rews   = [t[2] for t in qmix_req_feats]
                ep_nstates= [t[3] for t in qmix_req_feats]
                self.agent.qmix_replay.push(
                    ep_states, ep_acts, ep_rews, ep_nstates,
                    qmix_rf_vecs, gs_vec, gs_vec,
                    done=False,
                )

            if self._push_ctr >= STEP_BETWEEN_TRAIN:
                self._push_ctr = 0
                if self.use_qmix:
                    loss = self.agent.train_qmix()
                else:
                    loss = self.agent.train_dqn(nf, af, ac)
                if loss is not None:
                    self._loss_log.append(loss)

        # ── Cleanup ───────────────────────────────────────────────────────────
        self.requests     = [r for i, r in enumerate(self.requests)
                             if not self.requestState[i][5]]
        self.requestState = [s for s in self.requestState if not s[5]]

        wall_ms = (time.perf_counter() - t0) * 1000.0
        if self.timeSlot % LOG_EVERY == 0:
            avg_loss_str = ""
            if self._loss_log:
                avg_loss = float(np.mean(self._loss_log[-50:]))
                avg_loss_str = f"  loss={avg_loss:.5f}  eps={eps:.3f}"
            print(f"[{self.name}] ts={self.timeSlot:5d}"
                  f"  succ={success_req:3d}"
                  f"  remain={len(self.requests):4d}"
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

    def _route_seq(self, H, pending, eps, ent, dist, hop_counts, avail_cap):
        """
        Sequential: process one request at a time; each request immediately
        consumes its link before the next request decides.
        Returns list of (ridx, chosen_nbr_id, state_v, fid_uv).
        """
        chosen = []
        local_cap = dict(avail_cap)   # local copy to track within this hop step

        for ridx, rs, curr, dst_id, visited, nbrs in pending:
            # Filter by available capacity
            avail_nbrs = [n for n in nbrs
                          if local_cap.get((min(curr, n), max(curr, n)), 0) > 0]
            if not avail_nbrs:
                continue

            nbr, state_v, fid_uv_val = self._pick_action(
                H, curr, dst_id, avail_nbrs, dist, rs, hop_counts[ridx], eps)

            lk = (min(curr, nbr), max(curr, nbr))
            local_cap[lk] = max(0, local_cap.get(lk, 0) - 1)
            chosen.append((ridx, nbr, state_v, fid_uv_val))

        # Update avail_cap in-place to reflect seq consumption
        avail_cap.update(local_cap)
        return chosen

    def _route_parallel(self, H, pending, eps, ent, dist, hop_counts, avail_cap,
                         deconflict: bool):
        """
        Parallel: all requests choose simultaneously; optional b-matching deconflict.
        """
        # Step 1: each request selects its preferred neighbor
        raw = []
        for ridx, rs, curr, dst_id, visited, nbrs in pending:
            nbr, state_v, fid_uv_val = self._pick_action(
                H, curr, dst_id, nbrs, dist, rs, hop_counts[ridx], eps)
            raw.append((ridx, curr, nbr, state_v, fid_uv_val))

        if not deconflict:
            # Flock: accept all choices even if they conflict
            # Filter by actual capacity (first-come-first-served by list order)
            chosen = []
            local_cap = dict(avail_cap)
            for ridx, curr, nbr, state_v, fid_uv_val in raw:
                lk = (min(curr, nbr), max(curr, nbr))
                if local_cap.get(lk, 0) > 0:
                    local_cap[lk] -= 1
                    chosen.append((ridx, nbr, state_v, fid_uv_val))
            avail_cap.update(local_cap)
            return chosen

        # Guard / Hive: greedy b-matching
        # Batch ALL (ridx, curr, nbr) pairs into a single Q-net forward pass
        # instead of one call per pair (removes ~5000 individual torch calls/slot).
        if raw:
            D = OUT_DIM
            H_np = H.detach().numpy()   # (N, D) numpy once
            batch_rows = []
            for ridx, curr, nbr, state_v, fid_uv_val in raw:
                rs2       = self.requestState[ridx]
                dst_id2   = int(rs2[1].id)
                fid_sf    = float(rs2[6])
                hops_f    = min(hop_counts[ridx] / TTL_W, 1.0)
                fid_uv2   = float(dist[curr][nbr])
                row = np.empty(3 * D + 3, dtype=np.float32)
                row[:D]      = H_np[curr]
                row[D:2*D]   = H_np[dst_id2]
                row[2*D:3*D] = H_np[nbr]
                row[3*D]     = fid_uv2
                row[3*D+1]   = fid_sf
                row[3*D+2]   = hops_f
                batch_rows.append(row)
            batch_t = torch.tensor(np.stack(batch_rows), dtype=torch.float32)
            self.agent.qnet.eval()
            with torch.no_grad():
                scores_np = self.agent.qnet.net(batch_t).squeeze(1).numpy()
            candidates = [(ridx, curr, nbr, float(scores_np[i]))
                          for i, (ridx, curr, nbr, _, _) in enumerate(raw)]
        else:
            candidates = []

        matched = bmatching_nodes(candidates, dict(avail_cap))

        # Update avail_cap
        for ridx, curr, nbr, _ in candidates:
            if ridx in matched:
                lk = (min(curr, nbr), max(curr, nbr))
                avail_cap[lk] = max(0, avail_cap.get(lk, 0) - 1)

        chosen = []
        raw_dict = {ridx: (curr, nbr, state_v, fid_uv_val)
                    for ridx, curr, nbr, state_v, fid_uv_val in raw}
        for ridx, nbr_id in matched.items():
            curr, _, state_v, fid_uv_val = raw_dict[ridx]
            chosen.append((ridx, nbr_id, state_v, fid_uv_val))

        return chosen

    def _pick_action(self, H, curr, dst_id, nbrs, dist, rs, hops_used, eps):
        """Pick a neighbor via ε-greedy Q-scoring. Returns (nbr, state_v, fid_uv)."""
        # Always go direct if destination is reachable
        if dst_id in nbrs:
            nbr = dst_id
        elif _rng.random() < eps:
            nbr = _rng.choice(nbrs)
        else:
            fid_uv = [float(dist[curr][n]) for n in nbrs]
            scores  = self.agent.score_neighbors(
                H, curr, dst_id, nbrs, fid_uv,
                float(rs[6]), min(hops_used / TTL_W, 1.0))
            nbr = nbrs[int(np.argmax(scores))]

        fid_uv_val = float(dist[curr][nbr])
        state_v    = self._edge_state(
            H, curr, dst_id, nbr, fid_uv_val, float(rs[6]), hops_used)
        return nbr, state_v, fid_uv_val

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


# ── ShortestPath baseline (unchanged from v1) ──────────────────────────────────

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
