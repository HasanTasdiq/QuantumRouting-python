"""
QuRA-v2 local trainer — all four variants share one architecture.

Variants (controlled by `variant` param):
  seq   — sequential single-request routing, no coordination
  flock — parallel routing, no conflict resolution
  guard — parallel routing + greedy b-matching conflict resolution
  hive  — guard + MAPPO centralized actor-critic training

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
from .qnet_v2   import STATE_DIM, MAPPO_GLOBAL_DIM, GNN_NODE_DIM

# ── Constants ─────────────────────────────────────────────────────────────────
SIZE           = int(os.environ.get("SIZE", "100"))
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"
MODEL_DIR      = os.environ.get("MODEL_DIR", "/tmp/qrouting_model")

TTL_W = int(os.environ.get("TTL_W", "75"))
F_MIN = float(os.environ.get("F_MIN", "0.7"))

_SHAPING_COEFF = float(os.environ.get("SHAPING_COEFF", "0.1"))
_MAX_HOPS      = float(SIZE)

R_SUCCESS   = float(os.environ.get("R_SUCCESS", "10.0"))
R_FAIL_FMIN = float(os.environ.get("R_FAIL_FMIN", "-2.0"))
R_HOP       = float(os.environ.get("R_HOP", "-0.01"))
R_TTL       = float(os.environ.get("R_TTL", "-0.05"))
R_CONFLICT  = float(os.environ.get("R_CONFLICT", "-1.5"))
TEAM_ALPHA  = float(os.environ.get("TEAM_ALPHA", "0.1"))

_ttime_env = int(os.environ.get("TTIME", "10000"))
LOG_EVERY  = int(os.environ.get("LOG_EVERY", str(max(1, _ttime_env // 10))))
_EPS_MIN_OVERRIDE = os.environ.get("EPS_MIN")
_EPS_WARMUP_OVERRIDE = os.environ.get("EPS_WARMUP")
_EPS_DECAY_END_OVERRIDE = os.environ.get("EPS_DECAY_END")
MAPPO_FORCE_DEST = os.environ.get("MAPPO_FORCE_DEST", "0") == "1"
MAPPO_EPS_GREEDY = os.environ.get("MAPPO_EPS_GREEDY", "0") == "1"
HIVE_ALL_CANDIDATES = os.environ.get("HIVE_ALL_CANDIDATES", "1") == "1"
HIVE_BMATCH_NOISE = float(os.environ.get("HIVE_BMATCH_NOISE", "0.0"))
HIVE_AUX_TEMP = float(os.environ.get("HIVE_AUX_TEMP", "0.35"))
SAVE_BEST_POLICY = os.environ.get("SAVE_BEST_POLICY", "1") == "1"
BEST_WINDOW = int(os.environ.get("BEST_WINDOW", "500"))
BEST_WARMUP = int(os.environ.get("BEST_WARMUP", "500"))
LOAD_BEST_POLICY = os.environ.get("LOAD_BEST_POLICY", "1") == "1"
FLOCK_BATCH_SCORING  = os.environ.get("FLOCK_BATCH_SCORING",  "1") == "1"
GUARD_ALL_CANDIDATES = os.environ.get("GUARD_ALL_CANDIDATES", "1") == "1"
GUARD_BMATCH_NOISE   = float(os.environ.get("GUARD_BMATCH_NOISE", "0.0"))


def _variant_dqn_arch(variant: str) -> str | None:
    """Variant-specific DQN arch override; lets Flock bias toward throughput."""
    if variant == "flock":
        return os.environ.get("FLOCK_DQN_ARCH", os.environ.get("DQN_ARCH", "linear"))
    return os.environ.get("DQN_ARCH")


def _werner_swap(f1: float, f2: float) -> float:
    return f1 * f2 + (1.0 - f1) * (1.0 - f2) / 3.0


def _epsilon(ts: int) -> float:
    """Linear epsilon decay with optional env-controlled floor/schedule."""
    if INFERENCE_MODE:
        return 0.0
    if TRAINING_MODE == "long":
        s, e = 200_000, 800_000
    elif TRAINING_MODE == "paper":
        s, e = 4_000, 16_000
    elif TRAINING_MODE == "mid":
        s, e = 1_600, 6_400
    else:   # smoke
        s, e = 400, 1_600
    if _EPS_WARMUP_OVERRIDE is not None:
        s = int(_EPS_WARMUP_OVERRIDE)
    if _EPS_DECAY_END_OVERRIDE is not None:
        e = int(_EPS_DECAY_END_OVERRIDE)
    eps_min = float(_EPS_MIN_OVERRIDE) if _EPS_MIN_OVERRIDE is not None else 0.0
    if ts < s:
        return 1.0
    if ts >= e:
        return eps_min
    if e <= s:
        return eps_min
    frac = (ts - s) / (e - s)
    return max(eps_min, 1.0 - (1.0 - eps_min) * frac)


class QuRA_Local_v2(AlgorithmBase):
    """
    Single class implementing all four QuRA-v2 variants.

    variant : 'seq' | 'flock' | 'guard' | 'hive'
    """

    def __init__(self, topo, param=None, name='QuRA_v2', variant: str = 'guard'):
        super().__init__(topo)
        self.name     = name
        self.variant  = variant
        self.use_mappo = (variant == 'hive')
        self.use_qmix = self.use_mappo   # legacy bool name used by DQRLAgentV2

        self.requests      = []
        self.requestState  = []
        self.totalRequest  = 0
        self.totalWaiting  = 0
        self.totalQubits   = 0
        self._push_ctr     = 0
        self._loss_log: list[float] = []
        self._recent_slot_success: list[int] = []
        self._best_recent_rate = float("-inf")

        self._suppress_base_log = True
        self.agent = DQRLAgentV2(
            pid=0,
            num_nodes=SIZE,
            use_qmix=self.use_qmix,
            dqn_arch=_variant_dqn_arch(variant),
        )
        self._model_path = os.path.join(
            MODEL_DIR,
            f"{name.lower().replace(' ', '_').replace('-', '_')}.pt.gz")
        self._best_model_path = os.path.join(
            MODEL_DIR,
            f"{name.lower().replace(' ', '_').replace('-', '_')}_best.pt.gz")

        if INFERENCE_MODE:
            self._load_weights()

    # ── Weight I/O ────────────────────────────────────────────────────────────

    def _save_weights(self) -> None:
        self.agent.save_weights(self._model_path)
        print(f"[{self.name}] weights → {self._model_path}")

    def _load_weights(self) -> bool:
        load_path = self._best_model_path if (
            LOAD_BEST_POLICY and os.path.exists(self._best_model_path)
        ) else self._model_path
        ok = self.agent.load_weights(load_path)
        if ok:
            print(f"[{self.name}] weights loaded from {load_path}")
        else:
            print(f"[{self.name}] no checkpoint — random policy")
        return ok

    def _maybe_save_best(self, success_req: int) -> None:
        if INFERENCE_MODE or not SAVE_BEST_POLICY:
            return
        self._recent_slot_success.append(int(success_req))
        if len(self._recent_slot_success) > BEST_WINDOW:
            self._recent_slot_success = self._recent_slot_success[-BEST_WINDOW:]
        if self.timeSlot < BEST_WARMUP or len(self._recent_slot_success) < BEST_WINDOW:
            return
        curr_rate = float(np.mean(self._recent_slot_success)) / max(self.topo.numOfRequestPerRound, 1)
        if curr_rate > self._best_recent_rate + 1e-9:
            self._best_recent_rate = curr_rate
            self.agent.save_weights(self._best_model_path)
            print(f"[{self.name}] best → {self._best_model_path}  recent_rate={curr_rate:.4f}")

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
        n    = len(self.topo.nodes)
        ent  = np.zeros((n, n), dtype=np.float32)
        dist = np.zeros((n, n), dtype=np.float32)
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
        n    = len(self.topo.nodes)
        dens = np.zeros(n, dtype=np.float32)
        for rs in self.requestState:
            if not rs[5]:
                dens[int(rs[2])] += 1.0
        return dens

    def _compute_global_state(self, ent: np.ndarray,
                               dist: np.ndarray,
                               req_dens: np.ndarray,
                               pending: list | None = None,
                               bfs_dists: np.ndarray | None = None,
                               edge_demand: dict | None = None) -> np.ndarray:
        """
        MAPPO centralized critic state (28-dim):
        Base 8: [mean_deg, max_deg, mean_fid, req_load, edge_util, cap_mean, cap_std, active_node_frac]
        Dist 8: [deg_var, fid_p25, fid_p75, high_cong_frac, dens_std, cap_ratio, isolated_req_frac, total_cap_util]
        Pending 12: summary statistics over the full contested request frontier.
        """
        degrees     = np.sum(ent > 0, axis=1)
        link_fids   = dist[ent > 0]
        active_caps = ent[ent > 0]
        max_possible = max(float(len(self.topo.links)), 1.0)
        total_dens   = max(float(req_dens.sum()), 1.0)

        mean_deg  = float(degrees.mean()) / max(SIZE, 1)
        max_deg   = float(degrees.max())  / max(SIZE, 1)
        mean_fid  = float(link_fids.mean()) if link_fids.size > 0 else 0.0
        deg_std   = float(degrees.std())  / max(SIZE, 1)

        # Fidelity percentiles
        fid_p25 = float(np.percentile(link_fids, 25)) if link_fids.size > 0 else 0.0
        fid_p75 = float(np.percentile(link_fids, 75)) if link_fids.size > 0 else 0.0

        # High-congestion: fraction of nodes with degree > mean + std (normalised)
        mean_d_raw = float(degrees.mean())
        std_d_raw  = float(degrees.std()) + 1e-8
        high_cong_frac = float(np.mean(degrees > mean_d_raw + std_d_raw))

        # Request density spread
        dens_std = float(req_dens.std()) / total_dens

        # Cap ratio (max / mean) signals bottleneck edges
        cap_mean = float(active_caps.mean()) if active_caps.size > 0 else 0.0
        cap_max  = float(active_caps.max())  if active_caps.size > 0 else 0.0
        cap_ratio = (cap_max / max(cap_mean, 1e-8)) / max(SIZE, 1)

        # Fraction of active requests sitting at degree-0 node (isolated)
        isolated = 0.0
        active_count = 0
        for rs in self.requestState:
            if not rs[5]:
                active_count += 1
                if degrees[int(rs[2])] == 0:
                    isolated += 1.0
        isolated_req_frac = isolated / max(active_count, 1)

        # Total capacity utilisation (used caps vs max possible)
        total_cap_util = float(active_caps.sum()) / max(max_possible * SIZE, 1.0)

        base = np.array([
            mean_deg,
            max_deg,
            mean_fid,
            float(req_dens.sum())  / max(SIZE, 1),
            float(np.count_nonzero(ent > 0) / 2.0) / max_possible,
            cap_mean,
            float(active_caps.std()) if active_caps.size > 0 else 0.0,
            float(np.count_nonzero(degrees > 0)) / max(SIZE, 1),
            # distribution stats
            deg_std,
            fid_p25,
            fid_p75,
            high_cong_frac,
            dens_std,
            cap_ratio,
            isolated_req_frac,
            total_cap_util,
        ], dtype=np.float32)
        pending_summary = self._pending_summary(
            pending or [], ent, dist, bfs_dists, edge_demand or {}
        )
        return np.concatenate([base, pending_summary]).astype(np.float32)

    @staticmethod
    def _pending_summary(pending: list, ent: np.ndarray, dist: np.ndarray,
                         bfs_dists: np.ndarray | None, edge_demand: dict) -> np.ndarray:
        """Summarize the full pending frontier for the centralized critic."""
        if not pending:
            return np.zeros(12, dtype=np.float32)

        cand_counts = np.array([len(nbrs) for (_, _, _, _, _, nbrs) in pending], dtype=np.float32)
        total_options = max(float(cand_counts.sum()), 1.0)
        unique_edges = max(float(len(edge_demand)), 1.0)
        max_node_cap = max(float(ent.sum(axis=1).max()), 1.0)

        demand_over_cap = []
        overdemand = []
        best_progress = []
        best_fid_margin = []
        best_nbr_cap = []
        curr_degrees = []
        forced_frac = 0.0

        for _, rs, curr, dst_id, _, nbrs in pending:
            d_curr = float(bfs_dists[curr, dst_id]) if bfs_dists is not None else float(SIZE)
            curr_degrees.append(float(np.sum(ent[curr] > 0)) / max(SIZE, 1))
            if len(nbrs) <= 1:
                forced_frac += 1.0

            prog_scores = []
            fid_scores = []
            cap_scores = []
            for nbr in nbrs:
                lk = (min(curr, nbr), max(curr, nbr))
                cap = max(float(ent[curr][nbr]), 1.0)
                demand = float(edge_demand.get(lk, 0))
                demand_over_cap.append(demand / cap)
                overdemand.append(1.0 if demand > cap else 0.0)
                d_nbr = float(bfs_dists[nbr, dst_id]) if bfs_dists is not None else float(SIZE)
                prog_scores.append((d_curr - d_nbr) / max(SIZE, 1))
                fid_scores.append(_werner_swap(float(rs[6]), float(dist[curr][nbr])) - F_MIN)
                cap_scores.append(float(ent[nbr].sum()) / max_node_cap)

            best_progress.append(max(prog_scores) if prog_scores else 0.0)
            best_fid_margin.append(max(fid_scores) if fid_scores else 0.0)
            best_nbr_cap.append(max(cap_scores) if cap_scores else 0.0)

        return np.array([
            float(len(pending)) / max(SIZE, 1),
            float(cand_counts.mean()) / max(SIZE, 1),
            float(cand_counts.std()) / max(SIZE, 1),
            float(cand_counts.max()) / max(SIZE, 1),
            unique_edges / total_options,
            float(np.mean(demand_over_cap)) if demand_over_cap else 0.0,
            float(np.mean(overdemand)) if overdemand else 0.0,
            float(np.mean(best_progress)) if best_progress else 0.0,
            float(np.mean(best_fid_margin)) if best_fid_margin else 0.0,
            float(np.mean(best_nbr_cap)) if best_nbr_cap else 0.0,
            float(np.mean(curr_degrees)) if curr_degrees else 0.0,
            forced_frac / max(float(len(pending)), 1.0),
        ], dtype=np.float32)

    def _node_features(self, ent: np.ndarray,
                       dist: np.ndarray,
                       req_dens: np.ndarray) -> np.ndarray:
        """
        Per-node GNN input over the active entanglement topology.

        Features:
        [degree_norm, cap_sum_norm, mean_fidelity, max_fidelity,
         req_density_norm, active_flag, active_neighbor_req_mean,
         edge_capacity_std].

        Fully vectorized — no Python loop over nodes.
        """
        total_req  = max(float(req_dens.sum()), 1.0)
        active     = ent > 0                                       # (N, N) bool
        deg        = active.sum(axis=1).astype(np.float32)        # (N,)
        cap_sum    = ent.sum(axis=1).astype(np.float32)           # (N,)
        max_cap_sum = max(float(cap_sum.max()), 1.0)

        # Mean fidelity per node
        fid_sum  = (dist * active).sum(axis=1).astype(np.float32)
        mean_fid = fid_sum / np.maximum(deg, 1.0)

        # Max fidelity per node (0 where no neighbors)
        max_fid = np.where(active, dist, 0.0).max(axis=1).astype(np.float32)

        # Neighbor request-density mean via matrix multiply
        nbr_dens_sum  = active.astype(np.float32) @ req_dens.astype(np.float32)
        nbr_dens_mean = nbr_dens_sum / np.maximum(deg, 1.0) / total_req

        # Per-node edge-capacity std via E[X²] - E[X]²
        cap_mean_n = cap_sum / np.maximum(deg, 1.0)
        cap_sq_sum = (ent ** 2 * active).sum(axis=1).astype(np.float32)
        cap_var    = cap_sq_sum / np.maximum(deg, 1.0) - cap_mean_n ** 2
        cap_std    = np.sqrt(np.maximum(cap_var, 0.0))

        sz = max(SIZE, 1)
        return np.column_stack([
            deg / sz,
            cap_sum / max_cap_sum,
            mean_fid,
            max_fid,
            req_dens.astype(np.float32) / total_req,
            (deg > 0).astype(np.float32),
            nbr_dens_mean,
            cap_std,
        ]).astype(np.float32)

    @staticmethod
    def _edge_index(ent: np.ndarray) -> np.ndarray:
        """Directed edge index for the active entanglement graph."""
        src, dst = np.nonzero(ent > 0)
        if len(src) == 0:
            return np.zeros((2, 0), dtype=np.int64)
        return np.stack([src, dst]).astype(np.int64)

    # ── Edge-score state vector ───────────────────────────────────────────────

    @staticmethod
    def _edge_demand(pending: list) -> dict:
        """Count how many active requests can use each undirected edge this hop."""
        demand = {}
        for _, _, curr, _, _, nbrs in pending:
            for nbr in nbrs:
                lk = (min(curr, nbr), max(curr, nbr))
                demand[lk] = demand.get(lk, 0) + 1
        return demand

    @staticmethod
    def _edge_state(ent: np.ndarray, dist: np.ndarray,
                    req_dens: np.ndarray, bfs_dists: np.ndarray,
                    curr: int, dst: int, nbr: int,
                    fid_uv: float, fid_so_far: float,
                    hops_used: int, edge_demand: dict | None = None,
                    active_req_count: int = 1) -> np.ndarray:
        """
        Build STATE_DIM=29 edge-state vector.

        3 nodes × (norm_degree, norm_req_density, norm_bfs_to_dst) + fid_uv + fid_so_far + hops_frac

        BFS distance uses the graph at the START of the timeslot (precomputed).
        Degree and req_density are the live values (updated as links are consumed).
        """
        total_dens = max(float(req_dens.sum()), 1.0)

        def node_feat(nid: int) -> list:
            deg  = float(np.sum(ent[nid] > 0)) / max(SIZE, 1)
            dens = float(req_dens[nid]) / total_dens
            bfs  = float(bfs_dists[nid, dst]) / SIZE
            return [deg, dens, bfs]

        deg_curr = float(np.sum(ent[curr] > 0))
        deg_nbr = float(np.sum(ent[nbr] > 0))
        deg_dst = float(np.sum(ent[dst] > 0))
        d_curr = float(bfs_dists[curr, dst])
        d_nbr = float(bfs_dists[nbr, dst])
        pred_fid = _werner_swap(fid_so_far, fid_uv)
        active_fids = dist[ent > 0]
        active_edges = float(np.count_nonzero(ent > 0)) / 2.0
        edge_cap = float(ent[curr][nbr])
        max_cap = max(float(ent.max()), 1.0)
        hops_frac = min(float(hops_used) / TTL_W, 1.0)
        lk = (min(curr, nbr), max(curr, nbr))
        demand = float((edge_demand or {}).get(lk, 0))
        active_req = max(float(active_req_count), 1.0)
        nbr_total_cap = float(ent[nbr].sum())
        max_node_cap = max(float(ent.sum(axis=1).max()), 1.0)

        return np.array(
            node_feat(curr) + node_feat(dst) + node_feat(nbr) +
            [
                fid_uv,
                fid_so_far,
                hops_frac,
                edge_cap / max_cap,
                1.0 / (1.0 + min(deg_curr, deg_nbr)),
                pred_fid,
                pred_fid - F_MIN,
                (d_curr - d_nbr) / max(SIZE, 1),
                1.0 if d_nbr < d_curr else 0.0,
                1.0 if nbr == dst else 0.0,
                1.0 - hops_frac,
                min(float(req_dens.sum()) / max(SIZE, 1), 1.0),
                active_edges / max(float(len(ent) * max(len(ent) - 1, 1) / 2.0), 1.0),
                float(active_fids.mean()) if active_fids.size > 0 else 0.0,
                float(req_dens[nbr]) / total_dens,
                deg_dst / max(SIZE, 1),
                demand / active_req,
                min(demand / max(edge_cap, 1.0), 2.0),
                nbr_total_cap / max_node_cap,
                1.0 if demand > edge_cap else 0.0,
            ],
            dtype=np.float32
        )

    @staticmethod
    def _edge_state_batch(ent: np.ndarray, dist: np.ndarray,
                          req_dens: np.ndarray, bfs_mat: np.ndarray,
                          curr: int, dst: int, nbrs,
                          fid_so_far: float, hops_used: int,
                          edge_demand: dict | None = None,
                          active_req_count: int = 1) -> np.ndarray:
        """
        Vectorized version of _edge_state for K candidates at once.

        Replaces a Python loop over K calls to _edge_state with a single
        numpy array operation. Returns (K, STATE_DIM) float32 array.
        """
        nbr_arr   = np.asarray(nbrs, dtype=np.int64)
        K         = len(nbr_arr)
        N         = ent.shape[0]
        sz        = max(SIZE, 1)

        total_dens = max(float(req_dens.sum()), 1.0)
        active_req = max(float(active_req_count), 1.0)

        # Per-node precomputed features
        deg_raw  = (ent > 0).sum(axis=1).astype(np.float32)          # (N,)
        deg_norm = deg_raw / sz
        dens_norm = req_dens.astype(np.float32) / total_dens         # (N,)

        # BFS features for the three roles: curr, dst, nbr
        bfs_c = float(bfs_mat[curr, dst]) / sz
        bfs_d_val = float(bfs_mat[dst, dst]) / sz                    # always 0
        bfs_n = bfs_mat[nbr_arr, dst].astype(np.float32) / sz        # (K,)

        d_curr = float(bfs_mat[curr, dst])
        d_nbr  = bfs_mat[nbr_arr, dst].astype(np.float32)            # (K,)

        # Fidelity features
        fid_uv_arr    = dist[curr, nbr_arr].astype(np.float32)       # (K,)
        pred_fid      = fid_so_far * fid_uv_arr + (1.0 - fid_so_far) * (1.0 - fid_uv_arr) / 3.0

        hops_frac     = min(float(hops_used) / TTL_W, 1.0)

        # Edge capacity
        edge_cap = ent[curr, nbr_arr].astype(np.float32)             # (K,)
        max_cap  = max(float(ent.max()), 1.0)

        # Neighbor total capacity
        nbr_total_cap = ent[nbr_arr].sum(axis=1).astype(np.float32)  # (K,)
        max_node_cap  = max(float(ent.sum(axis=1).max()), 1.0)

        # Edge demand
        demand = np.zeros(K, dtype=np.float32)
        if edge_demand:
            for i, nbr in enumerate(nbr_arr):
                lk = (int(min(curr, nbr)), int(max(curr, nbr)))
                demand[i] = float(edge_demand.get(lk, 0))

        # Global features (scalar per slot)
        active_fids = dist[ent > 0]
        mean_fid    = float(active_fids.mean()) if active_fids.size > 0 else 0.0
        active_edges_frac = float(np.count_nonzero(ent > 0) / 2.0) / max(
            float(N * max(N - 1, 1) / 2.0), 1.0)
        global_load = min(float(req_dens.sum()) / sz, 1.0)

        # Assemble (K, 29) matrix using column_stack
        ones_K = np.ones(K, dtype=np.float32)
        return np.column_stack([
            # curr node (3)
            ones_K * deg_norm[curr],
            ones_K * dens_norm[curr],
            ones_K * bfs_c,
            # dst node (3)
            ones_K * deg_norm[dst],
            ones_K * dens_norm[dst],
            ones_K * bfs_d_val,
            # nbr node (3)
            deg_norm[nbr_arr],
            dens_norm[nbr_arr],
            bfs_n,
            # fidelity/hops (3)
            fid_uv_arr,
            ones_K * fid_so_far,
            ones_K * hops_frac,
            # edge capacity (1)
            edge_cap / max_cap,
            # bottleneck inverse degree (1)
            1.0 / (1.0 + np.minimum(deg_raw[curr], deg_raw[nbr_arr])),
            # predicted fidelity and margin (2)
            pred_fid,
            pred_fid - F_MIN,
            # BFS progress (2)
            (d_curr - d_nbr) / sz,
            (d_nbr < d_curr).astype(np.float32),
            # destination flags and TTL (3)
            (nbr_arr == dst).astype(np.float32),
            ones_K * (1.0 - hops_frac),
            ones_K * global_load,
            # topology features (3)
            ones_K * active_edges_frac,
            ones_K * mean_fid,
            # demand features (5)
            dens_norm[nbr_arr],
            ones_K * deg_norm[dst],
            demand / active_req,
            np.minimum(demand / np.maximum(edge_cap, 1.0), 2.0),
            nbr_total_cap / max_node_cap,
            (demand > edge_cap).astype(np.float32),
        ]).astype(np.float32)

    @staticmethod
    def _aux_policy_scores(ent: np.ndarray, dist: np.ndarray,
                           req_dens: np.ndarray, bfs_dists: np.ndarray,
                           curr: int, dst: int, nbrs: list[int],
                           fid_so_far: float, hops_used: int,
                           edge_demand: dict | None = None,
                           active_req_count: int = 1) -> np.ndarray:
        """Heuristic next-hop quality scores — fully vectorized over K candidates."""
        nbr_arr   = np.asarray(nbrs, dtype=np.int64)
        K         = len(nbr_arr)
        sz        = max(SIZE, 1)
        total_dens = max(float(req_dens.sum()), 1.0)
        active_req = max(float(active_req_count), 1.0)
        max_node_cap = max(float(ent.sum(axis=1).max()), 1.0)

        d_curr   = float(bfs_dists[curr, dst])
        d_nbr    = bfs_dists[nbr_arr, dst].astype(np.float32)
        progress = (d_curr - d_nbr) / sz

        fid_uv   = dist[curr, nbr_arr].astype(np.float32)
        pred_fid = fid_so_far * fid_uv + (1.0 - fid_so_far) * (1.0 - fid_uv) / 3.0
        fid_margin = pred_fid - F_MIN

        nbr_deg  = (ent[nbr_arr] > 0).sum(axis=1).astype(np.float32) / sz
        nbr_cap  = ent[nbr_arr].sum(axis=1).astype(np.float32) / max_node_cap
        nbr_load = req_dens[nbr_arr].astype(np.float32) / total_dens

        edge_cap = np.maximum(ent[curr, nbr_arr].astype(np.float32), 1.0)
        demand   = np.zeros(K, dtype=np.float32)
        if edge_demand:
            for i, nbr in enumerate(nbr_arr):
                lk = (int(min(curr, nbr)), int(max(curr, nbr)))
                demand[i] = float(edge_demand.get(lk, 0))

        demand_frac = demand / active_req
        overload    = np.maximum(demand - edge_cap, 0.0) / edge_cap
        slack       = np.minimum(edge_cap / np.maximum(demand, 1.0), 2.0) - 1.0
        hops_left   = 1.0 - min(float(hops_used) / TTL_W, 1.0)
        dest_flag   = (nbr_arr == dst).astype(np.float32)

        return (
            1.8  * progress
            + 1.4  * fid_margin
            + 0.8  * dest_flag
            + 0.35 * nbr_deg
            + 0.25 * nbr_cap
            + 0.20 * hops_left
            + 0.20 * slack
            - 0.45 * demand_frac
            - 0.55 * overload
            - 0.20 * nbr_load
        ).astype(np.float32)

    @staticmethod
    def _edge_state_and_aux_all_requests(
        pending: list,
        hop_counts: list,
        ent: np.ndarray,
        dist: np.ndarray,
        req_dens: np.ndarray,
        bfs_mat: np.ndarray,
        edge_demand: dict | None,
        active_req_count: int,
    ) -> tuple[list, list]:
        """
        Build (K_r, STATE_DIM) states and (K_r,) aux scores for ALL pending requests
        in ONE vectorized pass, sharing all hop-level precomputations.

        Replaces R separate calls to _edge_state_batch + _aux_policy_scores with a
        single numpy operation over sum_K = sum(K_r) candidates, precomputing
        deg_raw, max_cap, mean_fid, etc. exactly once per hop instead of R times.

        Returns (state_vecs_list, aux_scores_list) — each indexed by pending order.
        """
        if not pending:
            return [], []

        sz          = max(SIZE, 1)
        N           = ent.shape[0]
        total_dens  = max(float(req_dens.sum()), 1.0)
        active_req  = max(float(active_req_count), 1.0)

        # ── Hop-level precomputed: computed ONCE for all R requests ───────────
        deg_raw      = (ent > 0).sum(axis=1).astype(np.float32)         # (N,)
        deg_norm     = deg_raw / sz                                       # (N,)
        dens_norm    = req_dens.astype(np.float32) / total_dens          # (N,)
        max_cap      = max(float(ent.max()), 1.0)
        ent_row_sum  = ent.sum(axis=1)
        max_node_cap = max(float(ent_row_sum.max()), 1.0)
        active_fids  = dist[ent > 0]
        mean_fid     = float(active_fids.mean()) if active_fids.size > 0 else 0.0
        nz2          = float(np.count_nonzero(ent > 0)) / 2.0
        active_edges_frac = nz2 / max(float(N * (N - 1) / 2.0), 1.0)
        global_load  = min(float(req_dens.sum()) / sz, 1.0)

        # ── Build flat index arrays for all requests ──────────────────────────
        flat_curr_parts = []
        flat_dst_parts  = []
        flat_nbr_parts  = []
        flat_fid_parts  = []
        flat_hops_parts = []
        offsets         = []
        off             = 0

        for ridx, rs, curr, dst_id, visited, nbrs in pending:
            nbr_arr    = np.asarray(nbrs, dtype=np.int64)
            K_r        = len(nbr_arr)
            fid_so_far = float(rs[6])
            hops_frac  = min(float(hop_counts[ridx]) / TTL_W, 1.0)
            flat_curr_parts.append(np.full(K_r, curr,   dtype=np.int64))
            flat_dst_parts.append( np.full(K_r, dst_id, dtype=np.int64))
            flat_nbr_parts.append(nbr_arr)
            flat_fid_parts.append( np.full(K_r, fid_so_far, dtype=np.float32))
            flat_hops_parts.append(np.full(K_r, hops_frac,  dtype=np.float32))
            offsets.append((off, off + K_r))
            off += K_r

        flat_curr  = np.concatenate(flat_curr_parts)   # (sum_K,) int64
        flat_dst   = np.concatenate(flat_dst_parts)    # (sum_K,) int64
        flat_nbrs  = np.concatenate(flat_nbr_parts)    # (sum_K,) int64
        flat_fid   = np.concatenate(flat_fid_parts)    # (sum_K,) float32
        flat_hops  = np.concatenate(flat_hops_parts)   # (sum_K,) float32
        sum_K      = off

        # ── Vectorized feature lookup ─────────────────────────────────────────
        bfs_c    = bfs_mat[flat_curr, flat_dst].astype(np.float32) / sz
        bfs_n    = bfs_mat[flat_nbrs, flat_dst].astype(np.float32) / sz
        d_curr   = bfs_mat[flat_curr, flat_dst].astype(np.float32)
        d_nbr    = bfs_mat[flat_nbrs, flat_dst].astype(np.float32)
        fid_uv   = dist[flat_curr, flat_nbrs].astype(np.float32)
        pred_fid = flat_fid * fid_uv + (1.0 - flat_fid) * (1.0 - fid_uv) / 3.0
        edge_cap = ent[flat_curr, flat_nbrs].astype(np.float32)
        nbr_total = ent_row_sum[flat_nbrs].astype(np.float32)

        demand = np.zeros(sum_K, dtype=np.float32)
        if edge_demand:
            demand[:] = [
                float(edge_demand.get(
                    (int(min(c, n)), int(max(c, n))), 0.0))
                for c, n in zip(flat_curr.tolist(), flat_nbrs.tolist())
            ]

        # ── Assemble (sum_K, STATE_DIM=29) ───────────────────────────────────
        progress   = (d_curr - d_nbr) / sz
        fid_margin = pred_fid - float(F_MIN)
        safe_cap   = np.maximum(edge_cap, 1.0)

        all_states = np.column_stack([
            # curr (3)
            deg_norm[flat_curr],  dens_norm[flat_curr],  bfs_c,
            # dst (3)
            deg_norm[flat_dst],   dens_norm[flat_dst],   np.zeros(sum_K, dtype=np.float32),
            # nbr (3)
            deg_norm[flat_nbrs],  dens_norm[flat_nbrs],  bfs_n,
            # fidelity / hops (3)
            fid_uv, flat_fid, flat_hops,
            # edge capacity (1)
            edge_cap / max_cap,
            # bottleneck inv-degree (1)
            1.0 / (1.0 + np.minimum(deg_raw[flat_curr], deg_raw[flat_nbrs])),
            # predicted fidelity (2)
            pred_fid, fid_margin,
            # BFS progress (2)
            progress, (d_nbr < d_curr).astype(np.float32),
            # dest flags + TTL (3)
            (flat_nbrs == flat_dst).astype(np.float32),
            1.0 - flat_hops,
            np.full(sum_K, global_load, dtype=np.float32),
            # topology (2)
            np.full(sum_K, active_edges_frac, dtype=np.float32),
            np.full(sum_K, mean_fid, dtype=np.float32),
            # demand (5)
            dens_norm[flat_nbrs],
            deg_norm[flat_dst],
            demand / active_req,
            np.minimum(demand / safe_cap, 2.0),
            nbr_total / max_node_cap,
            (demand > edge_cap).astype(np.float32),
        ]).astype(np.float32)

        # ── Aux scores (shares flat arrays computed above) ────────────────────
        hops_left = 1.0 - flat_hops
        dest_flag = (flat_nbrs == flat_dst).astype(np.float32)
        overload  = np.maximum(demand - edge_cap, 0.0) / safe_cap
        slack     = np.minimum(edge_cap / np.maximum(demand, 1.0), 2.0) - 1.0
        all_aux = (
            1.8  * progress
            + 1.4  * fid_margin
            + 0.8  * dest_flag
            + 0.35 * deg_norm[flat_nbrs]
            + 0.25 * nbr_total / max_node_cap
            + 0.20 * hops_left
            + 0.20 * slack
            - 0.45 * demand / active_req
            - 0.55 * overload
            - 0.20 * dens_norm[flat_nbrs]
        ).astype(np.float32)

        state_vecs_list = [all_states[s:e] for s, e in offsets]
        aux_scores_list = [all_aux[s:e]    for s, e in offsets]
        return state_vecs_list, aux_scores_list

    @staticmethod
    def _aux_policy_target(scores: np.ndarray) -> np.ndarray:
        """Softmax over precomputed aux scores → teacher distribution."""
        if scores.size <= 1:
            return np.array([1.0], dtype=np.float32)
        s = scores / max(HIVE_AUX_TEMP, 1e-3)
        s = s - float(s.max())
        p = np.exp(s)
        return (p / max(float(p.sum()), 1e-8)).astype(np.float32)

    @staticmethod
    def _counterfactual_target(aux_scores: np.ndarray, action_i: int) -> float:
        """Chosen-vs-alternative score gap for dense counterfactual credit."""
        if aux_scores.size <= 1:
            return 0.0
        chosen = float(aux_scores[int(action_i)])
        others = np.delete(aux_scores.astype(np.float32), int(action_i))
        if others.size == 0:
            return 0.0
        return float(np.clip(chosen - float(others.mean()), -2.0, 2.0))

    # ── Potential-based shaping ───────────────────────────────────────────────

    @staticmethod
    def _precompute_bfs_mat(n: int, cap: dict) -> np.ndarray:
        """All-pairs unweighted shortest-path matrix via scipy BFS.

        Returns (N, N) float32 array where mat[u, v] = hop distance u→v.
        Unreachable pairs are set to N (our SIZE convention for infinity).
        """
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        if not cap:
            return np.full((n, n), float(n), dtype=np.float32)

        rows, cols, data = [], [], []
        for (u, v), c in cap.items():
            if c > 0:
                rows.extend([u, v])
                cols.extend([v, u])
                data.extend([1.0, 1.0])

        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        mat = shortest_path(graph, method='D', unweighted=True, directed=False)
        mat = np.where(np.isinf(mat), float(n), mat)
        return mat.astype(np.float32)

    @staticmethod
    def _phi_cache(bfs: np.ndarray, curr: int, dst: int) -> float:
        if curr == dst:
            return 0.0
        d = float(bfs[curr, dst])
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

        eps         = _epsilon(self.timeSlot)
        success_req = 0
        success_fid = 0
        total_fid   = 0.0

        # Per-request transition lists: ridx → [(s, a, r, ns, done), ...]
        req_transitions: dict[int, list] = {
            ridx: [] for ridx in range(len(self.requestState))
        }
        mappo_traj: dict[int, list] = {
            ridx: [] for ridx in range(len(self.requestState))
        }
        mappo_rejects: dict[int, list] = {
            ridx: [] for ridx in range(len(self.requestState))
        }
        # Guard all-candidates: ridx → [state_v, ...] for rejected hops
        guard_rejects: dict[int, list] = {}

        avail_cap = dict(cap)

        bfs_dists = self._precompute_bfs_mat(len(self.topo.nodes), cap)

        MAX_HOPS_PER_REQ = min(15, TTL_W)
        hop_counts = [0] * len(self.requestState)

        # ── MAPPO: encode topology + global state ONCE per slot ─────────────
        # node_features / GNN embedding / global_state all depend on ent and the
        # request frontier, both of which evolve hop-by-hop.  We pay once at slot
        # start and reuse for every hop: a minor approximation that saves 14/15 of
        # the expensive GNN + LayerNorm + percentile work.
        # _edge_state_batch inside _route_hive_all_candidates reads live ent, so
        # per-link capacity (the fast-changing quantity) is always current.
        _slot_gs_vec    = None
        _slot_node_feats = None
        _slot_edge_idx  = None
        _slot_node_embs = None
        if self.use_mappo:
            # Build initial pending frontier to seed global_state
            _init_pending = []
            for ridx, rs in enumerate(self.requestState):
                if rs[5]:
                    continue
                curr    = int(rs[2])
                dst_id  = rs[1].id
                visited = rs[3]
                nbrs    = [n for n in np.nonzero(ent[curr])[0]
                           if n not in visited and n != curr]
                if nbrs:
                    _init_pending.append((ridx, rs, curr, dst_id, visited, nbrs))
            _init_edge_demand = self._edge_demand(_init_pending)
            _slot_gs_vec     = self._compute_global_state(
                ent, dist, req_dens,
                pending=_init_pending,
                bfs_dists=bfs_dists,
                edge_demand=_init_edge_demand,
            )
            _slot_node_feats = self._node_features(ent, dist, req_dens)
            _slot_edge_idx   = self._edge_index(ent)
            _slot_node_embs  = self.agent.encode_mappo_graph(_slot_node_feats, _slot_edge_idx)

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
            edge_demand = self._edge_demand(pending)
            active_req_count = len(pending)

            if self.variant == 'seq':
                chosen = self._route_seq(
                    pending, eps, ent, dist, req_dens, bfs_dists, hop_counts, avail_cap,
                    edge_demand, active_req_count)
            elif self.variant == 'guard' and GUARD_ALL_CANDIDATES:
                chosen = self._route_guard_all_candidates(
                    pending, eps, ent, dist, req_dens, bfs_dists, hop_counts, avail_cap,
                    edge_demand=edge_demand, active_req_count=active_req_count,
                    guard_rejects=guard_rejects)
            else:
                chosen = self._route_parallel(
                    pending, eps, ent, dist, req_dens, bfs_dists, hop_counts, avail_cap,
                    deconflict=(self.variant in ('guard', 'hive')),
                    edge_demand=edge_demand,
                    active_req_count=active_req_count,
                    global_state=_slot_gs_vec,
                    node_features=_slot_node_feats,
                    edge_index=_slot_edge_idx,
                    node_embeddings=_slot_node_embs,
                    reject_traj=mappo_rejects if self.use_mappo else None)

            if self.use_mappo:
                for ridx, rejected_steps in mappo_rejects.items():
                    if rejected_steps:
                        mappo_traj[ridx].extend(rejected_steps)
                        rejected_steps.clear()

            for ridx, nbr_id, state_v, policy_info in chosen:
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

                # ── Bellman target: next-candidate set for Double DQN ─────────
                next_nbrs = [n for n in np.nonzero(ent[nbr_id])[0]
                             if n not in rs[3] and n != nbr_id]
                if done or not next_nbrs:
                    next_state_v = state_v   # bootstrap zeroed by done=True
                    next_cands_v = None
                    if not next_nbrs and not done:
                        done = True   # stranded with no reachable neighbour — treat as terminal
                elif self.use_mappo or len(next_nbrs) == 1:
                    next_state_v = self._edge_state_batch(
                        ent, dist, req_dens, bfs_dists, nbr_id, dst_id,
                        [next_nbrs[0]], f_new, hop_counts[ridx],
                        edge_demand, active_req_count)[0]
                    next_cands_v = None
                else:
                    nxt_vecs = self._edge_state_batch(
                        ent, dist, req_dens, bfs_dists, nbr_id, dst_id,
                        next_nbrs, f_new, hop_counts[ridx],
                        edge_demand, active_req_count)
                    # Store all candidates so train_dqn can do proper Double DQN:
                    # online-qnet selects argmax, target-qnet evaluates.
                    next_state_v = nxt_vecs[0]   # placeholder, overridden at train time
                    next_cands_v = nxt_vecs

                req_transitions[ridx].append((state_v, 0, reward, next_state_v, done,
                                              next_cands_v))

                if (self.use_mappo and policy_info is not None
                        and policy_info.get("trainable", True)):
                    step_info = dict(policy_info)
                    step_info["reward"] = float(reward)
                    step_info["done"] = bool(done)
                    mappo_traj[ridx].append(step_info)

        # ── Training ──────────────────────────────────────────────────────────
        if not INFERENCE_MODE:
            # Team reward: each request's terminal step gets a bonus proportional
            # to total slot successes — aligns individual incentives with team goal.
            team_bonus = TEAM_ALPHA * float(success_req) if self.use_mappo else 0.0

            for ridx, t_list in req_transitions.items():
                if not t_list and not (self.use_mappo and mappo_traj.get(ridx)):
                    continue   # request never made a trainable decision
                rs = self.requestState[ridx]
                if not rs[5]:
                    # Request dropped: attach R_TTL to last real transition
                    if t_list:
                        s, a, r, ns, _done, nc = t_list[-1]
                        t_list[-1] = (s, a, r + R_TTL, ns, True, nc)
                    if self.use_mappo and mappo_traj.get(ridx):
                        mappo_traj[ridx][-1]["reward"] = (
                            float(mappo_traj[ridx][-1]["reward"]) + R_TTL
                        )
                        mappo_traj[ridx][-1]["done"] = True
                # Inject team bonus on last step (terminal or not)
                if self.use_mappo and team_bonus > 0.0 and mappo_traj.get(ridx):
                    mappo_traj[ridx][-1]["reward"] = (
                        float(mappo_traj[ridx][-1]["reward"]) + team_bonus
                    )
                if not self.use_mappo:
                    self.agent.single_replay.push_sequence(t_list)
                    self._push_ctr += len(t_list)
                else:
                    self._push_ctr += len(mappo_traj.get(ridx, []))

            if self.use_mappo:
                for traj in mappo_traj.values():
                    if traj:
                        self.agent.mappo_replay.push_trajectory(traj)

            # Guard conflict-loss: each rejected hop → terminal R_CONFLICT
            # transition so the DQN learns contested edge-states have negative
            # immediate value. Pushed separately to avoid n-step contamination.
            if self.variant == 'guard' and guard_rejects:
                for sv_list in guard_rejects.values():
                    for sv in sv_list:
                        self.agent.single_replay.push_sequence(
                            [(sv, 0, R_CONFLICT, sv, True)])
                        self._push_ctr += 1
                guard_rejects.clear()

            while self._push_ctr >= STEP_BETWEEN_TRAIN:
                self._push_ctr -= STEP_BETWEEN_TRAIN
                loss = self.agent.train_mappo() if self.use_mappo else self.agent.train_dqn()
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
        self._maybe_save_best(success_req)

        self.printResult()
        return self.result

    # ── Routing strategies ────────────────────────────────────────────────────

    def _route_seq(self, pending, eps, ent, dist, req_dens, bfs_dists,
                   hop_counts, avail_cap, edge_demand, active_req_count):
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
            nbr, state_v, policy_info = self._pick_action(
                ent, dist, req_dens, bfs_dists,
                curr, dst_id, avail_nbrs, rs, hop_counts[ridx], eps,
                edge_demand=edge_demand, active_req_count=active_req_count)
            lk = (min(curr, nbr), max(curr, nbr))
            local_cap[lk] = max(0, local_cap.get(lk, 0) - 1)
            chosen.append((ridx, nbr, state_v, policy_info))

        return chosen

    def _route_parallel(self, pending, eps, ent, dist, req_dens, bfs_dists,
                         hop_counts, avail_cap, deconflict: bool,
                         edge_demand=None, active_req_count=1,
                         global_state=None, node_features=None, edge_index=None,
                         node_embeddings=None, reject_traj=None):
        """
        Parallel: all requests choose simultaneously; optional b-matching deconflict.
        Returns list of (ridx, chosen_nbr_id, state_v).
        """
        if self.use_mappo and deconflict and HIVE_ALL_CANDIDATES:
            return self._route_hive_all_candidates(
                pending, ent, dist, req_dens, bfs_dists, hop_counts, avail_cap,
                edge_demand=edge_demand,
                active_req_count=active_req_count,
                global_state=global_state,
                node_features=node_features,
                edge_index=edge_index,
                node_embeddings=node_embeddings)

        if (self.variant == 'flock' and not deconflict and not self.use_mappo
                and FLOCK_BATCH_SCORING):
            return self._route_flock_batched(
                pending, eps, ent, dist, req_dens, bfs_dists, hop_counts,
                avail_cap, edge_demand=edge_demand,
                active_req_count=active_req_count)

        raw = []
        for ridx, rs, curr, dst_id, visited, nbrs in pending:
            nbr, state_v, policy_info = self._pick_action(
                ent, dist, req_dens, bfs_dists,
                curr, dst_id, nbrs, rs, hop_counts[ridx], eps,
                edge_demand=edge_demand,
                active_req_count=active_req_count,
                global_state=global_state,
                node_features=node_features,
                edge_index=edge_index,
                node_embeddings=node_embeddings)
            raw.append((ridx, curr, nbr, state_v, policy_info))

        if not deconflict:
            chosen    = []
            local_cap = dict(avail_cap)
            for ridx, curr, nbr, state_v, policy_info in raw:
                lk = (min(curr, nbr), max(curr, nbr))
                if local_cap.get(lk, 0) > 0:
                    local_cap[lk] -= 1
                    chosen.append((ridx, nbr, state_v, policy_info))
            return chosen

        # Guard / Hive: greedy b-matching
        if not raw:
            return []

        # Use policy score for Hive because its actor depends on topology GNN
        # context, not just the chosen edge-state row.
        if self.use_mappo:
            scores_np = np.array([
                float(info.get("score", 0.0)) if info is not None else 0.0
                for (_, _, _, _, info) in raw
            ], dtype=np.float32)
        else:
            batch_rows = [state_v for (_, _, _, state_v, _) in raw]
            scores_np = self.agent.score_neighbors_v(np.stack(batch_rows))

        candidates = [(ridx, curr, nbr, float(scores_np[i]))
                      for i, (ridx, curr, nbr, _, _) in enumerate(raw)]
        matched    = bmatching_nodes(candidates, dict(avail_cap))

        for ridx, curr, nbr, _ in candidates:
            if ridx in matched:
                lk = (min(curr, nbr), max(curr, nbr))
                avail_cap[lk] = max(0, avail_cap.get(lk, 0) - 1)

        raw_dict = {ridx: (curr, nbr, state_v, policy_info)
                    for ridx, curr, nbr, state_v, policy_info in raw}
        chosen   = []
        for ridx, nbr_id in matched.items():
            _, _, state_v, policy_info = raw_dict[ridx]
            chosen.append((ridx, nbr_id, state_v, policy_info))

        if reject_traj is not None:
            for ridx, curr, nbr, state_v, policy_info in raw:
                if (ridx in matched or policy_info is None
                        or not policy_info.get("trainable", True)):
                    continue
                step_info = dict(policy_info)
                step_info["reward"] = R_CONFLICT
                step_info["done"] = False
                step_info["rejected"] = True
                reject_traj.setdefault(ridx, []).append(step_info)

        return chosen

    def _route_flock_batched(self, pending, eps, ent, dist, req_dens, bfs_dists,
                             hop_counts, avail_cap, edge_demand=None,
                             active_req_count=1):
        """
        Flock: batch all request-neighbor scores in one DQN forward pass.

        This keeps Flock's parallel/no-bmatching semantics while making the
        implementation much closer to the throughput-oriented intent of the
        original algorithm.
        """
        score_rows = []
        meta = []
        chosen = []
        local_cap = dict(avail_cap)

        for ridx, rs, curr, dst_id, visited, nbrs in pending:
            forced_nbr = None
            if dst_id in nbrs:
                forced_nbr = dst_id
            elif len(nbrs) == 1:
                forced_nbr = nbrs[0]
            elif not INFERENCE_MODE and _rng.random() < eps:
                forced_nbr = nbrs[int(_rng.randrange(len(nbrs)))]

            if forced_nbr is not None:
                lk = (min(curr, forced_nbr), max(curr, forced_nbr))
                if local_cap.get(lk, 0) <= 0:
                    continue
                state_v = self._edge_state_batch(
                    ent, dist, req_dens, bfs_dists,
                    curr, dst_id, [forced_nbr],
                    float(rs[6]), hop_counts[ridx], edge_demand, active_req_count)[0]
                local_cap[lk] -= 1
                chosen.append((ridx, forced_nbr, state_v, None))
                continue

            batch = self._edge_state_batch(
                ent, dist, req_dens, bfs_dists,
                curr, dst_id, list(nbrs),
                float(rs[6]), hop_counts[ridx], edge_demand, active_req_count)
            start = len(score_rows)
            score_rows.extend(batch)
            stop = len(score_rows)
            meta.append((ridx, curr, dst_id, list(nbrs), start, stop))

        if not score_rows:
            return chosen

        all_rows = np.asarray(score_rows, dtype=np.float32)
        all_scores = self.agent.score_neighbors_v(all_rows)

        for ridx, curr, dst_id, nbrs, start, stop in meta:
            action_i = int(np.argmax(all_scores[start:stop]))
            nbr = nbrs[action_i]
            lk = (min(curr, nbr), max(curr, nbr))
            if local_cap.get(lk, 0) <= 0:
                continue
            local_cap[lk] -= 1
            chosen.append((ridx, nbr, all_rows[start + action_i], None))

        return chosen

    def _route_guard_all_candidates(self, pending, eps, ent, dist, req_dens,
                                    bfs_dists, hop_counts, avail_cap,
                                    edge_demand=None, active_req_count=1,
                                    guard_rejects=None):
        """
        Guard (Fix 2): score every (request, candidate-neighbor) pair in one
        batched DQN forward, globally b-match respecting capacity, and record
        the best-scored rejected candidate per request for conflict-loss training
        (Fix 1).

        avail_cap is NOT mutated here; p4() decrements it when processing chosen.
        """
        if not pending:
            return []

        rows: list = []
        meta: list = []   # (ridx, curr, dst_id, nbrs_list, row_start, row_stop)
        for ridx, rs, curr, dst_id, visited, nbrs in pending:
            start = len(rows)
            batch = self._edge_state_batch(
                ent, dist, req_dens, bfs_dists,
                curr, dst_id, list(nbrs),
                float(rs[6]), hop_counts[ridx],
                edge_demand, active_req_count)
            rows.extend(batch)
            stop = len(rows)
            meta.append((ridx, curr, dst_id, list(nbrs), start, stop))

        if not rows:
            return []

        all_rows   = np.asarray(rows, dtype=np.float32)
        all_scores = self.agent.score_neighbors_v(all_rows)  # one forward pass

        per_req: dict[int, tuple] = {}   # ridx → (curr, nbrs, state_vecs)
        candidates: list = []
        for ridx, curr, dst_id, nbrs, start, stop in meta:
            per_req[ridx] = (curr, nbrs, all_rows[start:stop])
            for i, nbr in enumerate(nbrs):
                score = float(all_scores[start + i])
                if not INFERENCE_MODE:
                    # eps-scaled exploration noise: decays with epsilon so the
                    # b-matcher explores diverse assignments early and converges
                    # to learned scores late.  GUARD_BMATCH_NOISE adds persistent
                    # noise on top (useful at inference for tie-breaking).
                    noise_std = eps * 0.5 + GUARD_BMATCH_NOISE
                    if noise_std > 0.0:
                        score += _rng.gauss(0.0, noise_std)
                candidates.append((ridx, curr, nbr, score))

        if not candidates:
            return []

        matched = bmatching_nodes(candidates, dict(avail_cap))

        chosen: list = []
        for ridx, nbr_id in matched.items():
            curr, nbrs, state_vecs = per_req[ridx]
            try:
                action_i = nbrs.index(nbr_id)
            except ValueError:
                continue
            chosen.append((ridx, nbr_id, state_vecs[action_i], None))

        # Fix 1: record the best-scored candidate for each unmatched request so
        # p4() can push it as a terminal R_CONFLICT transition. This teaches the
        # DQN that high-demand edge-states have negative immediate value without
        # contaminating the main trajectory n-step window.
        if guard_rejects is not None and not INFERENCE_MODE:
            matched_set = set(matched.keys())
            for ridx, curr, dst_id, nbrs, start, stop in meta:
                if ridx in matched_set:
                    continue
                best_i = int(np.argmax(all_scores[start:stop]))
                guard_rejects.setdefault(ridx, []).append(all_rows[start + best_i])

        return chosen

    def _route_hive_all_candidates(self, pending, ent, dist, req_dens, bfs_dists,
                                   hop_counts, avail_cap, edge_demand=None,
                                   active_req_count=1, global_state=None,
                                   node_features=None, edge_index=None,
                                   node_embeddings=None):
        """Hive: score every request-neighbor option, then globally b-match."""
        if not pending:
            return []
        if global_state is None:
            global_state = np.zeros(MAPPO_GLOBAL_DIM, dtype=np.float32)
        if node_features is None:
            node_features = self._node_features(ent, dist, req_dens)
        if edge_index is None:
            edge_index = self._edge_index(ent)
        if node_embeddings is None:
            node_embeddings = self.agent.encode_mappo_graph(node_features, edge_index)

        # ── Collect metadata and build all state/aux vectors in one batch ───────
        # Filter out empty-nbrs entries first; pending is already filtered upstream
        # but guard against it defensively.
        active_pending = [(ridx, rs, curr, dst_id, visited, nbrs)
                          for ridx, rs, curr, dst_id, visited, nbrs in pending
                          if nbrs]
        if not active_pending:
            return []

        state_vecs_list, aux_scores_list = self._edge_state_and_aux_all_requests(
            active_pending, hop_counts, ent, dist, req_dens, bfs_dists,
            edge_demand, active_req_count,
        )

        ridx_order: list[int] = []
        candidate_ids_list: list = []
        curr_ids: list[int]    = []
        dst_ids: list[int]     = []
        nbrs_list: list        = []

        for ridx, rs, curr, dst_id, visited, nbrs in active_pending:
            ridx_order.append(ridx)
            candidate_ids_list.append(np.array(nbrs, dtype=np.int64))
            curr_ids.append(curr)
            dst_ids.append(dst_id)
            nbrs_list.append(list(nbrs))

        # ── ONE actor + ONE critic forward across all pending requests ────────
        batch_results = self.agent.score_mappo_candidates_batched(
            state_vecs_list, global_state, node_embeddings,
            curr_ids, dst_ids, candidate_ids_list,
        )

        # ── Build per_req dict and candidate list for b-matching ─────────────
        candidates = []
        per_req: dict[int, dict] = {}
        gs_np  = np.array(global_state, dtype=np.float32)
        nf_np  = np.array(node_features, dtype=np.float32)
        ei_np  = np.array(edge_index, dtype=np.int64)

        for i, ridx in enumerate(ridx_order):
            logits, logps, value = batch_results[i]
            aux_scores = aux_scores_list[i]
            per_req[ridx] = {
                "curr": curr_ids[i],
                "dst_id": dst_ids[i],
                "nbrs": nbrs_list[i],
                "candidate_states": state_vecs_list[i],
                "candidate_ids": candidate_ids_list[i],
                "logits": logits,
                "logps": logps,
                "value": value,
                "global_state": gs_np,
                "node_features": nf_np,
                "edge_index": ei_np,
                "aux_policy_target": self._aux_policy_target(aux_scores),
                "aux_policy_scores": aux_scores,
            }
            for action_i, nbr in enumerate(nbrs_list[i]):
                score = float(logits[action_i])
                if not INFERENCE_MODE and HIVE_BMATCH_NOISE > 0.0:
                    score += _rng.gauss(0.0, HIVE_BMATCH_NOISE)
                candidates.append((ridx, curr_ids[i], nbr, score))

        if not candidates:
            return []

        matched = bmatching_nodes(candidates, dict(avail_cap))
        chosen = []
        for ridx, nbr_id in matched.items():
            info = per_req.get(ridx)
            if info is None:
                continue
            try:
                action_i = info["nbrs"].index(nbr_id)
            except ValueError:
                continue
            curr = int(info["curr"])
            policy_info = {
                "candidate_states": info["candidate_states"],
                "action_idx": int(action_i),
                "old_logp": float(info["logps"][action_i]),
                "value": float(info["value"]),
                "global_state": info["global_state"],
                "node_features": info["node_features"],
                "edge_index": info["edge_index"],
                "curr_id": curr,
                "dst_id": int(info["dst_id"]),
                "candidate_ids": info["candidate_ids"],
                "aux_policy_target": info["aux_policy_target"],
                "counterfactual_target": self._counterfactual_target(
                    info["aux_policy_scores"], int(action_i)
                ),
                "score": float(info["logits"][action_i]),
                "trainable": True,
            }
            chosen.append((ridx, nbr_id, info["candidate_states"][action_i], policy_info))

        return chosen

    def _pick_action(self, ent, dist, req_dens, bfs_dists,
                     curr: int, dst_id: int, nbrs: list,
                     rs, hops_used: int, eps: float,
                     edge_demand=None, active_req_count=1, global_state=None,
                     node_features=None, edge_index=None,
                     node_embeddings=None):
        """
        Pick a neighbor via ε-greedy Q-scoring.
        Returns (nbr_id, state_v).
        """
        state_vecs = self._edge_state_batch(
            ent, dist, req_dens, bfs_dists,
            curr, dst_id, list(nbrs),
            float(rs[6]), hops_used, edge_demand, active_req_count)

        if self.use_mappo:
            if global_state is None:
                global_state = np.zeros(MAPPO_GLOBAL_DIM, dtype=np.float32)
            if node_features is None:
                node_features = self._node_features(ent, dist, req_dens)
            if edge_index is None:
                edge_index = self._edge_index(ent)
            if node_embeddings is None:
                node_embeddings = self.agent.encode_mappo_graph(node_features, edge_index)
            candidate_ids = np.array(nbrs, dtype=np.int64)
            aux_policy_scores = self._aux_policy_scores(
                ent, dist, req_dens, bfs_dists,
                curr, dst_id, list(nbrs), float(rs[6]), hops_used,
                edge_demand=edge_demand, active_req_count=active_req_count,
            )
            aux_policy_target = self._aux_policy_target(aux_policy_scores)
            forced_action = None
            trainable = True
            if MAPPO_FORCE_DEST and dst_id in nbrs:
                forced_action = int(nbrs.index(dst_id))
                trainable = False
            elif MAPPO_EPS_GREEDY and not INFERENCE_MODE and _rng.random() < eps:
                forced_action = int(_rng.randrange(len(nbrs)))
                trainable = False
            action_i, logp, value, score = self.agent.select_mappo_action(
                state_vecs, global_state, node_features, edge_index,
                curr, dst_id, candidate_ids,
                node_embeddings=node_embeddings,
                forced_action=forced_action,
                deterministic=INFERENCE_MODE)
            nbr = nbrs[action_i]
            policy_info = {
                "candidate_states": state_vecs,
                "action_idx": action_i,
                "old_logp": logp,
                "value": value,
                "global_state": np.array(global_state, dtype=np.float32),
                "node_features": np.array(node_features, dtype=np.float32),
                "edge_index": np.array(edge_index, dtype=np.int64),
                "curr_id": int(curr),
                "dst_id": int(dst_id),
                "candidate_ids": candidate_ids,
                "aux_policy_target": aux_policy_target,
                "counterfactual_target": self._counterfactual_target(
                    aux_policy_scores, int(action_i)
                ),
                "score": score,
                "trainable": trainable,
            }
            return nbr, state_vecs[action_i], policy_info

        if dst_id in nbrs:
            action_i = int(nbrs.index(dst_id))
        elif _rng.random() < eps:
            action_i = int(_rng.randrange(len(nbrs)))
        else:
            scores  = self.agent.score_neighbors_v(state_vecs)
            action_i  = int(np.argmax(scores))

        nbr = nbrs[action_i]
        return nbr, state_vecs[action_i], None

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
        G.add_nodes_from(range(len(self.topo.nodes)))
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
