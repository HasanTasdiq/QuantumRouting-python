"""
Local in-process QuRA training — replaces the HTTP/Redis/FedAvg machinery.

Four named variants:
  QuRA_DQRL_DIST   (Seq)  ─┐
  QuRA_Flock_DIST          ├─ single-agent Double DQN
  QuRA_Guard_DIST          ┘
  QuRA_Hive_DIST           ─── QMIX joint training

Also provides:
  ShortestPath             ─── BFS shortest-path baseline (no learning)

Physics (entanglement generation, qubit assignment) is inherited from
AlgorithmBase unchanged.  Only p4() and training differ from the original.

State: identical 20228-dim vector as the original distributed QuRA.
Reward: Paper Eq. 9  r*λ + N_success*μ + F_avg*ν  applied per timeslot.

Key optimisation: request attention is computed ONCE per p4() call
(one MHA pass) and reused for every hop of every request — instead of
one MHA pass per hop as in the distributed server.
"""

import os
import sys
import random as _rng
from collections import defaultdict

import networkx as nx
import numpy as np

# ── Path setup ───────────────────────────────────────────────────────────────
_here     = os.path.dirname(os.path.abspath(__file__))           # src/rl/pt
_src_dir  = os.path.normpath(os.path.join(_here, '../..'))       # src
_algo_dir = os.path.join(_src_dir, 'quantum', 'algorithm')

for _p in [_algo_dir, _src_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AlgorithmBase import AlgorithmBase

from .agent   import DQRLAgentDist
from .helpers import (
    get_request_embeddings,
    apply_request_attention,
    get_neighbor_context,
    get_global_state_vector,
    get_epsilon_linear,
    precompute_all_node_embeddings,
    dense_proj, dense_neighbor, mha, ln_req,
)
from .replay import STEP_BETWEEN_TRAIN

SIZE           = int(os.environ.get("SIZE", "100"))
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"

REWARD_LAMBDA = 0.3
REWARD_MU     = 1.0
REWARD_NU     = 0.5

MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/qrouting_model")

# Pre-allocated buffer for the local node-indicator slice (avoids np.zeros per call)
_local_buf = np.zeros(SIZE, dtype=np.float32)


# ── State construction (attention-cached) ────────────────────────────────────

def _build_state(dst_id: int, curr_id: int, req_idx: int,
                 attn_feats: np.ndarray,
                 ent_arr: np.ndarray, dist_flat: np.ndarray,
                 node_emb_cache: np.ndarray,
                 ent_flat: np.ndarray | None = None) -> np.ndarray:
    """
    Build 20228-dim state vector.

    Uses get_neighbor_context (np.nonzero + array indexing, ~50x faster than
    the Python for-loop) and a pre-allocated _local_buf to avoid per-call
    np.zeros allocation.

    ent_flat: maintained running flat copy of ent_arr (updated in-place as
    entanglements are consumed). When provided avoids a full flatten().
    """
    curr_emb = attn_feats[req_idx]                           # (64,) view
    context  = get_neighbor_context(                         # (64,)
        ent_arr[curr_id], curr_emb, node_emb_cache)

    _local_buf[:] = 0.0          # clear in-place (~5x faster than np.zeros)
    _local_buf[curr_id] = 1.0
    _local_buf[dst_id]  = 1.0

    _ef = ent_flat if ent_flat is not None else ent_arr.flatten()
    return np.concatenate([curr_emb, context, _local_buf, _ef, dist_flat]).astype(np.float32)


# ── Base local QuRA class ─────────────────────────────────────────────────────

class QuRA_Local(AlgorithmBase):
    """
    Base local QuRA class.
    Seq/Flock/Guard: use_qmix=False → single-agent DQN.
    Hive:            use_qmix=True  → QMIX joint training.
    """

    def __init__(self, topo, param=None, name='QuRA_Local', use_qmix=False):
        super().__init__(topo)
        self.name             = name
        self.use_qmix         = use_qmix
        self.requests         = []
        self.totalRequest     = 0
        self.totalWaitingTime = 0
        self.totalUsedQubits  = 0
        self.requestState     = []
        self._push_ctr        = 0

        self.agent = DQRLAgentDist(pid=0, model_name='dqrl_model')
        self._model_path = os.path.join(
            MODEL_DIR, f"{name.lower().replace(' ', '_')}.pt")

        if INFERENCE_MODE:
            self._load_weights()

    # ── Weight save / load ────────────────────────────────────────────────────

    def _save_weights(self) -> None:
        import torch
        os.makedirs(MODEL_DIR, exist_ok=True)
        # Save Q-network + shared embedding layers together so inference mode
        # reproduces the exact same state representation as training.
        torch.save({
            "model":        self.agent.model.state_dict(),
            "dense_proj":   dense_proj.state_dict(),
            "dense_neighbor": dense_neighbor.state_dict(),
            "mha":          mha.state_dict(),
            "ln_req":       ln_req.state_dict(),
        }, self._model_path)
        print(f"[{self.name}] weights saved → {self._model_path}")

    def _load_weights(self) -> bool:
        import torch
        if not os.path.exists(self._model_path):
            print(f"[{self.name}] no checkpoint at {self._model_path} — random policy")
            return False
        ckpt = torch.load(self._model_path, map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model" in ckpt:
            self.agent.model.load_state_dict(ckpt["model"])
            if "dense_proj"     in ckpt: dense_proj.load_state_dict(ckpt["dense_proj"])
            if "dense_neighbor" in ckpt: dense_neighbor.load_state_dict(ckpt["dense_neighbor"])
            if "mha"            in ckpt: mha.load_state_dict(ckpt["mha"])
            if "ln_req"         in ckpt: ln_req.load_state_dict(ckpt["ln_req"])
        else:
            # legacy: bare model state_dict (no embedding layers saved)
            self.agent.model.load_state_dict(ckpt)
        self.agent.target_model.load_state_dict(self.agent.model.state_dict())
        print(f"[{self.name}] weights loaded from {self._model_path}")
        return True

    # ── AlgorithmBase interface ───────────────────────────────────────────────

    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))
        self.srcDstPairs = []

        # Cap the pending queue to prevent unbounded growth during exploration.
        # A random policy early in training can have near-0% success rate, causing
        # the queue to grow by numOfRequestPerRound every timeslot.  Without a
        # limit, batch sizes balloon and each timeslot takes O(N²) time.
        # Keep at most 5× the per-round load; drop oldest pending requests.
        _max_q = max(self.topo.numOfRequestPerRound * 5, 50)
        if len(self.requests) > _max_q:
            self.requests = self.requests[-_max_q:]

        self.requestState = []
        index = 0
        for request in self.requests:
            src, dst = request[0], request[1]
            if (src, dst) not in self.srcDstPairs:
                self.srcDstPairs.append((src, dst))
            self.requestState.append(
                [src, dst, src.id, tuple([src.id]), index, False])
            index += 1

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime  += len(self.requests)
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.randPFT()

    def randPFT(self):
        assignable = True
        while assignable:
            assignable = False
            for link in self.topo.links:
                if link.assignable():
                    assignable = True
                    if np.random.random() > 0.5:
                        link.assignQubits()
                        self.totalUsedQubits += 2

    # ── Matrix helpers ────────────────────────────────────────────────────────

    def _get_matrices(self) -> tuple[np.ndarray, np.ndarray]:
        """Single pass over links → (ent_matrix, dist_matrix). Replaces two loops."""
        ent  = np.zeros((SIZE, SIZE), dtype=np.float32)
        dist = np.zeros((SIZE, SIZE), dtype=np.float32)
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                i, j = link.n1.id, link.n2.id
                ent[i][j] += 1;        ent[j][i] += 1
                dist[i][j] = link.fidelity; dist[j][i] = link.fidelity
        return ent, dist

    def _get_req_matrix(self) -> np.ndarray:
        """
        Returns (2*N, SIZE) matrix: first N rows are request descriptors,
        last N rows are path vectors.  Matches the format expected by
        get_request_embeddings() in helpers.py.
        """
        rows, paths = [], []
        for req in self.requestState:
            r = np.zeros(SIZE, dtype=np.float32)
            r[0] = req[0].id
            r[1] = req[1].id
            r[2] = req[2]
            r[4] = req[4]
            r[5] = float(req[5])
            rows.append(r)
            p = np.zeros(SIZE, dtype=np.float32)
            for nid in req[3]:
                if 0 <= nid < SIZE:
                    p[nid] = 1.0
            paths.append(p)
        if not rows:
            return np.zeros((0, SIZE), dtype=np.float32)
        return np.vstack(rows + paths).astype(np.float32)

    # ── Routing + training ────────────────────────────────────────────────────

    def p4(self):
        # Discard requests older than 1 timeslot (TTL=1)
        self.requests = [r for r in self.requests if self.timeSlot - r[2] < 1]
        self.requestState = [s for s in self.requestState if self.timeSlot - s[0].id < 1]

        if not self.requestState:
            for lst in (self.result.successfulRequestPerRound,
                        self.result.entanglementPerRound,
                        self.result.fidelityPerRound,
                        self.result.rewardPerRound):
                lst.append(0)
            self.printResult()
            return self.result

        # ── Per-timeslot pre-computation (amortised over all requests/hops) ───
        ent_arr, dist_arr = self._get_matrices()  # single link pass for both matrices
        dist_flat         = dist_arr.flatten()    # pre-flatten; dist is fixed this timeslot
        req_m             = self._get_req_matrix()

        # One MHA pass for all requests (was: one pass per hop per request)
        req_feats  = get_request_embeddings(req_m)
        attn_feats = apply_request_attention(req_feats)

        # One batched dense_neighbor call for all nodes (was: one call per neighbor per hop)
        node_emb_cache = precompute_all_node_embeddings()   # (SIZE, 64)

        eps             = get_epsilon_linear(self.timeSlot)
        success_req     = 0
        total_fidelity  = 0.0
        transitions     = []
        routed_links    = set()
        fidelity_track  = [1.0] * len(self.requestState)  # per-request fidelity

        # Maintain a flat copy of ent_arr updated in-place as links are consumed.
        # This avoids calling ent_arr.flatten() inside every _build_state call.
        ent_flat = ent_arr.flatten().copy()

        # ── Batched-hop routing loop ──────────────────────────────────────────
        # One hop at a time across ALL active requests, with a single batched
        # Q-network call per hop step — instead of N_requests × N_hops individual
        # calls.  Reduces PyTorch invocations from O(N*H) to O(H).
        # 15 hops: sufficient for a 10×10 grid (diameter 18); Werner-swap fidelity
        # at 15 hops is < 0.9^15 ≈ 0.20, so longer paths are physically useless.
        for _hop in range(15):
            # Gather all active requests that still have valid moves.
            # np.nonzero on the current node's row (C-level, ~100ns) replaces
            # a 100-iteration Python loop for finding entangled neighbors.
            pending = []   # (ridx, req_state, dst_id, curr, visited, neighbors)
            for ridx, req_state in enumerate(self.requestState):
                if req_state[5]:
                    continue
                curr    = int(req_state[2])
                dst_id  = req_state[1].id
                visited = set(req_state[3])
                raw = np.nonzero(ent_arr[curr])[0]   # C-level: indices with ent > 0
                neighbors = [i for i in raw if i not in visited and i != curr]
                if neighbors:
                    pending.append((ridx, req_state, dst_id, curr, visited, neighbors))

            if not pending:
                break

            # Build states; collect requests that need greedy Q-network selection
            greedy_batch = []   # (ridx, dst_id, curr, neighbors, state)
            hop_states   = {}   # ridx → (action_if_known, state, dst_id, curr, neighbors)

            for ridx, req_state, dst_id, curr, visited, neighbors in pending:
                state = _build_state(dst_id, curr, ridx,
                                     attn_feats, ent_arr, dist_flat,
                                     node_emb_cache, ent_flat)
                if dst_id in neighbors:
                    hop_states[ridx] = (dst_id, state, dst_id, curr, neighbors)
                elif _rng.random() < eps:
                    hop_states[ridx] = (_rng.choice(neighbors), state, dst_id, curr, neighbors)
                else:
                    greedy_batch.append((ridx, dst_id, curr, neighbors, state))

            # One batched Q-network call for all greedy decisions this hop step
            if greedy_batch:
                batch_states = np.stack([s for *_, s in greedy_batch])
                batch_qvals  = self.agent.batch_predict(batch_states)   # (N_greedy, SIZE)
                for i, (ridx, dst_id, curr, neighbors, state) in enumerate(greedy_batch):
                    action = max(neighbors, key=lambda n: float(batch_qvals[i][n]))
                    hop_states[ridx] = (action, state, dst_id, curr, neighbors)

            # Apply actions: consume entanglement, update fidelity, record transitions
            for ridx, req_state, _, curr_prev, visited, _ in pending:
                if ridx not in hop_states:
                    continue
                action, state, dst_id, curr, neighbors = hop_states[ridx]

                link_key = (min(curr, action), max(curr, action))
                if link_key in routed_links:
                    continue
                routed_links.add(link_key)

                # Consume entanglement and keep ent_flat in sync
                ent_arr[curr][action]  = max(0.0, ent_arr[curr][action]  - 1)
                ent_arr[action][curr]  = max(0.0, ent_arr[action][curr]  - 1)
                ent_flat[curr * SIZE + action] = ent_arr[curr][action]
                ent_flat[action * SIZE + curr] = ent_arr[action][curr]

                # Werner-swap fidelity
                hop_fid = float(dist_arr[curr][action])
                fidelity_track[ridx] = (fidelity_track[ridx] * hop_fid
                                        + (1.0 - fidelity_track[ridx]) * (1.0 - hop_fid) / 3.0)

                req_state[2] = action
                req_state[3] = tuple(visited | {action})

                done       = (action == dst_id)
                next_state = _build_state(dst_id, action, ridx,
                                          attn_feats, ent_arr, dist_flat,
                                          node_emb_cache, ent_flat)

                raw_r = 10.0 if done else -0.1
                transitions.append((state, action, raw_r, next_state, done, ridx))

                if done:
                    req_state[5] = True
                    success_req    += 1
                    total_fidelity += fidelity_track[ridx]

        # ── Training step ─────────────────────────────────────────────────────
        if not INFERENCE_MODE and transitions:
            avg_fid      = total_fidelity / max(success_req, 1)
            global_state = get_global_state_vector(ent_arr, req_m)

            if self.use_qmix:
                groups: dict = defaultdict(list)
                for (s, a, r, ns, d, ridx) in transitions:
                    groups[ridx].append((s, a, r, ns, d))

                for ridx, entries in groups.items():
                    ts_r = [
                        e[2] * REWARD_LAMBDA + success_req * REWARD_MU + avg_fid * REWARD_NU
                        for e in entries
                    ]
                    self.agent.push_qmix_transition(
                        [e[0] for e in entries], [e[1] for e in entries], ts_r,
                        [e[3] for e in entries], global_state, global_state,
                        entries[-1][4],
                    )
                    self._push_ctr += 1

                if self._push_ctr >= STEP_BETWEEN_TRAIN:
                    self._push_ctr = 0
                    loss = self.agent.qmix_train_step()
                    if loss is not None:
                        print(f"[{self.name}] ts={self.timeSlot}  succ={success_req}"
                              f"  qmix_loss={loss:.5f}"
                              f"  buf={len(self.agent.qmix_replay)}")
            else:
                for (s, a, r, ns, d, _) in transitions:
                    scaled = r * REWARD_LAMBDA + success_req * REWARD_MU + avg_fid * REWARD_NU
                    self.agent.remember(s, a, scaled, ns, d)
                    self._push_ctr += 1

                if self._push_ctr >= STEP_BETWEEN_TRAIN:
                    self._push_ctr = 0
                    loss = self.agent.replay()
                    if loss is not None:
                        print(f"[{self.name}] ts={self.timeSlot}  succ={success_req}"
                              f"  dqn_loss={loss:.5f}"
                              f"  buf={len(self.agent.single_replay)}")

        # ── Cleanup ───────────────────────────────────────────────────────────
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
            self.result.waitingTime = self.totalWaitingTime / self.totalRequest
            self.result.usedQubits  = self.totalUsedQubits  / self.totalRequest
        self.result.remainRequestPerRound.append(len(self.requests))


# ── Named QuRA variants ───────────────────────────────────────────────────────

class QuRA_DQRL_DIST(QuRA_Local):
    def __init__(self, topo, param=None, name='QuRA_Seq_DIST'):
        super().__init__(topo, param, name, use_qmix=False)


class QuRA_Flock_DIST(QuRA_Local):
    def __init__(self, topo, param=None, name='QuRA_Flock_DIST'):
        super().__init__(topo, param, name, use_qmix=False)


class QuRA_Guard_DIST(QuRA_Local):
    def __init__(self, topo, param=None, name='QuRA_Guard_DIST'):
        super().__init__(topo, param, name, use_qmix=False)


class QuRA_Hive_DIST(QuRA_Local):
    def __init__(self, topo, param=None, name='QuRA_Hive_DIST'):
        super().__init__(topo, param, name, use_qmix=True)


# ── ShortestPath baseline ─────────────────────────────────────────────────────

class ShortestPath(AlgorithmBase):
    """
    Shortest-path baseline: BFS over the per-timeslot entanglement graph.
    No learning.  Uses networkx for path finding.
    """

    def __init__(self, topo, param=None, name='ShortestPath'):
        super().__init__(topo)
        self.name             = name
        self.requests         = []
        self.totalRequest     = 0
        self.totalWaitingTime = 0
        self.totalUsedQubits  = 0
        self.requestState     = []

    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))
        self.srcDstPairs = []
        self.requestState = []
        index = 0
        for request in self.requests:
            src, dst = request[0], request[1]
            if (src, dst) not in self.srcDstPairs:
                self.srcDstPairs.append((src, dst))
            self.requestState.append(
                [src, dst, src.id, set([src.id]), index, False])
            index += 1

    def p2(self):
        self.AddNewSDpairs()
        self.totalWaitingTime += len(self.requests)
        self.result.idleTime  += len(self.requests)
        if len(self.srcDstPairs) > 0:
            self.result.numOfTimeslot += 1
            self.randPFT()

    def randPFT(self):
        assignable = True
        while assignable:
            assignable = False
            for link in self.topo.links:
                if link.assignable():
                    assignable = True
                    if np.random.random() > 0.5:
                        link.assignQubits()
                        self.totalUsedQubits += 2

    def p4(self):
        # Discard requests older than 1 timeslot (TTL=1)
        self.requests = [r for r in self.requests if self.timeSlot - r[2] < 1]
        self.requestState = [s for s in self.requestState if self.timeSlot - s[0].id < 1]

        # Build entanglement graph and track available link counts
        G = nx.Graph()
        G.add_nodes_from(range(SIZE))
        ent_avail: dict = {}

        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                u, v = link.n1.id, link.n2.id
                G.add_edge(u, v)
                key = (min(u, v), max(u, v))
                ent_avail[key] = ent_avail.get(key, 0) + 1

        success_req = 0
        for req_state in self.requestState:
            if req_state[5]:
                continue
            curr   = int(req_state[2])
            dst_id = req_state[1].id

            try:
                path = nx.shortest_path(G, curr, dst_id)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                continue

            routed = True
            for next_hop in path[1:]:
                key = (min(curr, next_hop), max(curr, next_hop))
                if ent_avail.get(key, 0) < 1:
                    routed = False
                    break
                ent_avail[key] -= 1
                curr = next_hop

            if routed and curr == dst_id:
                req_state[5] = True
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
            self.result.waitingTime = self.totalWaitingTime / self.totalRequest
            self.result.usedQubits  = self.totalUsedQubits  / self.totalRequest
        self.result.remainRequestPerRound.append(len(self.requests))
