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
    get_neighbor_embeddings,
    apply_neighbor_attention,
    get_global_state_vector,
    get_epsilon_linear,
)
from .replay import STEP_BETWEEN_TRAIN

SIZE           = int(os.environ.get("SIZE", "100"))
INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"

REWARD_LAMBDA = 0.3
REWARD_MU     = 1.0
REWARD_NU     = 0.5

MODEL_DIR = os.environ.get("MODEL_DIR", "/tmp/qrouting_model")


# ── State construction (attention-cached) ────────────────────────────────────

def _build_state(curr_req, attn_feats: np.ndarray,
                 ent_arr: np.ndarray, dist_arr: np.ndarray) -> np.ndarray:
    """
    Build 20228-dim state vector using pre-computed per-timeslot attn_feats.
    Avoids repeating the MHA forward pass for every hop of every request.

    curr_req: [src_id, dst_id, curr_node_id, path_vec(SIZE,), index, done]
    attn_feats: (2*N, 64) output of apply_request_attention() — index by req[4]
    """
    curr_index = int(curr_req[4])
    curr_emb   = attn_feats[curr_index]                          # (64,)
    neigh_embs = get_neighbor_embeddings(ent_arr, int(curr_req[2]))
    context    = apply_neighbor_attention(curr_emb, neigh_embs)  # (64,)

    local = np.zeros(SIZE, dtype=np.float32)
    local[int(curr_req[2])] = 10.0
    local[int(curr_req[1])] = 10.0

    return np.concatenate([
        curr_emb, context, local,
        ent_arr.flatten(),
        dist_arr.flatten(),
    ]).astype(np.float32)   # 64+64+100+10000+10000 = 20228


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
        torch.save(self.agent.model.state_dict(), self._model_path)
        print(f"[{self.name}] weights saved → {self._model_path}")

    def _load_weights(self) -> bool:
        import torch
        if not os.path.exists(self._model_path):
            print(f"[{self.name}] no checkpoint at {self._model_path} — random policy")
            return False
        self.agent.model.load_state_dict(
            torch.load(self._model_path, map_location="cpu", weights_only=True))
        self.agent.target_model.load_state_dict(self.agent.model.state_dict())
        print(f"[{self.name}] weights loaded from {self._model_path}")
        return True

    # ── AlgorithmBase interface ───────────────────────────────────────────────

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

    def _get_ent_matrix(self) -> np.ndarray:
        m = np.zeros((SIZE, SIZE), dtype=np.float32)
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                m[link.n1.id][link.n2.id] += 1
                m[link.n2.id][link.n1.id] += 1
        return m

    def _get_dist_matrix(self) -> np.ndarray:
        m = np.zeros((SIZE, SIZE), dtype=np.float32)
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                m[link.n1.id][link.n2.id] = link.fidelity
                m[link.n2.id][link.n1.id] = link.fidelity
        return m

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

    def _make_curr_req(self, req_state) -> list:
        """Convert a requestState entry to the curr_req format for _build_state."""
        path_vec = np.zeros(SIZE, dtype=np.float32)
        for nid in req_state[3]:
            if 0 <= nid < SIZE:
                path_vec[nid] = 1.0
        return [
            req_state[0].id,    # src id
            req_state[1].id,    # dst id
            int(req_state[2]),  # current node id
            path_vec,           # path vector (SIZE,)
            int(req_state[4]),  # index into req_matrix rows
            req_state[5],       # done flag
        ]

    # ── Routing + training ────────────────────────────────────────────────────

    def p4(self):
        if not self.requestState:
            for lst in (self.result.successfulRequestPerRound,
                        self.result.entanglementPerRound,
                        self.result.fidelityPerRound,
                        self.result.rewardPerRound):
                lst.append(0)
            self.printResult()
            return self.result

        ent_arr  = self._get_ent_matrix()    # (SIZE, SIZE) — updated in-place during routing
        dist_arr = self._get_dist_matrix()   # (SIZE, SIZE) — fixed snapshot for whole timeslot
        req_m    = self._get_req_matrix()    # (2*N, SIZE)
        eps      = get_epsilon_linear(self.timeSlot)

        # Compute attention ONCE for all requests in this timeslot.
        # This is the main speedup: O(1) MHA forward passes instead of O(N*hops).
        req_feats  = get_request_embeddings(req_m)     # (2*N, 64)
        attn_feats = apply_request_attention(req_feats) # (2*N, 64)

        success_req    = 0
        total_fidelity = 0.0
        transitions    = []         # (state, action, raw_reward, next_state, done, req_idx)
        routed_links   = set()      # conflict resolution: each link usable once per timeslot

        for req_idx, req_state in enumerate(self.requestState):
            if req_state[5]:
                continue

            dst_id   = req_state[1].id
            curr     = int(req_state[2])
            fidelity = 1.0
            routed   = False

            for _ in range(25):     # max hops per request per timeslot
                visited   = set(req_state[3])
                neighbors = [
                    i for i in range(SIZE)
                    if ent_arr[curr][i] >= 1 and i not in visited and i != curr
                ]
                if not neighbors:
                    break

                curr_req = self._make_curr_req(req_state)
                state    = _build_state(curr_req, attn_feats, ent_arr, dist_arr)

                # Masked action selection: only valid neighbors are eligible
                if dst_id in neighbors:
                    action = dst_id                     # direct hop to destination
                elif _rng.random() < eps:
                    action = _rng.choice(neighbors)    # random exploration over valid hops
                else:
                    q_vals = self.agent.batch_predict(state[np.newaxis])[0]  # (SIZE,)
                    action = max(neighbors, key=lambda n: float(q_vals[n]))  # greedy valid

                link_key = (min(curr, action), max(curr, action))
                if link_key in routed_links:
                    break
                routed_links.add(link_key)

                # Consume one entanglement on this link
                ent_arr[curr][action] = max(0.0, ent_arr[curr][action] - 1)
                ent_arr[action][curr] = max(0.0, ent_arr[action][curr] - 1)

                # Werner-swap fidelity accumulation
                hop_fid  = float(dist_arr[curr][action])
                fidelity = fidelity * hop_fid + (1.0 - fidelity) * (1.0 - hop_fid) / 3.0

                req_state[2] = action
                req_state[3] = tuple(visited | {action})
                curr = action

                done          = (curr == dst_id)
                next_curr_req = self._make_curr_req(req_state)
                next_state    = _build_state(next_curr_req, attn_feats, ent_arr, dist_arr)

                raw_r = 10.0 if done else -0.1
                transitions.append((state, action, raw_r, next_state, done, req_idx))

                if done:
                    routed = True
                    success_req    += 1
                    total_fidelity += fidelity
                    break

            req_state[5] = routed

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
        print(f"[{self.name}] ts={self.timeSlot}"
              f"  success={self.result.successfulRequest}"
              f"  remain={len(self.requests)}")


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
        print(f"[{self.name}] ts={self.timeSlot}"
              f"  success={self.result.successfulRequest}"
              f"  remain={len(self.requests)}")
