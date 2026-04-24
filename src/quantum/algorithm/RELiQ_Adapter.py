"""
RELiQ_Adapter: wraps a trained RELiQ policy as an AlgorithmBase baseline.

Inherits p2() / tryEntanglement() / postProcess() from AlgorithmBase so that
the physical-layer simulation (qubit assignment, entanglement generation,
link fidelity) is identical to all other algorithms in the experiment.

p4() is overridden to use RELiQ's trained DQN for per-request routing decisions.

Physics: QuRA's Werner swap model is patched into RELiQ's QuantumLink at import
time (via entanglement_lifetime=10 and Werner formula — see quantum_network.py).

Observation format built here matches --request-based-observation --netmon
--netmon-agg-type=sage training used in src/reliq/train.py.
"""

import os
import sys
import json
import copy
import random

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_algo_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir  = os.path.dirname(_algo_dir)
for _p in [_algo_dir, _src_dir,
           os.path.join(_src_dir, 'reliq'),
           os.path.join(_algo_dir, '..', '..', 'rl', 'pt')]:
    _ap = os.path.abspath(_p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

from AlgorithmBase import AlgorithmBase, AlgorithmResult

# ── Obs constants (must match training config in src/reliq/train.py) ──────────
_MAX_REQUESTS  = int(os.environ.get("MAX_REQUESTS",  "100"))
_NEIGHBOR_COUNT = int(os.environ.get("NEIGHBOR_COUNT", "6"))
# per-neighbor features: swap_prob + avail_links + top_fidelity + one_hot(dest_state,6)
_NEIGH_FEAT   = 9
# base features: one_hot(id, MAX_REQ) + link_fid + path_len + n_ent_at_target
_BASE_FEAT    = _MAX_REQUESTS + 3
_OBS_SIZE     = _BASE_FEAT + _NEIGHBOR_COUNT * _NEIGH_FEAT  # per-request feature dim


def _one_hot(idx: int, size: int) -> list:
    v = [0] * size
    if 0 <= idx < size:
        v[idx] = 1
    return v


def _dest_state_oh(other_node: int, visited: set, start: int, target: int) -> list:
    """RELiQ's 6-class dest_state one-hot."""
    if other_node in visited:
        state = 0 if other_node == start else 1
    elif other_node == target:
        state = 3
    else:
        state = 2
    return _one_hot(state, 6)


class RELiQ_Adapter(AlgorithmBase):
    """
    RELiQ as a drop-in AlgorithmBase baseline.

    Parameters
    ----------
    topo       : Topo object (same as other algorithms)
    model_path : path to a trained RELiQ model.pt checkpoint
                 (produced by src/reliq/train.py).
                 Falls back to shortest-path greedy if not found.
    name       : algorithm name for CSV logs
    """

    def __init__(self, topo, param=None,
                 name='RELiQ',
                 model_path='runs_quantum/RELiQ_QuRAPhysics/model.pt'):
        super().__init__(topo)
        self.name  = name
        self.requests      = []
        self.totalRequest  = 0
        self.totalWaitingTime = 0
        self.totalUsedQubits = 0

        self._policy = None
        self._model_path = model_path
        self._load_policy()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_policy(self):
        import torch

        if not os.path.exists(self._model_path):
            print(f"[RELiQ_Adapter] WARNING: no model at {self._model_path}"
                  " — using greedy-fidelity fallback.")
            return

        # Load args saved alongside the model (args.json in same dir)
        args_path = os.path.join(os.path.dirname(self._model_path), "args.json")
        model_args = {}
        if os.path.exists(args_path):
            with open(args_path) as f:
                model_args = json.load(f)

        try:
            from reliq.model import DQN, NetMon
            import torch.nn as nn

            # Reconstruct model from saved checkpoint
            checkpoint = torch.load(self._model_path, map_location="cpu",
                                    weights_only=False)

            # Detect model type and rebuild
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
                arch = checkpoint.get("arch", "netmon")
            else:
                state_dict = checkpoint
                arch = model_args.get("model", "dqn")

            # Build model with matching dimensions
            in_feats  = model_args.get("in_features",  _OBS_SIZE)
            n_actions = model_args.get("num_actions",  _NEIGHBOR_COUNT)
            hidden    = model_args.get("hidden",        64)

            if arch == "netmon" or "netmon" in arch:
                agg_type   = model_args.get("netmon_agg_type", "sage")
                iterations = model_args.get("netmon_iterations", 2)
                model = NetMon(
                    in_features=in_feats,
                    hidden_features=hidden,
                    encoder_units=(hidden,),
                    iterations=iterations,
                    activation_fn=nn.ReLU(),
                    rnn_type="none",
                    agg_type=agg_type,
                )
            else:
                mlp_units = model_args.get("mlp_units", (hidden, hidden))
                model = DQN(
                    in_features=in_feats,
                    mlp_units=mlp_units,
                    num_actions=n_actions,
                    activation_fn=nn.ReLU(),
                )

            # Load weights (ignore missing/unexpected keys gracefully)
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[RELiQ_Adapter] missing keys: {missing[:3]}...")
            model.eval()
            self._policy = model
            print(f"[RELiQ_Adapter] loaded policy from {self._model_path}")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[RELiQ_Adapter] model load failed ({e}), using greedy fallback.")
            self._policy = None

    # ── Observation construction ──────────────────────────────────────────────
    def _build_obs(self, req_idx: int) -> np.ndarray:
        """
        Build per-request observation matching --request-based-observation format.
        Returns float32 array of shape (_OBS_SIZE,).
        """
        req   = self.requestState[req_idx]
        src   = req[0]          # Node object
        dst   = req[1]          # Node object
        curr_id = int(req[2])   # current node id
        path  = req[3]          # tuple of visited node ids
        done  = req[5]

        visited = set(path)
        start_id = src.id
        target_id = dst.id

        ob = []

        # 1. Packet one-hot ID
        ob += _one_hot(req_idx, _MAX_REQUESTS)

        # 2. Current link fidelity (1.0 at beginning of hop)
        ob.append(1.0)

        # 3. Path hops completed (normalised by TTL)
        ob.append(float(len(path) - 1))

        # 4. Entanglements at target (approximation from ent_matrix)
        ent_matrix = self.get_ent_graph_matrix()
        n_ent_target = int(ent_matrix[target_id].sum())
        ob.append(float(n_ent_target))

        # 5. Per-neighbor features (up to _NEIGHBOR_COUNT)
        nodes_seen = 0
        for node in self.topo.nodes:
            if nodes_seen >= _NEIGHBOR_COUNT:
                break
            # Only neighbours with entanglement to current node
            if ent_matrix[curr_id][node.id] >= 1:
                swap_prob  = float(node.q)
                avail      = int(ent_matrix[curr_id][node.id])
                top_fid    = self._best_link_fidelity(curr_id, node.id)
                dest_oh    = _dest_state_oh(node.id, visited, start_id, target_id)
                ob += [swap_prob, float(avail), top_fid] + dest_oh
                nodes_seen += 1

        # Pad missing neighbours
        for _ in range(_NEIGHBOR_COUNT - nodes_seen):
            ob += [0.0, 0.0, 0.0] + _one_hot(1, 6)  # dest_state=1 → visited

        return np.array(ob, dtype=np.float32)

    def _best_link_fidelity(self, node_a: int, node_b: int) -> float:
        best = 0.0
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                a, b = link.n1.id, link.n2.id
                if (a == node_a and b == node_b) or (a == node_b and b == node_a):
                    best = max(best, link.fidelity)
        return best

    def get_ent_graph_matrix(self) -> np.ndarray:
        n = len(self.topo.nodes)
        m = np.zeros((n, n), dtype=np.float32)
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                i, j = link.n1.id, link.n2.id
                m[i][j] += 1
                m[j][i] += 1
        return m

    # ── Inference ─────────────────────────────────────────────────────────────
    def _select_next_hop(self, req_idx: int,
                         ent_matrix: np.ndarray,
                         req_state) -> int | None:
        """
        Returns the next-hop node id (or None if no valid hop).
        Uses RELiQ policy if loaded, else greedy fidelity.
        """
        curr_id  = int(req_state[2])
        visited  = set(req_state[3])
        target_id = req_state[1].id

        # Build neighbour list (nodes with entanglement to curr)
        neighbours = [
            n.id for n in self.topo.nodes
            if ent_matrix[curr_id][n.id] >= 1 and n.id not in visited
            and n.id != curr_id
        ]
        if not neighbours:
            return None

        if target_id in neighbours:
            return target_id  # route directly to destination if adjacent

        if self._policy is None:
            # Greedy fallback: pick neighbour with highest fidelity link
            best_n = max(neighbours,
                         key=lambda n: self._best_link_fidelity(curr_id, n))
            return best_n

        import torch
        obs = self._build_obs(req_idx)                  # (_OBS_SIZE,)
        t   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # shape: (1, 1, _OBS_SIZE)

        with torch.no_grad():
            try:
                adj = torch.zeros(1, 1, 1, dtype=torch.float32)  # dummy adj
                q_values = self._policy(t, adj)                   # (1,1,n_actions)
                q_values = q_values.squeeze().numpy()              # (n_actions,)
            except Exception:
                # Model signature mismatch — fall back to greedy
                return max(neighbours,
                           key=lambda n: self._best_link_fidelity(curr_id, n))

        # Map Q-values over available neighbours (sorted by node id as in training)
        neigh_sorted = sorted(neighbours)[:_NEIGHBOR_COUNT]
        if len(q_values.shape) == 0:
            q_values = np.array([float(q_values)])
        best_idx = int(np.argmax(q_values[:len(neigh_sorted)]))
        if best_idx < len(neigh_sorted):
            return neigh_sorted[best_idx]
        return neigh_sorted[0]

    # ── AlgorithmBase interface ───────────────────────────────────────────────
    def AddNewSDpairs(self):
        for (src, dst) in self.srcDstPairs:
            self.totalRequest += 1
            self.requests.append((src, dst, self.timeSlot))
        self.srcDstPairs = []
        index = 0
        self.requestState = []
        for request in self.requests:
            src, dst = request[0], request[1]
            if (src, dst) not in self.srcDstPairs:
                self.srcDstPairs.append((src, dst))
            self.requestState.append([src, dst, src.id, tuple([src.id]), index, False])
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
        successReq = 0
        ent_matrix = self.get_ent_graph_matrix()

        routed_links = set()  # conflict resolution: first-come-first-serve by request idx

        for req_idx, req_state in enumerate(self.requestState):
            if req_state[5]:   # already done
                continue

            src, dst = req_state[0], req_state[1]
            current   = req_state[2]
            visited   = set(req_state[3])

            hops = 0
            max_hops = 25
            fidelity  = 1.0
            routed    = False

            while hops < max_hops:
                next_hop = self._select_next_hop(req_idx, ent_matrix, req_state)
                if next_hop is None:
                    break

                link_key = (min(current, next_hop), max(current, next_hop))
                if link_key in routed_links:
                    # RELiQ paper conflict resolution: drop lower-priority request
                    break
                routed_links.add(link_key)

                # Apply Werner swap fidelity along path
                hop_fid = self._best_link_fidelity(current, next_hop)
                fidelity = fidelity * hop_fid + (1 - fidelity) * (1 - hop_fid) / 3.0

                # Update request state
                visited.add(next_hop)
                req_state[2] = next_hop
                req_state[3] = tuple(visited)
                current = next_hop
                hops += 1

                if current == dst.id:
                    routed = True
                    break

                ent_matrix = self.get_ent_graph_matrix()

            if routed:
                successReq += 1
                req_state[5] = True

        # Remove completed/expired requests
        self.requests = [
            r for i, r in enumerate(self.requests)
            if not self.requestState[i][5]
        ]
        self.requestState = [s for s in self.requestState if not s[5]]

        self.result.successfulRequest        += successReq
        self.result.successfulRequestPerRound.append(successReq)
        self.result.entanglementPerRound.append(successReq)
        self.result.fidelityPerRound.append(0)
        self.result.rewardPerRound.append(float(successReq))

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
