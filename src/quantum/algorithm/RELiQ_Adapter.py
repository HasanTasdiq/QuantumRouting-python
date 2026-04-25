"""
RELiQ_Adapter: wraps a trained RELiQ policy as an AlgorithmBase baseline.

Inherits p2() / tryEntanglement() / postProcess() from AlgorithmBase so that
the physical-layer simulation (qubit assignment, entanglement generation,
link fidelity) is identical to all other algorithms in the experiment.

p4() is overridden to use RELiQ's trained DQN for per-request routing decisions.

Physics: QuRA's Werner swap model is patched into RELiQ's QuantumLink at import
time (via entanglement_lifetime=10 and Werner formula — see quantum_network.py).

Observation format matches --request-based-observation --no-idle-action
--neighbors=6 training in src/reliq/train.py (plain DQN, no NetMon).
"""

import os
import sys

import numpy as np

# ── Path setup ────────────────────────────────────────────────────────────────
_algo_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir  = os.path.dirname(_algo_dir)
for _p in [_algo_dir, _src_dir,
           os.path.join(_src_dir, 'reliq')]:
    _ap = os.path.abspath(_p)
    if _ap not in sys.path:
        sys.path.insert(0, _ap)

from AlgorithmBase import AlgorithmBase, AlgorithmResult

# ── Default model path (absolute, relative to this file) ─────────────────────
_adapter_dir  = os.path.dirname(os.path.abspath(__file__))   # src/quantum/algorithm
_project_root = os.path.normpath(os.path.join(_adapter_dir, '../../..'))
_DEFAULT_MODEL_PATH = os.path.join(
    _project_root, 'runs_quantum', 'RELiQ_QuRAPhysics', 'model.pt'
)

# ── Obs constants (must match training config in src/reliq/train.py) ──────────
_MAX_REQUESTS   = int(os.environ.get("MAX_REQUESTS",   "100"))
_NEIGHBOR_COUNT = int(os.environ.get("NEIGHBOR_COUNT", "6"))
# per-neighbor features: swap_prob + avail_links + top_fidelity + one_hot(dest_state, 6)
_NEIGH_FEAT = 9
# base features: one_hot(id, MAX_REQ) + link_fid + path_len + n_ent_at_target
_BASE_FEAT  = _MAX_REQUESTS + 3
_OBS_SIZE   = _BASE_FEAT + _NEIGHBOR_COUNT * _NEIGH_FEAT


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
                 model_path=None):
        super().__init__(topo)
        self.name  = name
        self.requests         = []
        self.totalRequest     = 0
        self.totalWaitingTime = 0
        self.totalUsedQubits  = 0

        self._policy    = None
        self._model_path = model_path if model_path is not None else _DEFAULT_MODEL_PATH
        self._load_policy()

    # ── Model loading ─────────────────────────────────────────────────────────
    def _load_policy(self):
        import torch
        import torch.nn as nn

        if not os.path.exists(self._model_path):
            print(f"[RELiQ_Adapter] WARNING: no model at {self._model_path}"
                  " — using greedy-fidelity fallback.")
            return

        try:
            from reliq.model import DQN

            checkpoint = torch.load(self._model_path, map_location="cpu",
                                    weights_only=False)

            # Checkpoint format from reliq/util.py get_state_dict:
            #   {"type": "DQN", "state_dict": model.state_dict(), "args": args_dict}
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict  = checkpoint["state_dict"]
                model_type  = checkpoint.get("type", "DQN")
                if model_type != "DQN":
                    print(f"[RELiQ_Adapter] WARNING: checkpoint type is '{model_type}', "
                          f"expected 'DQN' — attempting DQN load anyway.")
            else:
                # Bare state_dict (legacy plain torch.save(model.state_dict(), ...))
                state_dict = checkpoint

            # Infer architecture from weight shapes rather than relying on saved args.
            # This is robust across different training runs.
            enc_weight_keys = sorted(
                k for k in state_dict if "encoder.linear_layers" in k and k.endswith(".weight")
            )
            in_feats  = state_dict[enc_weight_keys[0]].shape[1] if enc_weight_keys else _OBS_SIZE
            mlp_units = tuple(state_dict[k].shape[0] for k in enc_weight_keys)
            q_key     = next((k for k in state_dict if "q_net.fc.weight" in k), None)
            n_actions = state_dict[q_key].shape[0] if q_key else _NEIGHBOR_COUNT

            if in_feats != _OBS_SIZE:
                print(f"[RELiQ_Adapter] WARNING: checkpoint in_features={in_feats}, "
                      f"expected {_OBS_SIZE}. Obs mismatch — results may be noisy.")

            model = DQN(
                in_features=in_feats,
                mlp_units=mlp_units,
                num_actions=n_actions,
                activation_fn=nn.ReLU(),
            )
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[RELiQ_Adapter] missing keys ({len(missing)}): {missing[:3]}")
            if unexpected:
                print(f"[RELiQ_Adapter] unexpected keys ({len(unexpected)}): {unexpected[:3]}")
            model.eval()
            self._policy   = model
            self._n_actions = n_actions
            print(f"[RELiQ_Adapter] loaded DQN({in_feats}→{mlp_units}→{n_actions})"
                  f" from {self._model_path}")

        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[RELiQ_Adapter] model load failed ({e}), using greedy fallback.")
            self._policy = None

    # ── Observation construction ──────────────────────────────────────────────
    def _build_obs(self, req_idx: int, ent_matrix: np.ndarray) -> np.ndarray:
        """
        Build per-request observation matching --request-based-observation format.
        Returns float32 array of shape (_OBS_SIZE,).
        ent_matrix must be the shared, already-decremented matrix from p4().
        """
        req      = self.requestState[req_idx]
        curr_id  = int(req[2])
        path     = req[3]
        target_id = req[1].id
        start_id  = req[0].id
        visited   = set(path)

        ob = []

        # 1. Packet one-hot ID
        ob += _one_hot(req_idx, _MAX_REQUESTS)

        # 2. Current link fidelity (1.0 at beginning of hop)
        ob.append(1.0)

        # 3. Path hops completed
        ob.append(float(len(path) - 1))

        # 4. Entanglements at target
        ob.append(float(ent_matrix[target_id].sum()))

        # 5. Per-neighbor features (iterate nodes in order → matches training env)
        nodes_seen = 0
        for node in self.topo.nodes:
            if nodes_seen >= _NEIGHBOR_COUNT:
                break
            if ent_matrix[curr_id][node.id] >= 1:
                swap_prob = float(node.q)
                avail     = float(ent_matrix[curr_id][node.id])
                top_fid   = self._best_link_fidelity(curr_id, node.id)
                dest_oh   = _dest_state_oh(node.id, visited, start_id, target_id)
                ob += [swap_prob, avail, top_fid] + dest_oh
                nodes_seen += 1

        # Pad missing neighbours with all-zeros (matches training env padding)
        for _ in range(_NEIGHBOR_COUNT - nodes_seen):
            ob += [0.0] * _NEIGH_FEAT

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
        ent_matrix is the shared matrix maintained in p4(); must NOT be re-fetched here.
        """
        curr_id   = int(req_state[2])
        visited   = set(req_state[3])
        target_id = req_state[1].id

        # Candidate neighbours: entangled, unvisited, not self
        neighbours = [
            n.id for n in self.topo.nodes
            if ent_matrix[curr_id][n.id] >= 1
            and n.id not in visited
            and n.id != curr_id
        ]
        if not neighbours:
            return None

        # Route directly to destination when adjacent
        if target_id in neighbours:
            return target_id

        if self._policy is None:
            return max(neighbours, key=lambda n: self._best_link_fidelity(curr_id, n))

        import torch
        obs = self._build_obs(req_idx, ent_matrix)          # (_OBS_SIZE,)
        x   = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        # x shape: (1, 1, _OBS_SIZE) — matches DQN.forward(x, mask) signature

        with torch.no_grad():
            q_values = self._policy(x, x.new_zeros(1, 1, 1))  # mask unused by DQN
            q_values = q_values.squeeze().cpu().numpy()         # (_n_actions,)

        if q_values.ndim == 0:
            q_values = q_values.reshape(1)

        # Action i → the i-th neighbour in self.topo.nodes iteration order, which
        # must match _build_obs exactly so the Q-value indices line up correctly.
        neigh_set     = set(neighbours)
        neigh_ordered = [n.id for n in self.topo.nodes
                         if n.id in neigh_set][:_NEIGHBOR_COUNT]
        n_avail       = len(neigh_ordered)
        best_idx      = int(np.argmax(q_values[:n_avail]))
        return neigh_ordered[best_idx]

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

        # Build the entanglement matrix once and maintain it in-place throughout
        # p4(). After each committed hop (curr→next), we decrement the count so
        # subsequent hops and requests see the correct availability without having
        # to re-query topo (which won't reflect un-consumed links mid-timeslot).
        ent_matrix  = self.get_ent_graph_matrix()
        routed_links = set()   # conflict resolution: first-come-first-serve

        for req_idx, req_state in enumerate(self.requestState):
            if req_state[5]:   # already done
                continue

            dst     = req_state[1]
            current = int(req_state[2])
            visited = set(req_state[3])

            hops     = 0
            max_hops = 25
            fidelity = 1.0
            routed   = False

            while hops < max_hops:
                next_hop = self._select_next_hop(req_idx, ent_matrix, req_state)
                if next_hop is None:
                    break

                link_key = (min(current, next_hop), max(current, next_hop))
                if link_key in routed_links:
                    break
                routed_links.add(link_key)

                # Consume one entanglement on this link in the shared matrix
                ent_matrix[current][next_hop] = max(0.0, ent_matrix[current][next_hop] - 1)
                ent_matrix[next_hop][current] = max(0.0, ent_matrix[next_hop][current] - 1)

                # Accumulate Werner-swap fidelity along the path
                hop_fid  = self._best_link_fidelity(current, next_hop)
                fidelity = fidelity * hop_fid + (1 - fidelity) * (1 - hop_fid) / 3.0

                visited.add(next_hop)
                req_state[2] = next_hop
                req_state[3] = tuple(visited)
                current  = next_hop
                hops    += 1

                if current == dst.id:
                    routed = True
                    break

            if routed:
                successReq   += 1
                req_state[5]  = True

        # Remove completed requests
        self.requests     = [r for i, r in enumerate(self.requests)
                             if not self.requestState[i][5]]
        self.requestState = [s for s in self.requestState if not s[5]]

        self.result.successfulRequest            += successReq
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
