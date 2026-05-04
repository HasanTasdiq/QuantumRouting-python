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
_RELIQ_TTL      = int(os.environ.get("RELIQ_TTL",      "20"))
_ZERO_PACKET_ID = os.environ.get("RELIQ_ZERO_PACKET_ID", "1") == "1"
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
        self._pair_to_packet_id = {}
        self._next_packet_id    = 0
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
    def _packet_id_for(self, start_id: int, target_id: int) -> int:
        """
        RELiQ was trained with fixed persistent packet IDs.  QuRA generates
        requests without such identities, so by default we zero the ID slot
        instead of injecting a misleading per-timeslot index.  Set
        RELIQ_ZERO_PACKET_ID=0 to use a stable first-seen pair mapping.
        """
        if _ZERO_PACKET_ID:
            return -1
        key = (int(start_id), int(target_id))
        if key not in self._pair_to_packet_id:
            self._pair_to_packet_id[key] = self._next_packet_id % _MAX_REQUESTS
            self._next_packet_id += 1
        return self._pair_to_packet_id[key]

    def _neighbor_slots(self, curr_id: int,
                        ent_matrix: np.ndarray,
                        fid_matrix: np.ndarray) -> list:
        """
        Return RELiQ-style neighbor slots in physical node.links order.

        Original RELiQ observations iterate node.edges, not sorted node IDs.
        QuRA has one Link object per elementary link, so collapse duplicates
        while preserving insertion order.  Each slot is
        (neighbor_id, available_entanglements, top_fidelity).
        """
        slots = []
        seen = set()
        curr_node = self.topo.nodes[curr_id]
        for link in curr_node.links:
            other = link.theOtherEndOf(curr_node)
            nid = int(other.id)
            if nid in seen:
                continue
            seen.add(nid)
            slots.append((
                nid,
                float(ent_matrix[curr_id][nid]),
                float(fid_matrix[curr_id][nid]),
            ))
            if len(slots) >= _NEIGHBOR_COUNT:
                break
        return slots

    def _build_obs(self, req_idx: int,
                   ent_matrix: np.ndarray,
                   fid_matrix: np.ndarray,
                   current_fidelity: float) -> tuple[np.ndarray, list]:
        """
        Build per-request observation matching --request-based-observation format.
        Returns float32 array of shape (_OBS_SIZE,).
        ent_matrix / fid_matrix must be the shared, already-decremented matrices
        from p4() (built once via _get_matrices()).
        """
        req       = self.requestState[req_idx]
        curr_id   = int(req[2])
        path      = req[3]
        target_id = req[1].id
        start_id  = req[0].id
        visited   = set(path)

        ob = []

        # 1. Packet one-hot ID.  See _packet_id_for for why this is usually zero.
        ob += _one_hot(self._packet_id_for(start_id, target_id), _MAX_REQUESTS)

        # 2. Current request/link fidelity.
        ob.append(float(current_fidelity))

        # 3. Hops since last breakpoint in original RELiQ; no breakpoints in
        #    QuRA adapter, so use clamped path hops.
        ob.append(float(min(len(path) - 1, _RELIQ_TTL)))

        # 4. Entanglements at target
        ob.append(float(ent_matrix[target_id].sum()))

        # 5. Per-neighbor features in RELiQ's edge-slot order.
        neighbor_slots = self._neighbor_slots(curr_id, ent_matrix, fid_matrix)
        for nid, avail, top_fid in neighbor_slots:
            node      = self.topo.nodes[nid]
            swap_prob = float(node.q)
            dest_oh   = _dest_state_oh(int(nid), visited, start_id, target_id)
            ob += [swap_prob, avail, top_fid] + dest_oh

        # Pad missing neighbours exactly like RELiQ training:
        # [0, 0, 0] + one_hot(dest_state=1, 6)
        for _ in range(_NEIGHBOR_COUNT - len(neighbor_slots)):
            ob += [0.0, 0.0, 0.0] + _one_hot(1, 6)

        return np.array(ob, dtype=np.float32), neighbor_slots

    def _get_matrices(self) -> tuple:
        """Single pass over links → (ent_matrix, fid_matrix). Called once per p4()."""
        n   = len(self.topo.nodes)
        ent = np.zeros((n, n), dtype=np.float32)
        fid = np.zeros((n, n), dtype=np.float32)
        for link in self.topo.links:
            if link.isEntangled(self.timeSlot) and link.notSwapped() and not link.taken:
                i, j = link.n1.id, link.n2.id
                ent[i][j] += 1;          ent[j][i] += 1
                if link.fidelity > fid[i][j]:
                    fid[i][j] = link.fidelity
                    fid[j][i] = link.fidelity
        return ent, fid

    # kept for backward-compat callers outside p4(); not used in the hot path
    def get_ent_graph_matrix(self) -> np.ndarray:
        ent, _ = self._get_matrices()
        return ent

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
        self.result.numOfTimeslot += 1
        if len(self.srcDstPairs) > 0:
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
        import torch

        _F_MIN = float(os.environ.get("F_MIN", "0.7"))

        # Build both matrices in a single link pass (fixes A: no per-hop link scan).
        ent_matrix, fid_matrix = self._get_matrices()
        routed_links = set()
        success_req  = 0
        success_fid  = 0
        total_fid    = 0.0

        # Werner fidelity tracking (initialised to 1.0 for active requests)
        fidelity_track = {i: 1.0 for i in range(len(self.requestState))}

        # Batched-hop routing: one DQN forward per hop step across ALL requests,
        # mirroring the pattern in local_trainer.py. Reduces PyTorch invocations
        # from O(N_requests × N_hops) to O(N_hops) (fixes B).
        for _hop in range(25):
            # Collect all requests that still have moves available.
            # np.nonzero gives C-level neighbor discovery (fixes C).
            pending = []
            for ridx, req_state in enumerate(self.requestState):
                if req_state[5]:
                    continue
                curr    = int(req_state[2])
                visited = set(req_state[3])
                raw     = np.nonzero(ent_matrix[curr])[0]
                neighs  = [int(i) for i in raw if i not in visited and i != curr]
                if neighs:
                    pending.append((ridx, req_state, req_state[1].id, curr, visited, neighs))

            if not pending:
                break

            # Separate "go direct" from "need Q-values"
            direct_hops = {}   # ridx → next_hop
            greedy_batch = []  # (ridx, dst_id, curr, neighs, obs)

            for ridx, req_state, dst_id, curr, visited, neighs in pending:
                if dst_id in neighs:
                    direct_hops[ridx] = dst_id
                elif self._policy is None:
                    # greedy fallback: best fidelity neighbor (O(1) with fid_matrix)
                    direct_hops[ridx] = max(neighs,
                                            key=lambda n: float(fid_matrix[curr][n]))
                else:
                    obs, slots = self._build_obs(
                        ridx, ent_matrix, fid_matrix,
                        fidelity_track.get(ridx, 1.0))
                    greedy_batch.append((ridx, dst_id, curr, neighs, obs, slots))

            # One batched DQN call for all requests needing Q-value decisions
            if greedy_batch and self._policy is not None:
                obs_stack = np.stack([item[4] for item in greedy_batch])      # (N, _OBS_SIZE)
                x = torch.tensor(obs_stack, dtype=torch.float32).unsqueeze(1)  # (N, 1, _OBS_SIZE)
                with torch.no_grad():
                    q_batch = self._policy(x, x.new_zeros(x.shape[0], 1, 1))  # (N, n_actions)
                    q_batch = q_batch.squeeze(1).cpu().numpy()                 # (N, n_actions)

                for i, (ridx, dst_id, curr, neighs, _, slots) in enumerate(greedy_batch):
                    q = q_batch[i]
                    # Action index → RELiQ neighbor slot.  Apply an inference
                    # action mask so invalid/empty slots cannot be chosen.
                    valid = []
                    valid_set = set(neighs)
                    for action_idx, (nid, avail, _) in enumerate(slots):
                        if nid in valid_set and avail > 0:
                            valid.append((action_idx, nid))
                    if not valid:
                        continue
                    _, best_nid = max(valid, key=lambda item: q[item[0]])
                    direct_hops[ridx] = best_nid

            # Apply decided hops
            for ridx, req_state, dst_id, curr, visited, neighs in pending:
                if ridx not in direct_hops:
                    continue
                next_hop = direct_hops[ridx]
                link_key = (min(curr, next_hop), max(curr, next_hop))
                if link_key in routed_links:
                    continue
                routed_links.add(link_key)

                ent_matrix[curr][next_hop] = max(0.0, ent_matrix[curr][next_hop] - 1)
                ent_matrix[next_hop][curr] = max(0.0, ent_matrix[next_hop][curr] - 1)

                # Werner-swap fidelity update
                f_hop = float(fid_matrix[curr][next_hop])
                f_old = fidelity_track.get(ridx, 1.0)
                f_new = f_old * f_hop + (1.0 - f_old) * (1.0 - f_hop) / 3.0
                fidelity_track[ridx] = f_new

                visited.add(next_hop)
                req_state[2] = next_hop
                req_state[3] = tuple(visited)

                if next_hop == dst_id:
                    req_state[5] = True
                    # F_min gate: only count as success if fidelity meets threshold
                    if f_new >= _F_MIN:
                        success_req += 1
                        success_fid += 1
                        total_fid   += f_new

        # Drop all requests: served ones are done, unserved ones are dropped (no carryover)
        self.requests     = []
        self.requestState = []

        self.result.successfulRequest            += success_req
        self.result.successfulRequestPerRound.append(success_req)
        self.result.entanglementPerRound.append(success_req)
        avg_fid = total_fid / max(success_fid, 1)
        self.result.fidelityPerRound.append(avg_fid)
        self.result.rewardPerRound.append(float(success_req))

        self.printResult()
        return self.result

    def printResult(self):
        self.topo.clearAllEntanglements()
        if self.totalRequest > 0:
            self.result.waitingTime = self.totalWaitingTime / self.totalRequest
            self.result.usedQubits  = self.totalUsedQubits  / self.totalRequest
        self.result.remainRequestPerRound.append(len(self.requests))
