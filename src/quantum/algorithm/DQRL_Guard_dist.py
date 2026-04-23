"""
QuRA-Guard (Distributed): Parallel routing with priority-based conflict resolution.

All agents propose next-hop actions simultaneously (Phase 1 — no locks).
Actions are sorted by priority score before resources are committed (Phase 2).

Priority score (paper Eq. 16):
    score_i = MU_GUARD * Q_i + (1 - MU_GUARD) / SP_i
where SP_i is shortest-hop count from current node to destination.

Higher-priority agents get first access to entangled links.
Lower-priority agents whose link has already been claimed are blocked for that hop.

Inherits all infrastructure from QuRA_DQRL_DIST (shared_memory, Redis, QMIX agent).
"""

import time
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import shared_memory, Manager

import numpy as np

from DQRL_dist1 import QuRA_DQRL_DIST, _INFERENCE_MODE
from topo.mp_helper import node_locks

MU_GUARD = 0.5  # balances Q-value vs path-length urgency; tune in [0, 1]


class QuRA_Guard_DIST(QuRA_DQRL_DIST):
    def __init__(self, topo, param=None, name='QuRA_Guard_DIST'):
        super().__init__(topo, param=param, name=name)
        self.mu_guard = MU_GUARD

    def _priority_score(self, q_val, current_node_id, dst_id):
        """score_i = MU * Q_i + (1 - MU) / SP_i"""
        src_node = self.topo.nodes[current_node_id]
        dst_node = self.topo.nodes[dst_id]
        sp = self.topo.hopsAway(src_node, dst_node, 'Hop')
        sp = max(sp, 1)
        return self.mu_guard * q_val + (1.0 - self.mu_guard) / sp

    # ------------------------------------------------------------------
    def p4(self):
        """
        Override p4 to implement Guard-style two-phase routing.

        Phase 1: All requests independently query the agent (parallel).
        Phase 2: Sort proposed actions by priority; apply in order,
                 blocking lower-priority agents that claimed the same link.
        """
        from multiprocessing import Manager as MgrClass
        global node_locks

        print('start p4 Guard', self.name)

        if len(self.srcDstPairs) == 0:
            self.filterReqeuest()
            self.printResult()
            self.result.rewardPerRound.append(0)
            return self.result

        t = time.time()

        node_matrix_info = self.get_ent_graph_matrix_info()
        req_matrix_info = self.req_matrix_info()
        dist_matrix = self.dist_matrix()
        q_matrix = self.q_matrix()

        # --- Phase 1: batch predict for all requests at once (Improvement 2) ---
        try:
            ent_snap = self.get_ent_graph_matrix()
            req_snap = self.req_matrix()
            batch_actions = self.call_learn_and_predict_batch_api(
                self.requestState, ent_snap, req_snap, dist_matrix, self.timeSlot
            )
        except Exception:
            import traceback; traceback.print_exc()
            batch_actions = {}

        # Build proposals list directly from batch results
        args = [(node_matrix_info, req_matrix_info, dist_matrix, q_matrix,
                 reqState, node_locks) for reqState in self.requestState]
        proposals = []
        for i, reqState in enumerate(self.requestState):
            req_idx = reqState[4]
            if req_idx in batch_actions:
                # Fake a proposal tuple using the batch-predicted action
                # q_val defaults to 1.0 (we don't have the Q value from batch)
                ent_for_state = self.get_ent_graph_matrix()
                req_for_state = self.req_matrix()
                current_state_snap = (ent_for_state.tolist(), req_for_state.tolist())
                proposals.append((reqState, batch_actions[req_idx], 1.0, current_state_snap))
            else:
                proposals.append(None)

        # proposals: list of (reqState, next_node_id, q_val, current_state) or None

        # --- Phase 2: priority sort and resource commitment ---
        # Load the shared ent_matrix for atomic commitment
        shm_name, shape, dtype = node_matrix_info
        shm = shared_memory.SharedMemory(name=shm_name)
        ent_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        shm_name2, shape2, dtype2 = req_matrix_info
        shm2 = shared_memory.SharedMemory(name=shm_name2)
        req_matrix_shm = np.ndarray(shape2, dtype=dtype2, buffer=shm2.buf)

        # Score and sort
        scored = []
        for prop in proposals:
            if prop is None:
                continue
            reqState, next_node_id, q_val, current_state = prop
            src, dst, current_node_id, path, index, checked = reqState
            score = self._priority_score(q_val, current_node_id, dst.id)
            scored.append((score, prop))
        scored.sort(key=lambda x: x[0], reverse=True)

        claimed = set()  # (min_id, max_id) edges committed this round
        results_guard = []  # (success, actions, index, path)
        skipped_indices = set()

        for (score, prop) in scored:
            reqState, next_node_id, q_val, current_state = prop
            src, dst, current_node_id, path, index, checked = reqState
            path = list(path)

            edge_key = (min(current_node_id, next_node_id),
                        max(current_node_id, next_node_id))

            # Guard: block if link already claimed by higher-priority agent
            if edge_key in claimed or ent_matrix[current_node_id][next_node_id] < 1:
                skipped_indices.add(index)
                results_guard.append((False, [], index, path))
                continue

            # Commit
            ent_matrix[current_node_id][next_node_id] -= 1
            ent_matrix[next_node_id][current_node_id] -= 1
            claimed.add(edge_key)

            # Update path and request state
            path.append(next_node_id)
            success = (next_node_id == dst.id)
            req_done = success or (len(path) >= self.hopCountThreshold)
            reqState_new = (src, dst, next_node_id, tuple(path), index, req_done)
            self.requestState[index] = reqState_new
            req_matrix_shm[index][2] = next_node_id
            req_matrix_shm[index][5] = req_done
            req_matrix_shm[len(self.requestState) + index][next_node_id] = 1

            reward = 10 if success else (-10 if req_done else -1)
            if not success and req_done:
                for i in range(1, len(path)):
                    ent_matrix[path[i - 1]][path[i]] += 1
                    ent_matrix[path[i]][path[i - 1]] += 1

            next_state = (ent_matrix.copy().tolist(), req_matrix_shm.copy().tolist())
            T = [r for r in self.requestState if not r[5]]
            done_episode = req_done and len(T) == 1
            action_entry = [index, current_node_id, next_node_id, current_state,
                            done_episode, self.timeSlot, reward, next_state, dist_matrix, 0]
            results_guard.append((success, [action_entry], index, path))

        try:
            shm.close()
            shm2.close()
        except Exception:
            pass

        # Resolve conflicts and collect actions
        node_matrix = self.get_ent_graph_matrix()
        try:
            successReq = self.resolve_conflict(results_guard, q_matrix, node_matrix.copy())
        except Exception:
            import traceback; traceback.print_exc()
            successReq = sum(r[0] for r in results_guard)

        actions = []
        req_matrix = self.req_matrix()
        for r in results_guard:
            for a in r[1]:
                actions.append(a)

        self.result.successfulRequestPerRound.append(successReq)
        self.result.entanglementPerRound.append(successReq)
        self.result.fidelityPerRound.append(0)
        self.result.successfulRequest += successReq

        print('*=========Guard route time ======== ', time.time() - t)
        self.filterReqeuest()
        self.printResult()

        if self.timeSlot < 100000 and not _INFERENCE_MODE:
            # Skip training calls in inference mode — model is frozen.
            try:
                self.run_async_in_thread(self.call_update_reward(
                    successful_requests=self.result.successfulRequestPerRound[-1],
                    timeSlot=self.timeSlot,
                    actions=actions
                ))
            except Exception:
                import traceback; traceback.print_exc()

        self.result.rewardPerRound.append(0)
        return self.result

    def _propose_action(self, args):
        """
        Phase 1: query the agent for a proposed next-hop but do NOT commit
        any resources. Returns (reqState, next_node_id, q_val, current_state)
        or None if no action available.
        """
        node_matrix_info, req_matrix_info, dist_matrix, q_matrix, reqState, _locks = args

        shm_name, shape, dtype = node_matrix_info
        shm = shared_memory.SharedMemory(name=shm_name)
        ent_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        shm_name2, shape2, dtype2 = req_matrix_info
        shm2 = shared_memory.SharedMemory(name=shm_name2)
        req_matrix = np.ndarray(shape2, dtype=dtype2, buffer=shm2.buf)

        try:
            result = self.get_action(reqState, ent_matrix, req_matrix, dist_matrix, self.timeSlot)
            if result is None:
                return None
            current_state, next_node_id = result
            # Snapshot BEFORE closing shm (avoid use-after-close)
            current_state_snap = (ent_matrix.copy().tolist(), req_matrix.copy().tolist())
        except Exception:
            import traceback; traceback.print_exc()
            return None
        finally:
            try:
                shm.close(); shm2.close()
            except Exception:
                pass

        # Approximate Q-value (use 1.0 as default when not returned by API)
        q_val = 1.0
        return (reqState, next_node_id, q_val, current_state_snap)
