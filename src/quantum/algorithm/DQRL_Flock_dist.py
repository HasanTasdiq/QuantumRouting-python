"""
QuRA-Flock (Distributed): Parallel multi-agent routing with NO conflict resolution.

All requests route in parallel using the distributed FastAPI agent (port 8080/8000).
When multiple requests claim the same entangled link, the first to atomically
decrement the ent_matrix wins; later arrivals fall back to numtry retries.

Key difference from QuRA-Seq (DQRL_dist1.py):
  - Node locks are NOT acquired — requests compete without serialisation.
  - Higher throughput in capacity-limited regimes; more conflicts in dense load.

Inherits all infrastructure from QuRA_DQRL_DIST (shared_memory, Redis, QMIX agent).
"""

import time
import random
import threading
from multiprocessing import shared_memory, Manager

import numpy as np

from DQRL_dist1 import QuRA_DQRL_DIST
from topo.mp_helper import executor as executor2, node_locks


class QuRA_Flock_DIST(QuRA_DQRL_DIST):
    def __init__(self, topo, param=None, name='QuRA_Flock_DIST'):
        super().__init__(topo, param=param, name=name)

    # ------------------------------------------------------------------
    def route_schedule_single(self, args):
        """
        Per-request routing loop — same as Seq but without node-lock acquisition.
        Requests race on ent_matrix without coordination.
        """
        if len(args) == 7:
            node_matrix_info, req_matrix_info, dist_matrix, q_matrix, reqState, _node_locks, precomputed_action = args
        else:
            node_matrix_info, req_matrix_info, dist_matrix, q_matrix, reqState, _node_locks = args
            precomputed_action = None

        shm_name, shape, dtype = node_matrix_info
        shm = shared_memory.SharedMemory(name=shm_name)
        ent_matrix = np.ndarray(shape, dtype=dtype, buffer=shm.buf)

        shm_name2, shape2, dtype2 = req_matrix_info
        shm2 = shared_memory.SharedMemory(name=shm_name2)
        req_matrix = np.ndarray(shape2, dtype=dtype2, buffer=shm2.buf)

        src, dst, current_node_id, path, index, checked = reqState
        path = list(path)
        success = False
        good_to_search = True
        numtry = 0
        maxTry = self.maxTry
        fidelity = 1.0
        actions = []
        a_id = 0
        total_fidelity = 0.0

        while good_to_search and not success and numtry <= maxTry:
            try:
                if precomputed_action is not None:
                    result = (None, precomputed_action)
                    precomputed_action = None
                else:
                    result = self.get_action(reqState, ent_matrix, req_matrix, dist_matrix, self.timeSlot)
            except Exception:
                import traceback; traceback.print_exc()
                break

            if result is None:
                break

            current_state, next_node_id = result
            current_state = (ent_matrix.copy().tolist(), req_matrix.copy().tolist())

            if current_node_id == next_node_id:
                numtry += 1
                if numtry > maxTry:
                    good_to_search = False
                req_done = not good_to_search
                reqState = (src, dst, next_node_id, tuple(path), index, req_done)
                self.requestState[index] = reqState
                req_matrix[index][2] = next_node_id
                req_matrix[index][5] = req_done
                req_matrix[len(self.requestState) + index][next_node_id] = 1
                continue

            # --- No locks: attempt to claim ent_matrix entry atomically ---
            if ent_matrix[current_node_id][next_node_id] >= 1:
                ent_matrix[current_node_id][next_node_id] -= 1
                ent_matrix[next_node_id][current_node_id] -= 1
                numtry = 0
            else:
                numtry += 1
                if numtry > maxTry:
                    good_to_search = False
                req_done = not good_to_search
                reqState = (src, dst, next_node_id, tuple(path), index, req_done)
                self.requestState[index] = reqState
                req_matrix[index][2] = next_node_id
                req_matrix[index][5] = req_done
                req_matrix[len(self.requestState) + index][next_node_id] = 1
                reward = -1
                next_state = (ent_matrix.copy().tolist(), req_matrix.copy().tolist())
                T = [r for r in self.requestState if not r[5]]
                done_episode = req_done and len(T) == 1
                actions.append([index, current_node_id, next_node_id, current_state,
                                 done_episode, self.timeSlot, reward, next_state, dist_matrix, a_id])
                a_id += 1
                continue

            if next_node_id == current_node_id or next_node_id in path:
                good_to_search = False

            path.append(next_node_id)
            req_done = (not good_to_search) or success
            reqState = (src, dst, next_node_id, tuple(path), index, req_done)
            self.requestState[index] = reqState
            req_matrix[index][2] = next_node_id
            req_matrix[index][5] = req_done
            req_matrix[len(self.requestState) + index][next_node_id] = 1

            if next_node_id == dst.id and good_to_search:
                success = True
                good_to_search = False

            current_node_id = next_node_id
            reward = -1
            if req_done:
                if success:
                    reward = 10
                    total_fidelity += fidelity
                else:
                    for i in range(1, len(path)):
                        ent_matrix[path[i - 1]][path[i]] += 1
                        ent_matrix[path[i]][path[i - 1]] += 1
                    reward = -10

            next_state = (ent_matrix.copy().tolist(), req_matrix.copy().tolist())
            T = [r for r in self.requestState if not r[5]]
            done_episode = req_done and len(T) == 1
            actions.append([index, current_node_id, next_node_id, current_state,
                             done_episode, self.timeSlot, reward, next_state, dist_matrix, a_id])
            a_id += 1

        try:
            shm.close()
            shm2.close()
        except Exception:
            pass

        return (success, actions, index, path)
