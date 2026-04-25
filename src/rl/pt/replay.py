"""
Replay buffer for PyTorch QuRA port.

Two classes:
  ReplayBuffer        — single-agent transitions (Seq / Flock / Guard)
  QMixEpisodeBuffer   — joint timeslot transitions (Hive / QMIX)

Both use the same REPLAY_MEMORY_SIZE / MINIBATCH_SIZE constants
as the original TF helper so training dynamics are preserved.
"""
import random
from collections import deque

import numpy as np


# ── Constants (mirror dist_agent_helper.py) ─────────────────────────────────
import os

TRAINING_MODE = os.environ.get("TRAINING_MODE", "paper")

if TRAINING_MODE == "paper":
    START_EPSILON_DECAYING = 3000
    END_EPSILON_DECAYING   = 8000
    REPLAY_MEMORY_SIZE     = 50_000
    MIN_REPLAY_MEMORY_SIZE = 512
    MINIBATCH_SIZE         = 512
    UPDATE_TARGET_EVERY    = 100
elif TRAINING_MODE == "mid":
    START_EPSILON_DECAYING = 200
    END_EPSILON_DECAYING   = 1500
    REPLAY_MEMORY_SIZE     = 20_000
    MIN_REPLAY_MEMORY_SIZE = 128
    MINIBATCH_SIZE         = 64
    UPDATE_TARGET_EVERY    = 20
else:  # smoke
    START_EPSILON_DECAYING = 10
    END_EPSILON_DECAYING   = 40
    REPLAY_MEMORY_SIZE     = 500
    MIN_REPLAY_MEMORY_SIZE = 20
    MINIBATCH_SIZE         = 8
    UPDATE_TARGET_EVERY    = 10

MAX_REQUESTS_SMOKE = 15
MAX_REQUESTS_PAPER = 100


# ── Single-agent replay (Seq / Flock / Guard) ────────────────────────────────
class ReplayBuffer:
    def __init__(self, capacity: int = REPLAY_MEMORY_SIZE):
        self._buf = deque(maxlen=capacity)

    def push(self, state, action: int, reward: float,
             next_state, done: bool):
        self._buf.append((
            np.asarray(state,      dtype=np.float32),
            int(action),
            float(reward),
            np.asarray(next_state, dtype=np.float32),
            float(done),
        ))

    def sample(self, batch_size: int):
        batch = random.sample(self._buf, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.stack(states),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.stack(next_states),
            np.array(dones,   dtype=np.float32),
        )

    def __len__(self):
        return len(self._buf)


# ── Joint timeslot replay (Hive / QMIX) ─────────────────────────────────────
class QMixEpisodeBuffer:
    """
    Stores one joint transition per timeslot, matching the structure used
    by DQRLAgentDist.train_qmix():

        (ts_states, ts_actions, scaled_rewards, ts_next_states,
         global_state, next_global_state, done)

    ts_states / ts_actions / ts_next_states are lists of per-request arrays.
    global_state / next_global_state are flat numpy arrays.
    """
    def __init__(self, capacity: int = REPLAY_MEMORY_SIZE):
        self._buf = deque(maxlen=capacity)

    def push(self, ts_states, ts_actions, ts_rewards, ts_next_states,
             global_state, next_global_state, done: bool):
        self._buf.append((
            list(ts_states),
            list(ts_actions),
            list(ts_rewards),
            list(ts_next_states),
            np.asarray(global_state,      dtype=np.float32).flatten(),
            np.asarray(next_global_state, dtype=np.float32).flatten(),
            float(done),
        ))

    def sample(self, batch_size: int, max_requests: int, state_dim: int):
        """
        Returns padded numpy arrays ready for the QMIX train step:
          padded_states      (B, max_requests, state_dim)
          padded_actions     (B, max_requests)
          mask               (B, max_requests)
          padded_next_states (B, max_requests, state_dim)
          global_states      (B, g_dim)
          next_global_states (B, g_dim)
          rewards            (B, 1)
          dones              (B, 1)
        """
        batch = random.sample(self._buf, batch_size)
        B = batch_size

        g_dim = batch[0][4].shape[0]

        ps       = np.zeros((B, max_requests, state_dim), dtype=np.float32)
        pa       = np.zeros((B, max_requests),            dtype=np.int64)
        mask     = np.zeros((B, max_requests),            dtype=np.float32)
        pns      = np.zeros((B, max_requests, state_dim), dtype=np.float32)
        gs       = np.zeros((B, g_dim),                   dtype=np.float32)
        ngs      = np.zeros((B, g_dim),                   dtype=np.float32)
        rewards  = np.zeros((B, 1),                       dtype=np.float32)
        dones    = np.zeros((B, 1),                       dtype=np.float32)

        for i, (ss, aa, rr, ns, g, ng, d) in enumerate(batch):
            n = min(len(ss), max_requests)
            if n > 0:
                ps[i,  :n] = np.stack(ss[:n]).astype(np.float32)
                pa[i,  :n] = np.array(aa[:n], dtype=np.int64)
                pns[i, :n] = np.stack(ns[:n]).astype(np.float32)
                mask[i, :n] = 1.0
            gs[i]      = g
            ngs[i]     = ng
            rewards[i] = np.sum(rr)
            dones[i]   = d

        return ps, pa, mask, pns, gs, ngs, rewards, dones

    def __len__(self):
        return len(self._buf)
