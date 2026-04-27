"""
Replay buffers for QuRA-v2.

NStepPERBuffer  — n-step returns + proportional Prioritized Experience Replay.
                  Used by Seq / Flock / Guard (single-agent DQN).

QMixEpisodicBuffer — collects full routing episodes (one per request-group
                     per timeslot) for QMIX joint training.
                     Uses ragged batches + mask; no MAX_REQUESTS padding.
"""
from __future__ import annotations
import random
from collections import deque

import numpy as np

# ── Training schedule (shared by all variants) ────────────────────────────────
import os
TRAINING_MODE = os.environ.get("TRAINING_MODE", "paper")

if TRAINING_MODE == "smoke":
    CAPACITY             = 5_000
    MIN_REPLAY           = 64
    MINIBATCH_SIZE       = 32
    STEP_BETWEEN_TRAIN   = 4
    N_STEP               = 3
    GAMMA                = 0.9
elif TRAINING_MODE == "mid":
    CAPACITY             = 50_000
    MIN_REPLAY           = 500
    MINIBATCH_SIZE       = 64
    STEP_BETWEEN_TRAIN   = 8
    N_STEP               = 3
    GAMMA                = 0.9
else:   # paper
    CAPACITY             = 200_000
    MIN_REPLAY           = 1_000
    MINIBATCH_SIZE       = 128
    STEP_BETWEEN_TRAIN   = 10
    N_STEP               = 3
    GAMMA                = 0.9

UPDATE_TARGET_EVERY  = 200
PER_ALPHA            = 0.6   # priority exponent
PER_BETA_START       = 0.4
PER_BETA_END         = 1.0
PER_BETA_STEPS       = 50_000


class NStepPERBuffer:
    """
    Proportional Prioritized Experience Replay with n-step returns.

    Transitions are stored as:
      (state_vec, action_idx, n_step_return, next_state_vec, done, gamma_n)

    state_vec    : 1-D numpy array (variable length — no fixed dims assumed)
    action_idx   : int (relative neighbor index, not absolute node id)
    n_step_return: float (sum of discounted rewards over n steps)
    next_state_vec: 1-D numpy array (state at t+n)
    done         : bool
    gamma_n      : γ^n (needed for target computation)
    """

    def __init__(self, capacity: int = CAPACITY, alpha: float = PER_ALPHA):
        self.capacity  = capacity
        self.alpha     = alpha
        self._buf: list     = [None] * capacity
        self._prios: np.ndarray = np.zeros(capacity, dtype=np.float32)
        self._ptr      = 0
        self._size     = 0

        # n-step accumulation window (deque of (state, action, reward, next, done))
        self._nstep: deque = deque()

    def _flush_nstep(self, bootstrap_state, bootstrap_done: bool) -> None:
        """Convert the n-step window into a single transition and store it."""
        if not self._nstep:
            return
        state0, action0, _, _, _ = self._nstep[0]
        ret      = 0.0
        gamma_n  = 1.0
        for _, _, r, _, d in self._nstep:
            ret     += gamma_n * r
            gamma_n *= GAMMA
            if d:
                break   # episode ended within window
        done_final = bootstrap_done

        self._store(state0, action0, ret, bootstrap_state, done_final, gamma_n)

    def _store(self, state, action: int, ret: float,
               next_state, done: bool, gamma_n: float) -> None:
        max_p = self._prios[:self._size].max() if self._size > 0 else 1.0
        self._buf[self._ptr]   = (state, action, ret, next_state, done, gamma_n)
        self._prios[self._ptr] = max_p
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push(self, state, action: int, reward: float,
             next_state, done: bool) -> None:
        self._nstep.append((state, action, reward, next_state, done))
        if len(self._nstep) >= N_STEP or done:
            # Flush oldest entry
            self._flush_nstep(
                self._nstep[-1][3],   # next_state of last step
                self._nstep[-1][4],   # done of last step
            )
            self._nstep.popleft()

    def flush_episode(self) -> None:
        """Call at episode end to drain remaining entries in the n-step window."""
        while self._nstep:
            self._flush_nstep(
                self._nstep[-1][3],
                self._nstep[-1][4],
            )
            self._nstep.popleft()

    def sample(self, batch_size: int, beta: float = PER_BETA_START
               ) -> tuple:
        """
        Returns (states, actions, returns, next_states, dones, gammas, weights, indices).
        weights: importance-sampling weights for PER bias correction.
        """
        if self._size == 0:
            raise RuntimeError("Buffer empty")

        prios = self._prios[:self._size] ** self.alpha
        probs = prios / prios.sum()
        idx   = np.random.choice(self._size, batch_size, replace=False, p=probs)

        weights = (self._size * probs[idx]) ** (-beta)
        weights /= weights.max()

        batch = [self._buf[i] for i in idx]
        s, a, r, ns, d, g = zip(*batch)

        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns).astype(np.float32),
            np.array(d, dtype=np.float32),
            np.array(g, dtype=np.float32),
            weights.astype(np.float32),
            idx,
        )

    def update_priorities(self, indices: np.ndarray,
                          td_errors: np.ndarray) -> None:
        for i, e in zip(indices, td_errors):
            self._prios[i] = abs(float(e)) + 1e-6

    def __len__(self) -> int:
        return self._size


class QMixEpisodicBuffer:
    """
    Episodic replay buffer for QMIX (Hive variant).

    Each episode = one timeslot's routing decisions across all active requests.
    Stores ragged episodes; samples a batch of episodes and pads to the
    maximum episode length in that batch (not a global MAX_REQUESTS constant).

    Episode entry:
      states     : list of (state_vec,)     per request
      actions    : list of int              per request (relative neighbor idx)
      rewards    : list of float            per request
      next_states: list of (state_vec,)
      req_feats  : list of (2D,) concat(emb_u, emb_dst) per request
      global_state     : (G,) compact global state
      next_global_state: (G,)
      done       : bool
    """

    def __init__(self, capacity: int = CAPACITY // 10):
        self.capacity = capacity
        self._buf: list = []
        self._ptr       = 0
        self._size      = 0

    def push(self, states: list, actions: list[int], rewards: list[float],
             next_states: list, req_feats: list,
             global_state: np.ndarray, next_global_state: np.ndarray,
             done: bool) -> None:
        ep = (states, actions, rewards, next_states,
              req_feats, global_state, next_global_state, done)
        if len(self._buf) < self.capacity:
            self._buf.append(ep)
        else:
            self._buf[self._ptr] = ep
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(self, batch_size: int) -> tuple:
        """
        Returns a padded batch.

        Pads to max episode length R_max in this batch (not global constant).

        Returns:
          ps    : (B, R_max, state_dim) float32
          pa    : (B, R_max) int64
          mask  : (B, R_max) float32 — 1 for real, 0 for pad
          pns   : (B, R_max, state_dim)
          rf    : (B, R_max, 2*D) request features
          gs    : (B, G)
          ngs   : (B, G)
          rew   : (B, 1)   mean reward across requests
          dones : (B, 1)
        """
        episodes = random.sample(self._buf[:self._size], batch_size)

        R_max     = max(len(ep[0]) for ep in episodes)
        state_dim = len(episodes[0][0][0]) if episodes[0][0] else 1
        req_dim   = len(episodes[0][4][0]) if episodes[0][4] else 1
        G_dim     = len(episodes[0][5])

        ps  = np.zeros((batch_size, R_max, state_dim), dtype=np.float32)
        pa  = np.zeros((batch_size, R_max),            dtype=np.int64)
        mask= np.zeros((batch_size, R_max),            dtype=np.float32)
        pns = np.zeros((batch_size, R_max, state_dim), dtype=np.float32)
        rf  = np.zeros((batch_size, R_max, req_dim),   dtype=np.float32)
        gs  = np.zeros((batch_size, G_dim),            dtype=np.float32)
        ngs = np.zeros((batch_size, G_dim),            dtype=np.float32)
        rew = np.zeros((batch_size, 1),                dtype=np.float32)
        don = np.zeros((batch_size, 1),                dtype=np.float32)

        for b, (states, actions, rewards, nstates,
                req_feats, gstate, ngstate, done) in enumerate(episodes):
            R = len(states)
            ps[b, :R]  = np.stack(states)
            pa[b, :R]  = actions
            mask[b, :R]= 1.0
            pns[b, :R] = np.stack(nstates)
            rf[b, :R]  = np.stack(req_feats)
            gs[b]      = gstate
            ngs[b]     = ngstate
            rew[b, 0]  = float(np.mean(rewards))
            don[b, 0]  = float(done)

        return ps, pa, mask, pns, rf, gs, ngs, rew, don

    def __len__(self) -> int:
        return self._size
