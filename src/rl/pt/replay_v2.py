"""
Replay buffers for QuRA-v2.

NStepPERBuffer  — n-step returns + proportional Prioritized Experience Replay.
                  Used by Seq / Flock / Guard (single-agent DQN).

QMixEpisodicBuffer — legacy QMIX buffer kept for compatibility.

MAPPORolloutBuffer — on-policy trajectories for Hive's MAPPO actor-critic.
"""
from __future__ import annotations
import random
from collections import deque

import numpy as np

# ── Training schedule (shared by all variants) ────────────────────────────────
import os
TRAINING_MODE = os.environ.get("TRAINING_MODE", "paper")

# Hyperparameter rationale (10×10 grid, E[hops]≈4.2):
#   N_STEP=8       — 1.5× E[hops] covers ~95% of paths (some reach 10+ hops)
#   GAMMA=0.99     — γ^4.2=0.959 preserves end-to-end credit; γ=0.95 squashes to 0.81
#   MIN_REPLAY     — 10× MINIBATCH for stable initial Q-fit
#   CAPACITY       — holds full epsilon-decay era (~520 trans/slot × decay window)
if TRAINING_MODE == "smoke":
    CAPACITY             = 50_000
    MIN_REPLAY           = 640
    MINIBATCH_SIZE       = 64
    STEP_BETWEEN_TRAIN   = 4
    N_STEP               = 8
    GAMMA                = 0.99
elif TRAINING_MODE == "mid":
    CAPACITY             = 200_000
    MIN_REPLAY           = 1_280
    MINIBATCH_SIZE       = 128
    STEP_BETWEEN_TRAIN   = 8
    N_STEP               = 8
    GAMMA                = 0.99
elif TRAINING_MODE == "long":
    # 1M timeslots × ~111 trans/slot ÷ 50 = ~2.2M grad updates
    CAPACITY             = 1_000_000
    MIN_REPLAY           = 5_000
    MINIBATCH_SIZE       = 256
    STEP_BETWEEN_TRAIN   = 50
    N_STEP               = 8
    GAMMA                = 0.99
else:   # paper (20k timeslots)
    CAPACITY             = 200_000   # 1M caused O(N) PER sampling to dominate wall time
    MIN_REPLAY           = 5_000
    MINIBATCH_SIZE       = 256
    STEP_BETWEEN_TRAIN   = 10
    N_STEP               = 8
    GAMMA                = 0.99

CAPACITY           = int(os.environ.get("CAPACITY", str(CAPACITY)))
MIN_REPLAY         = int(os.environ.get("MIN_REPLAY", str(MIN_REPLAY)))
MINIBATCH_SIZE     = int(os.environ.get("MINIBATCH_SIZE", str(MINIBATCH_SIZE)))
STEP_BETWEEN_TRAIN = int(os.environ.get("STEP_BETWEEN_TRAIN", str(STEP_BETWEEN_TRAIN)))
N_STEP             = int(os.environ.get("N_STEP", str(N_STEP)))
GAMMA              = float(os.environ.get("GAMMA", str(GAMMA)))

# Target network update frequency (Mnih 2015 rule: ~0.1% of total gradient updates).
# At paper scale (~220k updates) → C=1000.  At long (~2.2M updates) → C=3000.
if TRAINING_MODE == "smoke":
    UPDATE_TARGET_EVERY = 100
elif TRAINING_MODE == "mid":
    UPDATE_TARGET_EVERY = 500
elif TRAINING_MODE == "long":
    UPDATE_TARGET_EVERY = 3_000
else:
    UPDATE_TARGET_EVERY = 1_000
PER_ALPHA      = 0.6   # priority exponent
PER_BETA_START = 0.4
PER_BETA_END   = 1.0
# Anneal beta from 0.4 → 1.0 over the first ~half of expected gradient updates.
# Expected grad updates per mode (WHILE loop, 111/60/35 trans/slot ÷ STEP_BETWEEN_TRAIN):
#   long (~2.2M) → 1.1M    paper (~221k) → 110k    mid (~60k) → 30k    smoke (~14k) → 7k
if TRAINING_MODE == "smoke":
    PER_BETA_STEPS = 7_000
elif TRAINING_MODE == "mid":
    PER_BETA_STEPS = 30_000
elif TRAINING_MODE == "long":
    PER_BETA_STEPS = 1_100_000
else:   # paper
    PER_BETA_STEPS = 110_000

PPO_EPOCHS       = int(os.environ.get("PPO_EPOCHS", "4"))
PPO_CLIP_EPS     = float(os.environ.get("PPO_CLIP_EPS", "0.2"))
PPO_ENTROPY_COEF = float(os.environ.get("PPO_ENTROPY_COEF", "0.01"))
PPO_VALUE_COEF   = float(os.environ.get("PPO_VALUE_COEF", "0.5"))
PPO_VALUE_CLIP   = float(os.environ.get("PPO_VALUE_CLIP", "0.2"))
PPO_ADV_CLIP     = float(os.environ.get("PPO_ADV_CLIP", "5.0"))
PPO_TARGET_KL    = float(os.environ.get("PPO_TARGET_KL", "0.02"))
PPO_AUX_POLICY_COEF = float(os.environ.get("PPO_AUX_POLICY_COEF", "0.1"))
PPO_MATCH_RANK_COEF = float(os.environ.get("PPO_MATCH_RANK_COEF", "0.0"))
PPO_MATCH_RANK_MARGIN = float(os.environ.get("PPO_MATCH_RANK_MARGIN", "0.05"))
PPO_CF_VALUE_COEF = float(os.environ.get("PPO_CF_VALUE_COEF", "0.1"))
PPO_CF_ADV_COEF = float(os.environ.get("PPO_CF_ADV_COEF", "0.15"))
GAE_LAMBDA       = float(os.environ.get("GAE_LAMBDA", "0.95"))


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

    def _flush_nstep(self, bootstrap_state, bootstrap_done: bool,
                     bootstrap_cands=None) -> None:
        """Convert the n-step window into a single transition and store it."""
        if not self._nstep:
            return
        state0, action0 = self._nstep[0][0], self._nstep[0][1]
        ret      = 0.0
        gamma_n  = 1.0
        for entry in self._nstep:
            r, d = entry[2], entry[4]
            ret     += gamma_n * r
            gamma_n *= GAMMA
            if d:
                break   # episode ended within window
        done_final = bootstrap_done

        self._store(state0, action0, ret, bootstrap_state, done_final, gamma_n,
                    bootstrap_cands)

    def _store(self, state, action: int, ret: float,
               next_state, done: bool, gamma_n: float,
               next_cands=None) -> None:
        max_p = self._prios[:self._size].max() if self._size > 0 else 1.0
        self._buf[self._ptr]   = (state, action, ret, next_state, done, gamma_n,
                                  next_cands)
        self._prios[self._ptr] = max_p
        self._ptr  = (self._ptr + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def push(self, state, action: int, reward: float,
             next_state, done: bool, next_cands=None) -> None:
        self._nstep.append((state, action, reward, next_state, done, next_cands))
        if len(self._nstep) >= N_STEP or done:
            last = self._nstep[-1]
            self._flush_nstep(last[3], last[4], last[5])
            self._nstep.popleft()

    def flush_episode(self) -> None:
        """Call at episode end to drain remaining entries in the n-step window."""
        while self._nstep:
            last = self._nstep[-1]
            self._flush_nstep(last[3], last[4], last[5])
            self._nstep.popleft()

    def push_sequence(self, transitions: list) -> None:
        """
        Push one request's trajectory as an isolated n-step sequence.

        Clears the shared deque first so cross-request contamination is
        impossible — each call is a fresh episode.

        transitions: list of (state, action, reward, next_state, done[, next_cands])
        """
        self._nstep.clear()
        for t in transitions:
            s, a, r, ns, done = t[0], t[1], t[2], t[3], t[4]
            nc = t[5] if len(t) > 5 else None
            self._nstep.append((s, a, r, ns, done, nc))
            if len(self._nstep) >= N_STEP or done:
                last = self._nstep[-1]
                self._flush_nstep(last[3], last[4], last[5])
                self._nstep.popleft()
        self.flush_episode()

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
        # replace=True: O(1) vs O(N) for replace=False; duplicates are rare
        # when batch_size << buffer_size and don't materially hurt PER quality.
        idx   = np.random.choice(self._size, batch_size, replace=True, p=probs)

        weights = (self._size * probs[idx]) ** (-beta)
        weights /= weights.max()

        batch = [self._buf[i] for i in idx]
        s, a, r, ns, d, g, nc = zip(*batch)

        return (
            np.stack(s).astype(np.float32),
            np.array(a, dtype=np.int64),
            np.array(r, dtype=np.float32),
            np.stack(ns).astype(np.float32),
            np.array(d, dtype=np.float32),
            np.array(g, dtype=np.float32),
            weights.astype(np.float32),
            idx,
            list(nc),   # per-sample next-candidate matrices (None or ndarray)
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


class MAPPORolloutBuffer:
    """
    On-policy rollout storage for QuRA-Hive MAPPO.

    Each flattened sample stores the full variable candidate set for one
    request decision, the chosen candidate index, the old policy log-prob,
    centralized critic value, GAE return, and advantage.
    """

    def __init__(self, capacity: int = CAPACITY):
        self.capacity = capacity
        self._buf: list = []

    def push_trajectory(self, transitions: list) -> None:
        """
        transitions entries:
          {
            "candidate_states": np.ndarray(K, STATE_DIM),
            "action_idx": int,
            "old_logp": float,
            "value": float,
            "global_state": np.ndarray(G,),
            "node_features": np.ndarray(N, GNN_NODE_DIM),
            "edge_index": np.ndarray(2, E),
            "curr_id": int,
            "dst_id": int,
            "candidate_ids": np.ndarray(K,),
            "aux_policy_target": np.ndarray(K,),
            "counterfactual_target": float,
            "reward": float,
            "done": bool,
          }
        """
        if not transitions:
            return

        rewards = [float(t["reward"]) for t in transitions]
        values = [float(t["value"]) for t in transitions]
        dones = [bool(t["done"]) for t in transitions]

        adv = np.zeros(len(transitions), dtype=np.float32)
        last_gae = 0.0
        next_value = 0.0
        for i in range(len(transitions) - 1, -1, -1):
            nonterminal = 0.0 if dones[i] else 1.0
            delta = rewards[i] + GAMMA * next_value * nonterminal - values[i]
            last_gae = delta + GAMMA * GAE_LAMBDA * nonterminal * last_gae
            adv[i] = last_gae
            next_value = values[i]

        returns = adv + np.array(values, dtype=np.float32)

        for t, a, ret in zip(transitions, adv, returns):
            sample = {
                "candidate_states": t["candidate_states"].astype(np.float32),
                "action_idx": int(t["action_idx"]),
                "old_logp": float(t["old_logp"]),
                "old_value": float(t["value"]),
                "global_state": t["global_state"].astype(np.float32),
                "node_features": t["node_features"].astype(np.float32),
                "edge_index": t["edge_index"].astype(np.int64),
                "curr_id": int(t["curr_id"]),
                "dst_id": int(t["dst_id"]),
                "candidate_ids": t["candidate_ids"].astype(np.int64),
                "aux_policy_target": t["aux_policy_target"].astype(np.float32),
                "counterfactual_target": float(t.get("counterfactual_target", 0.0)),
                "return": float(ret),
                "advantage": float(a),
            }
            self._buf.append(sample)

        if len(self._buf) > self.capacity:
            self._buf = self._buf[-self.capacity:]

    def sample(self, batch_size: int) -> list:
        return random.sample(self._buf, min(batch_size, len(self._buf)))

    def samples(self) -> list:
        """Return a snapshot of all currently collected on-policy samples."""
        return list(self._buf)

    def clear(self) -> None:
        self._buf.clear()

    def __len__(self) -> int:
        return len(self._buf)
