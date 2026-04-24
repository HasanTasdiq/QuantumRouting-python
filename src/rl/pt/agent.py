"""
PyTorch port of DQRLAgentDist (originally src/rl/DQRLAgentDist_API.py).

Supports:
  - Single-agent DQN with experience replay  (Seq / Flock / Guard)
  - QMIX joint training step                  (Hive)
  - Double DQN target network
  - Redis model checkpoint I/O
  - INFERENCE_MODE env-var guard
"""
import copy
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_flat import QRoutingGATFlat
from .qmixer  import QMixer
from .replay  import (
    ReplayBuffer, QMixEpisodeBuffer,
    MINIBATCH_SIZE, MIN_REPLAY_MEMORY_SIZE, UPDATE_TARGET_EVERY,
    START_EPSILON_DECAYING, END_EPSILON_DECAYING,
    MAX_REQUESTS_SMOKE, MAX_REQUESTS_PAPER,
    TRAINING_MODE,
)
from .redis_io import (
    save_model_to_redis, load_model_from_redis,
    save_worker_model_to_redis, save_model_to_disk, load_model_from_disk,
    fedavg_aggregate,
)

# ── Hyper-parameters (mirror DQRLAgentDist_API.py) ──────────────────────────
LR          = 1e-4
CLIP_VALUE  = 0.1
GAMMA       = 0.9
REWARD_LAMBDA = 0.3
REWARD_MU     = 1.0
REWARD_NU     = 0.5
ENTANGLEMENT_LIFETIME = 10

SIZE = 100
STATE_DIM   = 128 + SIZE + 2 * (SIZE ** 2)  # 20228
GLOBAL_STATE_DIM = SIZE * SIZE + SIZE        # 10100
EMBED_DIM   = 32   # QMixer embed

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"
EPSILON_ = 1.0     # global epsilon — decayed in update_reward


class DQRLAgentDist:
    def __init__(self, pid: int = 0, model_name: str = "dqrl_model",
                 num_nodes: int = SIZE, hidden_dim: int = 64):
        self.pid        = pid
        self.model_name = model_name
        self.num_nodes  = num_nodes
        self.state_dim  = 128 + num_nodes + 2 * (num_nodes ** 2)
        self.global_state_dim = num_nodes * num_nodes + num_nodes

        self.MAX_REQUESTS = (MAX_REQUESTS_SMOKE
                             if TRAINING_MODE == "smoke"
                             else MAX_REQUESTS_PAPER)

        # Networks
        self.model = QRoutingGATFlat(
            num_nodes=num_nodes,
            input_flat_dim=self.state_dim,
            hidden_dim=hidden_dim,
        )
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())

        # QMIX mixer
        self.mixer        = QMixer(self.MAX_REQUESTS, self.global_state_dim, EMBED_DIM)
        self.target_mixer = copy.deepcopy(self.mixer)

        # Optimiser for joint agent + mixer update
        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.mixer.parameters()),
            lr=LR,
        )

        # Replay buffers
        self.single_replay = ReplayBuffer()
        self.qmix_replay   = QMixEpisodeBuffer()

        self.target_update_counter = 0
        self.reqState_qs = {}

    # ── Weight I/O ────────────────────────────────────────────────────────────
    def save_weights(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load_weights(self, path: str) -> None:
        self.model.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True))
        self.target_model.load_state_dict(self.model.state_dict())

    def save_to_redis(self, r=None) -> None:
        save_model_to_redis(self.model, self.model_name, r)

    def load_from_redis(self, r=None) -> bool:
        return load_model_from_redis(self.model, self.model_name, r) is not None

    def save_worker_to_redis(self, worker_id: int, r=None) -> None:
        save_worker_model_to_redis(self.model, worker_id,
                                   base_name=self.model_name, r=r)

    def save_to_disk(self) -> None:
        save_model_to_disk(self.model)

    def load_from_disk(self) -> bool:
        return load_model_from_disk(self.model)

    # ── Inference ─────────────────────────────────────────────────────────────
    @torch.no_grad()
    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        if random.random() < eps:
            return random.randrange(self.num_nodes)
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q = self.model(t)          # (1, num_nodes)
        return int(q.argmax(dim=1).item())

    @torch.no_grad()
    def batch_predict(self, states: np.ndarray) -> np.ndarray:
        """Batched forward pass; returns (N, num_nodes) Q-values."""
        t = torch.tensor(states, dtype=torch.float32)
        return self.model(t).numpy()

    # ── Single-agent replay push ───────────────────────────────────────────────
    def remember(self, state, action: int, reward: float,
                 next_state, done: bool) -> None:
        if not INFERENCE_MODE:
            self.single_replay.push(state, action, reward, next_state, done)

    # ── Single-agent DQN update ───────────────────────────────────────────────
    def replay(self, batch_size: int = MINIBATCH_SIZE) -> float | None:
        if INFERENCE_MODE:
            return None
        if len(self.single_replay) < MIN_REPLAY_MEMORY_SIZE:
            return None

        states, actions, rewards, next_states, dones = \
            self.single_replay.sample(batch_size)

        st  = torch.tensor(states,      dtype=torch.float32)
        a   = torch.tensor(actions,     dtype=torch.long)
        r   = torch.tensor(rewards,     dtype=torch.float32)
        nst = torch.tensor(next_states, dtype=torch.float32)
        d   = torch.tensor(dones,       dtype=torch.float32)

        with torch.no_grad():
            # Double DQN: select with online, evaluate with target
            online_next = self.model(nst)                  # (B, A)
            best_a = online_next.argmax(dim=1, keepdim=True)
            target_next = self.target_model(nst)
            q_target = r + GAMMA * target_next.gather(1, best_a).squeeze(1) * (1 - d)

        q_pred = self.model(st).gather(1, a.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), CLIP_VALUE)
        self.optimizer.step()

        self._maybe_update_target()
        return loss.item()

    # ── QMIX joint train step (Hive) ─────────────────────────────────────────
    def push_qmix_transition(self, ts_states, ts_actions, ts_rewards,
                              ts_next_states, global_state,
                              next_global_state, done: bool) -> None:
        if not INFERENCE_MODE:
            self.qmix_replay.push(
                ts_states, ts_actions, ts_rewards, ts_next_states,
                global_state, next_global_state, done,
            )

    def qmix_train_step(self) -> float | None:
        if INFERENCE_MODE:
            return None
        if len(self.qmix_replay) < MIN_REPLAY_MEMORY_SIZE:
            return None

        (ps, pa, mask, pns, gs, ngs, rewards, dones) = \
            self.qmix_replay.sample(
                MINIBATCH_SIZE, self.MAX_REQUESTS, self.state_dim)

        B  = MINIBATCH_SIZE
        MR = self.MAX_REQUESTS

        ps_t   = torch.tensor(ps,      dtype=torch.float32)
        pa_t   = torch.tensor(pa,      dtype=torch.long)
        mask_t = torch.tensor(mask,    dtype=torch.float32)
        pns_t  = torch.tensor(pns,     dtype=torch.float32)
        gs_t   = torch.tensor(gs,      dtype=torch.float32)
        ngs_t  = torch.tensor(ngs,     dtype=torch.float32)
        r_t    = torch.tensor(rewards, dtype=torch.float32)
        d_t    = torch.tensor(dones,   dtype=torch.float32)

        # Flatten for batched model forward pass
        flat_ps  = ps_t.view(B * MR, self.state_dim)
        flat_pns = pns_t.view(B * MR, self.state_dim)

        # Online forward
        all_qs = self.model(flat_ps).view(B, MR, self.num_nodes)  # (B, MR, A)
        chosen_qs = all_qs.gather(2, pa_t.unsqueeze(2)).squeeze(2)  # (B, MR)
        agent_qs = chosen_qs * mask_t

        # Double DQN for target
        with torch.no_grad():
            online_next = self.model(flat_pns).view(B, MR, self.num_nodes)
            best_a_next = online_next.argmax(dim=2, keepdim=True)
            target_next = self.target_model(flat_pns).view(B, MR, self.num_nodes)
            target_qs   = target_next.gather(2, best_a_next).squeeze(2) * mask_t

            q_tot_target = self.target_mixer(target_qs, ngs_t)
            y = r_t + GAMMA * q_tot_target * (1 - d_t)

        q_tot_online = self.mixer(agent_qs, gs_t)
        loss = F.smooth_l1_loss(q_tot_online, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.model.parameters()) + list(self.mixer.parameters()),
            CLIP_VALUE,
        )
        self.optimizer.step()

        self._maybe_update_target()
        return loss.item()

    def _maybe_update_target(self) -> None:
        self.target_update_counter += 1
        if self.target_update_counter >= UPDATE_TARGET_EVERY:
            self.target_model.load_state_dict(self.model.state_dict())
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.target_update_counter = 0
