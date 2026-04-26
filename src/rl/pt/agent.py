"""
PyTorch port of DQRLAgentDist.

Supports:
  - Single-agent DQN with experience replay  (Seq / Flock / Guard)
  - QMIX joint training step                  (Hive)
  - Double DQN target network

OOM fix: qmix_train_step reads MR from ps.shape[1] (the actual batch
max-request count returned by QMixEpisodeBuffer.sample) instead of the
global MAX_REQUESTS constant.  Combined with MINIBATCH_SIZE=32, peak
QMIX tensor memory is ~50-130 MB instead of 8+ GB.
"""
import copy
import gzip
import io
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .gat_flat import QRoutingGATFlat
from .qmixer  import QMixer
from .replay  import (
    ReplayBuffer, QMixEpisodeBuffer,
    MINIBATCH_SIZE, MIN_REPLAY_MEMORY_SIZE, UPDATE_TARGET_EVERY,
    MAX_REQUESTS_SMOKE, MAX_REQUESTS_PAPER,
    TRAINING_MODE,
)

LR          = 1e-4
CLIP_VALUE  = 0.1
GAMMA       = 0.9
REWARD_LAMBDA = 0.3
REWARD_MU     = 1.0
REWARD_NU     = 0.5

SIZE = 100
STATE_DIM        = 128 + SIZE + 2 * (SIZE ** 2)   # 20228
GLOBAL_STATE_DIM = SIZE * SIZE + SIZE              # 10100
EMBED_DIM        = 32   # QMixer embedding

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"
MODEL_SAVE_PATH = os.environ.get(
    "MODEL_SAVE_PATH", "/tmp/qrouting_model/trained_weights_pt.pkl")


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

        self.model = QRoutingGATFlat(
            num_nodes=num_nodes,
            input_flat_dim=self.state_dim,
            hidden_dim=hidden_dim,
        )
        self.target_model = copy.deepcopy(self.model)
        self.target_model.load_state_dict(self.model.state_dict())

        self.mixer        = QMixer(self.MAX_REQUESTS, self.global_state_dim, EMBED_DIM)
        self.target_mixer = copy.deepcopy(self.mixer)

        self.optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.mixer.parameters()),
            lr=LR,
        )

        self.single_replay = ReplayBuffer()
        self.qmix_replay   = QMixEpisodeBuffer()

        self.target_update_counter = 0
        self.reqState_qs = {}

    # ── Weight I/O ────────────────────────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        buf = io.BytesIO()
        torch.save(self.model.state_dict(), buf)
        with gzip.open(path, "wb") as f:
            f.write(buf.getvalue())

    def load_weights(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with gzip.open(path, "rb") as f:
                raw = f.read()
            state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
            self.model.load_state_dict(state)
            self.target_model.load_state_dict(state)
            return True
        except Exception as e:
            print(f"[agent] load_weights error: {e}")
            return False

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def act(self, state: np.ndarray, eps: float = 0.0) -> int:
        import random
        if random.random() < eps:
            return random.randrange(self.num_nodes)
        t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        q = self.model(t)
        return int(q.argmax(dim=1).item())

    @torch.no_grad()
    def batch_predict(self, states: np.ndarray) -> np.ndarray:
        t = torch.tensor(states, dtype=torch.float32)
        return self.model(t).numpy()

    # ── Single-agent DQN ──────────────────────────────────────────────────────

    def remember(self, state, action: int, reward: float,
                 next_state, done: bool) -> None:
        if not INFERENCE_MODE:
            self.single_replay.push(state, action, reward, next_state, done)

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
            online_next = self.model(nst)
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

    # ── QMIX joint training ───────────────────────────────────────────────────

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

        B  = ps.shape[0]
        MR = ps.shape[1]   # actual max requests in this batch (dynamic, not self.MAX_REQUESTS)

        ps_t   = torch.tensor(ps,      dtype=torch.float32)   # (B, MR, state_dim)
        pa_t   = torch.tensor(pa,      dtype=torch.long)       # (B, MR)
        mask_t = torch.tensor(mask,    dtype=torch.float32)    # (B, MR)
        pns_t  = torch.tensor(pns,     dtype=torch.float32)   # (B, MR, state_dim)
        gs_t   = torch.tensor(gs,      dtype=torch.float32)   # (B, g_dim)
        ngs_t  = torch.tensor(ngs,     dtype=torch.float32)   # (B, g_dim)
        r_t    = torch.tensor(rewards, dtype=torch.float32)   # (B, 1)
        d_t    = torch.tensor(dones,   dtype=torch.float32)   # (B, 1)

        flat_ps  = ps_t.view(B * MR, self.state_dim)
        flat_pns = pns_t.view(B * MR, self.state_dim)

        all_qs    = self.model(flat_ps).view(B, MR, self.num_nodes)
        chosen_qs = all_qs.gather(2, pa_t.unsqueeze(2)).squeeze(2)
        agent_qs  = chosen_qs * mask_t

        with torch.no_grad():
            online_next = self.model(flat_pns).view(B, MR, self.num_nodes)
            best_a_next = online_next.argmax(dim=2, keepdim=True)
            target_next = self.target_model(flat_pns).view(B, MR, self.num_nodes)
            target_qs   = target_next.gather(2, best_a_next).squeeze(2) * mask_t

            # QMixer expects agent_qs shape (B, MR) but mixer was built for MAX_REQUESTS.
            # Pad to self.MAX_REQUESTS if MR < MAX_REQUESTS so mixer weights match.
            if MR < self.MAX_REQUESTS:
                pad = self.MAX_REQUESTS - MR
                target_qs_pad = F.pad(target_qs, (0, pad))
            else:
                target_qs_pad = target_qs

            q_tot_target = self.target_mixer(target_qs_pad, ngs_t)
            y = r_t + GAMMA * q_tot_target * (1 - d_t)

        if MR < self.MAX_REQUESTS:
            agent_qs_pad = F.pad(agent_qs, (0, self.MAX_REQUESTS - MR))
        else:
            agent_qs_pad = agent_qs

        q_tot_online = self.mixer(agent_qs_pad, gs_t)
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
