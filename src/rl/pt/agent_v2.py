"""
DQRLAgentV2 — integration layer for QuRA-v2.

Wraps EdgeQNet + optional QMixerV2.  QuantumGAT is NOT used here — node
embeddings are replaced by hand-crafted features in local_trainer_v2._edge_state
so that (a) there are no frozen/random encoder weights, and (b) every gradient
update directly improves the routing policy.

Gradient-flow assertion: run one dummy train_step; assert EdgeQNet weights change.
"""
import copy
import io
import os
import gzip

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .qnet_v2 import EdgeQNet, QMixerV2, STATE_DIM, QMIX_GLOBAL_DIM, QMIX_REQ_DIM
from .replay_v2 import (
    NStepPERBuffer, QMixEpisodicBuffer,
    MINIBATCH_SIZE, MIN_REPLAY, UPDATE_TARGET_EVERY,
    GAMMA, PER_BETA_START, PER_BETA_END, PER_BETA_STEPS,
    STEP_BETWEEN_TRAIN,
)

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"
LR             = 5e-4
LR_MIN         = 1e-5
CLIP_NORM      = 1.0
GLOBAL_STATE_DIM = QMIX_GLOBAL_DIM   # kept for any external import

# LR anneals from LR → LR_MIN over the full expected training run.
# Matches WHILE-loop training frequency (~11/7/8 trains per slot).
# Can be overridden via LR_ANNEAL_STEPS env var.
from .replay_v2 import TRAINING_MODE as _TMODE
if os.environ.get("LR_ANNEAL_STEPS"):
    LR_ANNEAL_STEPS = int(os.environ["LR_ANNEAL_STEPS"])
elif _TMODE == "smoke":
    LR_ANNEAL_STEPS = 14_000   # 2000 slots × ~7 trains/slot
elif _TMODE == "mid":
    LR_ANNEAL_STEPS = 60_000   # 8000 slots × ~7.5 trains/slot
else:   # paper
    LR_ANNEAL_STEPS = 220_000  # 20000 slots × ~11 trains/slot


class DQRLAgentV2:
    """
    Single shared agent for all four QuRA-v2 variants.

    The GAT encoder has been removed.  Edge-state vectors are built from
    hand-crafted features (degree, request density, BFS distance) by the
    trainer, so no frozen embeddings contaminate Q-learning.

    Parameters
    ----------
    pid       : int  — process id (debugging only)
    num_nodes : int  — N (unused internally; kept for API compat)
    use_qmix  : bool — whether to train a QMIX mixer (Hive variant)
    """

    def __init__(self, pid: int = 0, num_nodes: int = 100,
                 use_qmix: bool = False):
        self.pid       = pid
        self.num_nodes = num_nodes
        self.use_qmix  = use_qmix

        self.qnet        = EdgeQNet(state_dim=STATE_DIM)
        self.target_qnet = copy.deepcopy(self.qnet)

        self.mixer        = QMixerV2() if use_qmix else None
        self.target_mixer = copy.deepcopy(self.mixer) if use_qmix else None

        params = list(self.qnet.parameters())
        if self.mixer:
            params += list(self.mixer.parameters())

        self.optimizer    = torch.optim.Adam(params, lr=LR)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=LR_ANNEAL_STEPS, eta_min=LR_MIN)

        self.single_replay = NStepPERBuffer()
        self.qmix_replay   = QMixEpisodicBuffer()

        self._target_ctr   = 0
        self._train_ctr    = 0
        self._beta         = PER_BETA_START
        self._beta_step    = (PER_BETA_END - PER_BETA_START) / max(PER_BETA_STEPS, 1)

    # ── Target network sync ───────────────────────────────────────────────────

    def _maybe_update_target(self) -> None:
        self._target_ctr += 1
        if self._target_ctr >= UPDATE_TARGET_EVERY:
            self.target_qnet.load_state_dict(self.qnet.state_dict())
            if self.mixer:
                self.target_mixer.load_state_dict(self.mixer.state_dict())
            self._target_ctr = 0

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_neighbors_v(self, state_vecs: np.ndarray) -> np.ndarray:
        """
        Score pre-built edge-state vectors.

        state_vecs: (K, STATE_DIM) float32 ndarray
        Returns   : (K,) Q-value scores
        """
        self.qnet.eval()
        t = torch.tensor(state_vecs, dtype=torch.float32)
        return self.qnet.net(t).squeeze(1).numpy()

    def remember(self, state_v: np.ndarray, neighbor_idx: int,
                 reward: float, next_state_v: np.ndarray,
                 done: bool) -> None:
        """Legacy single-push interface (used by assert_gradients_flow)."""
        if not INFERENCE_MODE:
            self.single_replay.push(state_v, neighbor_idx, reward, next_state_v, done)

    # ── DQN training ──────────────────────────────────────────────────────────

    def train_dqn(self) -> float | None:
        """
        One Double-DQN training step over a PER minibatch.

        Stored state vectors already contain all features; no graph
        re-encoding needed.  Returns loss or None if buffer not ready.
        """
        if INFERENCE_MODE or len(self.single_replay) < MIN_REPLAY:
            return None

        self._beta = min(PER_BETA_END, self._beta + self._beta_step)
        s, a, r, ns, d, g, w, idx = self.single_replay.sample(
            MINIBATCH_SIZE, self._beta)

        st  = torch.tensor(s,  dtype=torch.float32)   # (B, STATE_DIM)
        r_t = torch.tensor(r,  dtype=torch.float32)   # (B,)
        nst = torch.tensor(ns, dtype=torch.float32)   # (B, STATE_DIM)
        d_t = torch.tensor(d,  dtype=torch.float32)   # (B,)
        g_t = torch.tensor(g,  dtype=torch.float32)   # (B,)  γ^n
        w_t = torch.tensor(w,  dtype=torch.float32)   # (B,)

        self.qnet.train()
        q_pred = self.qnet.net(st).squeeze(1)          # (B,)

        with torch.no_grad():
            # next_state already stores the argmax-Q candidate edge vector
            # (set by local_trainer_v2 at storage time), so bootstrapping
            # against target_qnet(next_state) is the correct Bellman target.
            q_ns_target = self.target_qnet.net(nst).squeeze(1)   # (B,)
            q_target    = r_t + g_t * q_ns_target * (1.0 - d_t)

        td_err = (q_pred - q_target).detach()
        loss   = (w_t * F.smooth_l1_loss(q_pred, q_target, reduction='none')).mean()

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(list(self.qnet.parameters()), CLIP_NORM)
        self.optimizer.step()
        self.lr_scheduler.step()

        self.qnet.eval()
        self.single_replay.update_priorities(idx, td_err.numpy())
        self._maybe_update_target()
        return loss.item()

    # ── QMIX training (Hive) ─────────────────────────────────────────────────

    def train_qmix(self) -> float | None:
        if not self.use_qmix or INFERENCE_MODE:
            return None
        if len(self.qmix_replay) < MIN_REPLAY:
            return None

        ps, pa, mask, pns, rf, gs, ngs, rew, don = self.qmix_replay.sample(
            min(MINIBATCH_SIZE, len(self.qmix_replay)))

        B, R, SD = ps.shape

        ps_t   = torch.tensor(ps,   dtype=torch.float32)
        pa_t   = torch.tensor(pa,   dtype=torch.long)
        mask_t = torch.tensor(mask, dtype=torch.float32)
        pns_t  = torch.tensor(pns,  dtype=torch.float32)
        rf_t   = torch.tensor(rf,   dtype=torch.float32)
        gs_t   = torch.tensor(gs,   dtype=torch.float32)
        ngs_t  = torch.tensor(ngs,  dtype=torch.float32)
        r_t    = torch.tensor(rew,  dtype=torch.float32)
        d_t    = torch.tensor(don,  dtype=torch.float32)

        self.qnet.train()
        self.mixer.train()

        flat_ps  = ps_t.view(B * R, SD)
        flat_pns = pns_t.view(B * R, SD)

        q_all     = self.qnet.net(flat_ps).view(B, R)
        chosen_qs = q_all * mask_t

        with torch.no_grad():
            q_ns_target  = self.target_qnet.net(flat_pns).view(B, R)
            target_qs    = q_ns_target * mask_t
            q_tot_target = self.target_mixer(target_qs, mask_t, rf_t, ngs_t)
            y = r_t + GAMMA * q_tot_target * (1.0 - d_t)

        q_tot = self.mixer(chosen_qs, mask_t, rf_t, gs_t)
        loss  = F.smooth_l1_loss(q_tot, y)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.qnet.parameters()) + list(self.mixer.parameters()),
            CLIP_NORM)
        self.optimizer.step()
        self.lr_scheduler.step()

        self.qnet.eval()
        self.mixer.eval()
        self._maybe_update_target()
        return loss.item()

    # ── Weight I/O ────────────────────────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        buf  = io.BytesIO()
        ckpt = {"qnet": self.qnet.state_dict()}
        if self.mixer:
            ckpt["mixer"] = self.mixer.state_dict()
        torch.save(ckpt, buf)
        with gzip.open(path, "wb") as f:
            f.write(buf.getvalue())

    def load_weights(self, path: str) -> bool:
        if not os.path.exists(path):
            return False
        try:
            with gzip.open(path, "rb") as f:
                raw = f.read()
            ckpt = torch.load(io.BytesIO(raw), map_location="cpu",
                              weights_only=True)
            self.qnet.load_state_dict(ckpt["qnet"])
            self.target_qnet.load_state_dict(ckpt["qnet"])
            if self.mixer and "mixer" in ckpt:
                self.mixer.load_state_dict(ckpt["mixer"])
                self.target_mixer.load_state_dict(ckpt["mixer"])
            return True
        except Exception as e:
            print(f"[agent_v2] load error: {e}")
            return False

    # ── Gradient flow self-test ───────────────────────────────────────────────

    def assert_gradients_flow(self) -> None:
        """
        Assert EdgeQNet (and mixer if Hive) receive gradients during a train step.
        Catches accidental torch.no_grad wrapping or disconnected modules.
        """
        rng = np.random.default_rng(42)

        w0_qnet = next(self.qnet.parameters()).data.clone()

        # Push enough transitions to cross MIN_REPLAY
        N_PUSH = MIN_REPLAY + MINIBATCH_SIZE + 10
        sv = rng.standard_normal(STATE_DIM).astype(np.float32)
        for _ in range(N_PUSH):
            # push_sequence: each call is a clean 1-step terminal episode
            self.single_replay.push_sequence(
                [(sv, 0, 1.0, sv * 0.9, True)])

        loss = self.train_dqn()
        assert loss is not None, "train_dqn returned None — buffer not ready?"

        w1_qnet = next(self.qnet.parameters()).data.clone()
        assert not torch.equal(w0_qnet, w1_qnet), \
            "EdgeQNet weights did not change — gradient flow broken!"

        if self.use_qmix and self.mixer:
            w0_mix = next(self.mixer.parameters()).data.clone()
            # Push a minimal QMIX episode
            ep_s  = [sv]
            ep_a  = [0]
            ep_r  = [1.0]
            ep_ns = [sv * 0.9]
            ep_rf = [rng.standard_normal(QMIX_REQ_DIM).astype(np.float32)]
            gs    = rng.standard_normal(QMIX_GLOBAL_DIM).astype(np.float32)
            for _ in range(MIN_REPLAY + 10):
                self.qmix_replay.push(ep_s, ep_a, ep_r, ep_ns, ep_rf, gs, gs, done=True)
            mix_loss = self.train_qmix()
            assert mix_loss is not None, "train_qmix returned None"
            w1_mix = next(self.mixer.parameters()).data.clone()
            assert not torch.equal(w0_mix, w1_mix), \
                "QMixerV2 weights did not change — gradient flow broken!"

        print("[agent_v2] gradient flow assertion PASS")
