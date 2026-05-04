"""
DQRLAgentV2 — integration layer for QuRA-v2.

Wraps EdgeQNet + optional MAPPOActorCritic.  QuantumGAT is NOT used here — node
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

from .qnet_v2 import (
    EdgeQNet, MAPPOActorCritic, STATE_DIM,
    MAPPO_GLOBAL_DIM, GNN_NODE_DIM, QMIX_GLOBAL_DIM, QMIX_REQ_DIM,
)
from .replay_v2 import (
    NStepPERBuffer, MAPPORolloutBuffer,
    MINIBATCH_SIZE, MIN_REPLAY, UPDATE_TARGET_EVERY,
    GAMMA, PER_BETA_START, PER_BETA_END, PER_BETA_STEPS,
    PPO_EPOCHS, PPO_CLIP_EPS, PPO_ENTROPY_COEF, PPO_VALUE_COEF,
    PPO_VALUE_CLIP, PPO_ADV_CLIP, PPO_TARGET_KL, PPO_AUX_POLICY_COEF,
    PPO_MATCH_RANK_COEF, PPO_MATCH_RANK_MARGIN,
    PPO_CF_VALUE_COEF, PPO_CF_ADV_COEF,
)

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"
LR             = float(os.environ.get("LR", "5e-4"))
MAPPO_LR       = float(os.environ.get("MAPPO_LR", "1e-4"))
LR_MIN         = float(os.environ.get("LR_MIN", "1e-6"))
CLIP_NORM      = float(os.environ.get("CLIP_NORM", "0.5"))
GLOBAL_STATE_DIM = MAPPO_GLOBAL_DIM   # kept for any external import
PPO_MAX_SAMPLES = int(os.environ.get("PPO_MAX_SAMPLES", "0"))

# LR anneals from LR → LR_MIN over the full expected training run.
# Matches WHILE-loop training frequency (~11/7/8 trains per slot).
# Can be overridden via LR_ANNEAL_STEPS env var.
from .replay_v2 import TRAINING_MODE as _TMODE
if os.environ.get("LR_ANNEAL_STEPS"):
    LR_ANNEAL_STEPS = int(os.environ["LR_ANNEAL_STEPS"])
elif _TMODE == "smoke":
    LR_ANNEAL_STEPS = 14_000      # 2000 slots × ~7 trains/slot
elif _TMODE == "mid":
    LR_ANNEAL_STEPS = 60_000      # 8000 slots × ~7.5 trains/slot
elif _TMODE == "long":
    LR_ANNEAL_STEPS = 2_200_000   # 1M slots × ~111 trans/slot ÷ 50 STEP_BETWEEN_TRAIN
else:   # paper
    LR_ANNEAL_STEPS = 220_000     # 20000 slots × ~11 trains/slot


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
    use_qmix  : bool — legacy name; True enables MAPPO for Hive
    """

    def __init__(self, pid: int = 0, num_nodes: int = 100,
                 use_qmix: bool = False,
                 dqn_arch: str | None = None):
        self.pid       = pid
        self.num_nodes = num_nodes
        self.use_qmix  = use_qmix
        self.use_mappo = use_qmix
        self.dqn_arch  = dqn_arch

        self.qnet        = None if self.use_mappo else EdgeQNet(
            state_dim=STATE_DIM, arch=dqn_arch)
        self.target_qnet = copy.deepcopy(self.qnet) if self.qnet else None
        self.actor_critic = MAPPOActorCritic() if self.use_mappo else None

        params = (list(self.actor_critic.parameters())
                  if self.use_mappo else list(self.qnet.parameters()))

        init_lr = MAPPO_LR if self.use_mappo else LR
        self.optimizer    = torch.optim.Adam(params, lr=init_lr)
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=LR_ANNEAL_STEPS, eta_min=LR_MIN)

        self.single_replay = NStepPERBuffer()
        self.mappo_replay  = MAPPORolloutBuffer()

        self._target_ctr   = 0
        self._train_ctr    = 0
        self._beta         = PER_BETA_START
        self._beta_step    = (PER_BETA_END - PER_BETA_START) / max(PER_BETA_STEPS, 1)

    # ── Target network sync ───────────────────────────────────────────────────

    def _maybe_update_target(self) -> None:
        self._target_ctr += 1
        if self.qnet and self._target_ctr >= UPDATE_TARGET_EVERY:
            self.target_qnet.load_state_dict(self.qnet.state_dict())
            self._target_ctr = 0

    # ── Inference ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def score_neighbors_v(self, state_vecs: np.ndarray) -> np.ndarray:
        """
        Score pre-built edge-state vectors.

        state_vecs: (K, STATE_DIM) float32 ndarray
        Returns   : (K,) Q-value scores
        """
        if self.use_mappo:
            self.actor_critic.eval()
            t = torch.tensor(state_vecs, dtype=torch.float32)
            return self.actor_critic.logits(t).numpy()
        if getattr(self.qnet, "arch", "") == "linear":
            w = self.qnet.net.weight.detach().cpu().numpy().reshape(-1)
            b = float(self.qnet.net.bias.detach().cpu().numpy()[0])
            return state_vecs.astype(np.float32, copy=False) @ w + b
        self.qnet.eval()
        t = torch.tensor(state_vecs, dtype=torch.float32)
        return self.qnet.net(t).squeeze(1).numpy()

    @torch.no_grad()
    def encode_mappo_graph(self, node_features: np.ndarray,
                           edge_index: np.ndarray) -> torch.Tensor:
        """Encode one active topology snapshot for reuse across a routing hop."""
        if not self.use_mappo:
            raise RuntimeError("encode_mappo_graph called on non-MAPPO agent")
        self.actor_critic.eval()
        nf = torch.tensor(node_features, dtype=torch.float32)
        ei = torch.tensor(edge_index, dtype=torch.long)
        return self.actor_critic.encode_graph(nf, ei)

    @torch.no_grad()
    def select_mappo_action(self, candidate_states: np.ndarray,
                            global_state: np.ndarray,
                            node_features: np.ndarray,
                            edge_index: np.ndarray,
                            curr_id: int,
                            dst_id: int,
                            candidate_ids: np.ndarray,
                            node_embeddings: torch.Tensor | None = None,
                            forced_action: int | None = None,
                            deterministic: bool = False) -> tuple[int, float, float]:
        """
        Select a candidate index for Hive and return (idx, old_logp, value).
        """
        if not self.use_mappo:
            raise RuntimeError("select_mappo_action called on non-MAPPO agent")

        self.actor_critic.eval()
        cs = torch.tensor(candidate_states, dtype=torch.float32)
        gs = torch.tensor(global_state, dtype=torch.float32)
        nf = torch.tensor(node_features, dtype=torch.float32)
        ei = torch.tensor(edge_index, dtype=torch.long)
        cids = torch.tensor(candidate_ids, dtype=torch.long)
        if node_embeddings is None:
            node_embeddings = self.actor_critic.encode_graph(nf, ei)
        logits = self.actor_critic.logits_from_embeddings(
            cs, node_embeddings, curr_id, dst_id, cids)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
        dist = torch.distributions.Categorical(logits=logits)
        if forced_action is not None:
            action = int(forced_action)
        elif deterministic:
            action = int(torch.argmax(logits).item())
        else:
            action = int(dist.sample().item())
        logp = float(dist.log_prob(torch.tensor(action)).item())
        value = float(self.actor_critic.value_from_embeddings(
            cs, gs, node_embeddings, curr_id, dst_id, cids).item())
        chosen_logit = float(logits[action].item())
        return action, logp, value, chosen_logit

    @torch.no_grad()
    def score_mappo_candidates(self, candidate_states: np.ndarray,
                               global_state: np.ndarray,
                               node_features: np.ndarray,
                               edge_index: np.ndarray,
                               curr_id: int,
                               dst_id: int,
                               candidate_ids: np.ndarray,
                               node_embeddings: torch.Tensor | None = None
                               ) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Score all next-hop candidates without sampling an action.

        Returns (logits, log_probs, value) for one request's candidate set.
        """
        if not self.use_mappo:
            raise RuntimeError("score_mappo_candidates called on non-MAPPO agent")

        self.actor_critic.eval()
        cs = torch.tensor(candidate_states, dtype=torch.float32)
        gs = torch.tensor(global_state, dtype=torch.float32)
        nf = torch.tensor(node_features, dtype=torch.float32)
        ei = torch.tensor(edge_index, dtype=torch.long)
        cids = torch.tensor(candidate_ids, dtype=torch.long)
        if node_embeddings is None:
            node_embeddings = self.actor_critic.encode_graph(nf, ei)
        logits = self.actor_critic.logits_from_embeddings(
            cs, node_embeddings, curr_id, dst_id, cids)
        logps = torch.log_softmax(logits, dim=0)
        value = self.actor_critic.value_from_embeddings(
            cs, gs, node_embeddings, curr_id, dst_id, cids)
        return (
            logits.detach().cpu().numpy(),
            logps.detach().cpu().numpy(),
            float(value.item()),
        )

    def remember(self, state_v: np.ndarray, neighbor_idx: int,
                 reward: float, next_state_v: np.ndarray,
                 done: bool, next_cands: np.ndarray | None = None) -> None:
        """Legacy single-push interface (used by assert_gradients_flow)."""
        if not INFERENCE_MODE:
            self.single_replay.push(state_v, neighbor_idx, reward, next_state_v,
                                    done, next_cands)

    # ── DQN training ──────────────────────────────────────────────────────────

    def train_dqn(self) -> float | None:
        """
        One Double-DQN training step over a PER minibatch.

        Stored state vectors already contain all features; no graph
        re-encoding needed.  Returns loss or None if buffer not ready.
        """
        if self.use_mappo or INFERENCE_MODE or len(self.single_replay) < MIN_REPLAY:
            return None

        self._beta = min(PER_BETA_END, self._beta + self._beta_step)
        s, a, r, ns, d, g, w, idx, next_cands_list = self.single_replay.sample(
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
            # Double DQN — batched implementation to avoid per-sample Python loops.
            #
            # Partition the batch into two groups:
            #   simple : done=True OR nc is None OR nc.shape[0]==1
            #            → evaluate target_qnet on stored next-state vector (nst)
            #   multi  : nc is an (K, STATE_DIM) matrix with K>1
            #            → stack all K-candidate rows, run one online forward to
            #              find argmax, then one target forward to get Q-value.
            #
            # Both groups use a single batched forward pass each.
            q_ns_target = torch.zeros(len(s), dtype=torch.float32)

            simple_idx, multi_idx, multi_nc = [], [], []
            for i, nc in enumerate(next_cands_list):
                if d[i] > 0.5:
                    pass   # terminal — q_ns_target[i] stays 0
                elif nc is None or nc.shape[0] <= 1:
                    simple_idx.append(i)
                else:
                    multi_idx.append(i)
                    multi_nc.append(nc)

            # Simple group: one batched target forward
            if simple_idx:
                si = torch.tensor(simple_idx, dtype=torch.long)
                q_ns_target[si] = self.target_qnet.net(nst[si]).squeeze(1)

            # Multi-candidate group: pad to max_K, online argmax, target eval
            if multi_idx:
                max_k = max(nc.shape[0] for nc in multi_nc)
                sdim  = multi_nc[0].shape[1]
                B_m   = len(multi_nc)
                padded = np.zeros((B_m, max_k, sdim), dtype=np.float32)
                lengths = np.zeros(B_m, dtype=np.int64)
                for j, nc in enumerate(multi_nc):
                    k = nc.shape[0]
                    padded[j, :k] = nc
                    lengths[j]    = k

                pad_t = torch.tensor(padded, dtype=torch.float32)     # (B_m, max_k, sdim)
                flat  = pad_t.view(B_m * max_k, sdim)
                # Mask padded positions with -inf before argmax
                online_q = self.qnet.net(flat).squeeze(1).view(B_m, max_k)
                for j in range(B_m):
                    online_q[j, lengths[j]:] = float('-inf')
                best_k = online_q.argmax(dim=1)                        # (B_m,)
                # Gather best-candidate rows for target evaluation
                best_rows = pad_t[torch.arange(B_m), best_k]          # (B_m, sdim)
                q_best = self.target_qnet.net(best_rows).squeeze(1)   # (B_m,)
                mi = torch.tensor(multi_idx, dtype=torch.long)
                q_ns_target[mi] = q_best

            q_target = r_t + g_t * q_ns_target * (1.0 - d_t)

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

    # ── MAPPO training (Hive) ────────────────────────────────────────────────

    def train_mappo(self) -> float | None:
        if not self.use_mappo or INFERENCE_MODE:
            return None
        if len(self.mappo_replay) < MIN_REPLAY:
            return None

        rollout = self.mappo_replay.samples()
        if not rollout:
            return None
        # Drop samples with degenerate log-probs (can cause ratio explosion)
        rollout = [b for b in rollout if float(b["old_logp"]) > -20.0]
        if not rollout:
            return None
        if PPO_MAX_SAMPLES > 0 and len(rollout) > PPO_MAX_SAMPLES:
            idx = np.random.choice(len(rollout), PPO_MAX_SAMPLES, replace=False)
            rollout = [rollout[int(i)] for i in idx]

        adv_np = np.array([
            b["advantage"] + PPO_CF_ADV_COEF * b.get("counterfactual_target", 0.0)
            for b in rollout
        ], dtype=np.float32)
        adv_mean = float(adv_np.mean())
        adv_std = float(adv_np.std() + 1e-8)

        losses = []
        for _ in range(PPO_EPOCHS):
            order = np.random.permutation(len(rollout))
            stop_early = False
            for start in range(0, len(order), MINIBATCH_SIZE):
                batch = [rollout[i] for i in order[start:start + MINIBATCH_SIZE]]
                if not batch:
                    continue

                policy_losses, value_losses, entropies, approx_kls = [], [], [], []
                aux_policy_losses, match_rank_losses, cf_value_losses = [], [], []
                self.actor_critic.train()
                graph_cache: dict[tuple, torch.Tensor] = {}

                for b in batch:
                    cs = torch.tensor(b["candidate_states"], dtype=torch.float32)
                    gs = torch.tensor(b["global_state"], dtype=torch.float32)
                    cids = torch.tensor(b["candidate_ids"], dtype=torch.long)
                    curr_id = int(b["curr_id"])
                    dst_id = int(b["dst_id"])
                    action = torch.tensor(int(b["action_idx"]), dtype=torch.long)
                    old_logp = torch.tensor(float(b["old_logp"]), dtype=torch.float32)
                    old_value = torch.tensor(float(b["old_value"]), dtype=torch.float32)
                    ret = torch.tensor(float(b["return"]), dtype=torch.float32)
                    aux_policy_target = torch.tensor(
                        b["aux_policy_target"], dtype=torch.float32
                    )
                    cf_target = torch.tensor(
                        float(b.get("counterfactual_target", 0.0)),
                        dtype=torch.float32,
                    )
                    raw_advantage = (
                        float(b["advantage"])
                        + PPO_CF_ADV_COEF * float(b.get("counterfactual_target", 0.0))
                    )
                    adv = torch.tensor(
                        (raw_advantage - adv_mean) / adv_std,
                        dtype=torch.float32,
                    )
                    adv = torch.clamp(adv, -PPO_ADV_CLIP, PPO_ADV_CLIP)

                    nf_np = b["node_features"]
                    ei_np = b["edge_index"]
                    graph_key = (
                        nf_np.shape, nf_np.tobytes(),
                        ei_np.shape, ei_np.tobytes(),
                    )
                    node_embeddings = graph_cache.get(graph_key)
                    if node_embeddings is None:
                        nf = torch.tensor(nf_np, dtype=torch.float32)
                        ei = torch.tensor(ei_np, dtype=torch.long)
                        node_embeddings = self.actor_critic.encode_graph(nf, ei)
                        graph_cache[graph_key] = node_embeddings

                    logits = self.actor_critic.logits_from_embeddings(
                        cs, node_embeddings, curr_id, dst_id, cids)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=10.0, neginf=-10.0)
                    dist = torch.distributions.Categorical(logits=logits)
                    logp = dist.log_prob(action)
                    entropy = dist.entropy()
                    value = self.actor_critic.value_from_embeddings(
                        cs, gs, node_embeddings, curr_id, dst_id, cids)
                    cf_pred = self.actor_critic.counterfactual_from_embeddings(
                        cs, gs, node_embeddings, curr_id, dst_id, cids,
                        int(action.item()))

                    log_ratio = torch.clamp(logp - old_logp, -10.0, 10.0)
                    ratio = torch.exp(log_ratio)
                    unclipped = ratio * adv
                    clipped = torch.clamp(ratio, 1.0 - PPO_CLIP_EPS,
                                          1.0 + PPO_CLIP_EPS) * adv
                    policy_losses.append(-torch.min(unclipped, clipped))
                    value_clipped = old_value + torch.clamp(
                        value - old_value, -PPO_VALUE_CLIP, PPO_VALUE_CLIP)
                    value_losses.append(torch.max(
                        F.smooth_l1_loss(value, ret),
                        F.smooth_l1_loss(value_clipped, ret),
                    ))
                    if aux_policy_target.numel() == logits.numel():
                        aux_policy_losses.append(
                            F.kl_div(
                                torch.log_softmax(logits, dim=0),
                                aux_policy_target,
                                reduction="batchmean",
                            )
                        )
                    cf_value_losses.append(F.smooth_l1_loss(cf_pred, cf_target))
                    if raw_advantage > 0.0 and logits.numel() > 1:
                        chosen_logit = logits[action]
                        other_mask = torch.ones(logits.size(0), dtype=torch.bool)
                        other_mask[action] = False
                        other_logits = logits[other_mask]
                        if other_logits.numel() > 0:
                            match_rank_losses.append(
                                F.relu(PPO_MATCH_RANK_MARGIN - (chosen_logit - other_logits)).mean()
                            )
                    entropies.append(entropy)
                    approx_kls.append(old_logp - logp)

                loss = (
                    torch.stack(policy_losses).mean()
                    + PPO_VALUE_COEF * torch.stack(value_losses).mean()
                    - PPO_ENTROPY_COEF * torch.stack(entropies).mean()
                )
                if aux_policy_losses:
                    loss = loss + PPO_AUX_POLICY_COEF * torch.stack(aux_policy_losses).mean()
                if cf_value_losses:
                    loss = loss + PPO_CF_VALUE_COEF * torch.stack(cf_value_losses).mean()
                if match_rank_losses:
                    loss = loss + PPO_MATCH_RANK_COEF * torch.stack(match_rank_losses).mean()

                if not torch.isfinite(loss):
                    self.optimizer.zero_grad()
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(self.actor_critic.parameters()), CLIP_NORM)
                self.optimizer.step()
                self.lr_scheduler.step()
                losses.append(float(loss.item()))
                if PPO_TARGET_KL > 0.0:
                    approx_kl = float(torch.stack(approx_kls).mean().item())
                    if approx_kl > PPO_TARGET_KL:
                        stop_early = True
                        break
            if stop_early:
                break

        self.actor_critic.eval()
        self.mappo_replay.clear()
        return float(np.mean(losses)) if losses else None

    # ── Weight I/O ────────────────────────────────────────────────────────────

    def save_weights(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        buf  = io.BytesIO()
        if self.use_mappo:
            ckpt = {
                "algo": "mappo",
                "actor_critic": self.actor_critic.state_dict(),
            }
        else:
            ckpt = {
                "algo": "dqn",
                "arch": getattr(self.qnet, "arch", self.dqn_arch or "deep"),
                "qnet": self.qnet.state_dict(),
            }
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
            if self.use_mappo:
                key = "actor_critic"
                if key not in ckpt:
                    print(f"[agent_v2] checkpoint is not MAPPO: {path}")
                    return False
                self.actor_critic.load_state_dict(ckpt[key])
            else:
                ckpt_arch = ckpt.get("arch")
                live_arch = getattr(self.qnet, "arch", None)
                if ckpt_arch and live_arch and ckpt_arch != live_arch:
                    print(f"[agent_v2] checkpoint arch={ckpt_arch} differs from live arch={live_arch}: {path}")
                    return False
                self.qnet.load_state_dict(ckpt["qnet"])
                self.target_qnet.load_state_dict(ckpt["qnet"])
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

        if self.use_mappo:
            w0 = next(self.actor_critic.parameters()).data.clone()
            sv = rng.standard_normal((3, STATE_DIM)).astype(np.float32)
            gs = rng.standard_normal(MAPPO_GLOBAL_DIM).astype(np.float32)
            nf = rng.standard_normal((8, GNN_NODE_DIM)).astype(np.float32)
            ei = np.array([
                [0, 1, 1, 2, 2, 3, 4, 5],
                [1, 0, 2, 1, 3, 2, 5, 4],
            ], dtype=np.int64)
            cids = np.array([1, 2, 3], dtype=np.int64)
            for _ in range(MIN_REPLAY + 10):
                self.mappo_replay.push_trajectory([
                    {
                        "candidate_states": sv,
                        "action_idx": 0,
                        "old_logp": -1.0,
                        "value": 0.0,
                        "global_state": gs,
                        "node_features": nf,
                        "edge_index": ei,
                        "curr_id": 0,
                        "dst_id": 3,
                        "candidate_ids": cids,
                        "aux_policy_target": np.array([1.0, 0.0, 0.0], dtype=np.float32),
                        "counterfactual_target": 0.5,
                        "reward": 1.0,
                        "done": True,
                    }
                ])
            loss = self.train_mappo()
            assert loss is not None, "train_mappo returned None"
            w1 = next(self.actor_critic.parameters()).data.clone()
            assert not torch.equal(w0, w1), \
                "MAPPO actor-critic weights did not change — gradient flow broken!"
            print("[agent_v2] gradient flow assertion PASS")
            return

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
        print("[agent_v2] gradient flow assertion PASS")
