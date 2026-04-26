"""
PyTorch port of state-construction helpers (originally in dist_agent_helper.py).

Provides:
  get_request_embeddings()     — one-hot feature vectors for req_matrix rows
  apply_request_attention()    — MHA self-attention over request features
  get_neighbor_embeddings()    — project neighbor one-hot vecs to 64-D
  apply_neighbor_attention()   — dot-product attention, curr_emb × neighbors
  schedule_routing_state_dist() — full 20228-dim state vector for one request
  get_global_state_vector()    — 10100-dim global state (ent_flat + density)
  get_mask_one_req_schedule_route() — valid action mask for one request
  get_epsilon_linear()         — epsilon schedule

All module-level dense layers (dense_proj, dense_neighbor, mha, ln) are
instantiated once so their weights are shared / learnable across calls.
"""
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SIZE = int(os.environ.get("SIZE", "100"))

# ── Module-level shared layers (mirror dist_agent_helper module globals) ─────
dense_proj     = nn.Linear(SIZE, 64)
dense_neighbor = nn.Linear(SIZE, 64)
mha            = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
ln_req         = nn.LayerNorm(64)

# Put in eval mode by default; training is handled by the agent's optimizer
dense_proj.eval()
dense_neighbor.eval()
mha.eval()
ln_req.eval()


# ── epsilon schedule ─────────────────────────────────────────────────────────
def get_epsilon_linear(timeSlot: int, eps_start: float = 1.0) -> float:
    if os.environ.get("INFERENCE_MODE", "0") == "1":
        return 0.0
    training_mode = os.environ.get("TRAINING_MODE", "paper")
    if training_mode == "paper":
        start_decay, end_decay = 3000, 8000
    elif training_mode == "mid":
        start_decay, end_decay = 200, 1500
    else:   # smoke
        start_decay, end_decay = 10, 40
    if timeSlot < start_decay:
        return eps_start
    if timeSlot >= end_decay:
        return 0.0
    ratio = (timeSlot - start_decay) / (end_decay - start_decay)
    return max(0.0, eps_start * (1.0 - ratio))


# ── Request embeddings ────────────────────────────────────────────────────────
@torch.no_grad()
def get_request_embeddings(req_matrix) -> np.ndarray:
    """
    One-hot feature for each row in req_matrix → apply dense_proj → (N, 64).
    req_matrix rows: [src, dst, current_node, path, index, done, ...]
    """
    features = []
    for req in req_matrix:
        vec = np.zeros(SIZE, dtype=np.float32)
        if not req[5]:
            src_idx = int(req[0])
            dst_idx = int(req[1])
            if 0 <= src_idx < SIZE:
                vec[src_idx] = 1.0
            if 0 <= dst_idx < SIZE:
                vec[dst_idx] = 10.0
        features.append(vec)
    t = torch.tensor(np.stack(features), dtype=torch.float32)   # (N, SIZE)
    return dense_proj(t).numpy()   # (N, 64)


@torch.no_grad()
def apply_request_attention(req_feats: np.ndarray) -> np.ndarray:
    """
    Self-attention over request feature vectors.
    Input:  (N, 64) numpy array
    Output: (N, 64) numpy array
    Matches TF: proj → MHA(q=proj, k=proj, v=proj) → residual + LayerNorm
    """
    t = torch.tensor(req_feats, dtype=torch.float32).unsqueeze(0)  # (1, N, 64)
    attn_out, _ = mha(t, t, t, need_weights=False)
    out = ln_req(t + attn_out).squeeze(0)                          # (N, 64)
    return out.numpy()


# ── Neighbor embeddings ───────────────────────────────────────────────────────

@torch.no_grad()
def precompute_all_node_embeddings() -> np.ndarray:
    """
    Batch all SIZE node embeddings in a single forward pass. Call once per
    timeslot and pass the result to get_neighbor_context() as node_emb_cache.
    Returns (SIZE, 64).
    """
    eye = torch.eye(SIZE, dtype=torch.float32)   # (SIZE, SIZE)
    return dense_neighbor(eye).numpy()            # (SIZE, 64)


def get_neighbor_context(ent_row: np.ndarray, curr_emb: np.ndarray,
                          node_emb_cache: np.ndarray) -> np.ndarray:
    """
    Combined replacement for get_neighbor_embeddings + apply_neighbor_attention.

    Uses np.nonzero (C-level) instead of a Python for-loop over SIZE nodes to
    find valid neighbours. For SIZE=100 this is ~50x faster than the loop.

    ent_row       : (SIZE,) row of ent_arr for current node
    curr_emb      : (64,)   request attention embedding
    node_emb_cache: (SIZE, 64) from precompute_all_node_embeddings()
    Returns       : (64,) context vector
    """
    neighbor_ids = np.nonzero(ent_row)[0]                  # C-level, ~100 ns
    if len(neighbor_ids) == 0:
        return np.zeros(64, dtype=np.float32)
    stack  = node_emb_cache[neighbor_ids]                  # (K, 64) — fancy index
    scores = (curr_emb @ stack.T) * 0.125                  # 1/sqrt(64) = 0.125
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    return (weights @ stack).astype(np.float32)            # (64,)


@torch.no_grad()
def get_neighbor_embeddings(state_graph, current_node_id: int,
                             node_emb_cache: np.ndarray | None = None):
    """Legacy interface kept for external callers. Prefer get_neighbor_context."""
    neighbors = state_graph[current_node_id]
    results = []
    for node_id, has_link in enumerate(neighbors):
        if has_link > 0:
            if node_emb_cache is not None:
                results.append(node_emb_cache[node_id])
            else:
                feat = np.zeros(SIZE, dtype=np.float32)
                feat[node_id] = 1.0
                t = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
                results.append(dense_neighbor(t).squeeze(0).numpy())
    return results


def apply_neighbor_attention(curr_emb: np.ndarray,
                              neighbor_embs: list) -> np.ndarray:
    """
    Dot-product attention: curr_emb (64,) over neighbors [(64,)...].
    Returns context vector (64,).
    """
    if not neighbor_embs:
        return np.zeros(64, dtype=np.float32)
    stack  = np.stack(neighbor_embs)                    # (K, 64)
    query  = curr_emb[np.newaxis, :]                    # (1, 64)
    scores = query @ stack.T / np.sqrt(64.0)            # (1, K)
    weights = np.exp(scores - scores.max())             # softmax, numerically stable
    weights /= weights.sum()
    return (weights @ stack).flatten().astype(np.float32)  # (64,)


# ── Full state vector ─────────────────────────────────────────────────────────
def schedule_routing_state_dist(curr_req, ent_matrix, req_matrix,
                                 dist_matrix) -> np.ndarray:
    """
    Builds the 20228-dim state vector for one request.
    Mirrors dist_agent_helper.schedule_routing_state_dist().
    """
    ent_arr  = np.array(ent_matrix,  dtype=np.float32)
    dist_arr = np.array(dist_matrix, dtype=np.float32)

    req_feats   = get_request_embeddings(req_matrix)       # (N, 64)
    attn_feats  = apply_request_attention(req_feats)       # (N, 64)

    curr_index  = int(curr_req[4])
    curr_emb    = attn_feats[curr_index]                   # (64,)

    neighbor_embs = get_neighbor_embeddings(ent_arr, int(curr_req[2]))
    context_vec   = apply_neighbor_attention(curr_emb, neighbor_embs)  # (64,)

    local = np.zeros(SIZE, dtype=np.float32)
    local[int(curr_req[2])] = 10.0
    local[int(curr_req[1])] = 10.0

    return np.concatenate([
        curr_emb,
        context_vec,
        local,
        ent_arr.flatten(),
        dist_arr.flatten(),
    ]).astype(np.float32)   # 64+64+100+10000+10000 = 20228


def get_global_state_vector(ent_matrix, req_matrix) -> np.ndarray:
    """
    Global state used for QMIX: (SIZE*SIZE + SIZE,) = (10100,).
    Mirrors dist_agent_helper.get_global_state_vector().
    """
    flat_ent = np.array(ent_matrix, dtype=np.float32).flatten()
    density  = np.zeros(SIZE, dtype=np.float32)
    if req_matrix is not None:
        for req in req_matrix:
            if len(req) > 5 and not req[5]:
                curr_node = int(req[2])
                if 0 <= curr_node < SIZE:
                    density[curr_node] += 1.0
    return np.concatenate([flat_ent, density])


# ── Action mask ───────────────────────────────────────────────────────────────
def get_mask_one_req_schedule_route(req_state, ent_matrix,
                                    req_matrix) -> np.ndarray:
    """
    Returns (SIZE,) mask with 1 at valid next-hop neighbours, 0 elsewhere.
    Falls back to all-ones if no valid neighbours (avoid null action space).
    """
    mask = np.zeros(SIZE, dtype=np.float32)
    src, dst, current_node_id, path, index, done = (
        req_state[0], req_state[1], int(req_state[2]),
        req_state[3], req_state[4], req_state[5],
    )
    state_graph = np.array(ent_matrix)
    neighbours = [i for i, x in enumerate(state_graph[current_node_id]) if x >= 1]
    for n in neighbours:
        if not path[n] and n != current_node_id:
            mask[n] = 1.0
    if mask.sum() == 0:
        mask[:] = 1.0   # fallback: allow all
    return mask
