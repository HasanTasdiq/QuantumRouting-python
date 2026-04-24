"""
PyTorch predict/worker server — replaces src/rl/dist_agent_predict.py
                                          and src/rl/dist_agent.py.

Role is determined by env vars:
  WORKER_ID=-1  (default)  → Predict server on PREDICT_PORT (8080)
                              Handles /learn_predict_batch, /learn_predict
                              Runs FedAvg aggregation in background
  WORKER_ID=0..N-1         → Training worker on BASE_WORKER_PORT+WORKER_ID
                              Handles /update_reward (receives transitions, trains)
                              Periodically syncs global model from Redis

All roles expose /health and /debug_memory.

Redis key schema (shared with original TF layout):
  {model_name}_weights          global model (FedAvg output)
  {model_name}_worker{id}_weights  per-worker model
  action_{actionId}             pickled action tuple written by routing algo
  batch_{ts}                    pickled batch (written by routing algo)
  reqId_{id}_ent/req/dist_matrix  per-request matrices
  batch_{batchId}_ent/req/dist  per-batch matrices
"""

import asyncio
import gc
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Optional

import numpy as np
import psutil
import redis
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# ── Path setup ───────────────────────────────────────────────────────────────
import sys
_rl_pt_dir = os.path.dirname(os.path.abspath(__file__))
_rl_dir    = os.path.dirname(_rl_pt_dir)
if _rl_dir    not in sys.path: sys.path.insert(0, _rl_dir)
if _rl_pt_dir not in sys.path: sys.path.insert(0, _rl_pt_dir)

from agent   import DQRLAgentDist
from helpers import (
    get_request_embeddings, apply_request_attention,
    get_neighbor_embeddings, apply_neighbor_attention,
    schedule_routing_state_dist, get_global_state_vector,
    get_mask_one_req_schedule_route, get_epsilon_linear,
    SIZE,
)
from redis_io import (
    get_redis, load_model_from_disk, load_model_from_redis,
    save_model_to_redis, save_worker_model_to_redis,
    save_model_to_disk, fedavg_aggregate,
)

# ── Config ───────────────────────────────────────────────────────────────────
INFERENCE_MODE    = os.environ.get("INFERENCE_MODE",    "0") == "1"
GLOBAL_MODEL_NAME = os.environ.get("GLOBAL_MODEL_NAME", "dqrl_model")
WORKER_ID         = int(os.environ.get("WORKER_ID",     "-1"))
PREDICT_PORT      = int(os.environ.get("PREDICT_PORT",  "8080"))
BASE_WORKER_PORT  = int(os.environ.get("BASE_WORKER_PORT", "8000"))
NUM_WORKERS       = int(os.environ.get("NUM_WORKERS",   "4"))
AGG_INTERVAL_S    = int(os.environ.get("AGG_INTERVAL_S", "30"))
TRAINING_MODE     = os.environ.get("TRAINING_MODE", "paper")

# Training steps between replay calls
STEP_BETWEEN_TRAIN = 5 if TRAINING_MODE == "smoke" else 200
# How many replay steps before pushing worker weights to Redis for FedAvg
FEDAVG_PUSH_EVERY  = 50 if TRAINING_MODE == "smoke" else 500

IS_PREDICT_SERVER = (WORKER_ID == -1)

app = FastAPI()
agent: DQRLAgentDist = None
_r:    redis.Redis    = None
train_executor = ThreadPoolExecutor(max_workers=4)

# ── Per-timeslot attention cache (predict server) ────────────────────────────
_attn_cache: Dict[int, np.ndarray] = {}
_attn_lock = asyncio.Lock()

# ── Training state (worker) ──────────────────────────────────────────────────
_train_step_counter: int = 0
_replay_call_counter: int = 0
_loss_history: List[float] = []


# ── Pydantic models ──────────────────────────────────────────────────────────
class LearnPredictRequest(BaseModel):
    reqIndex: int
    timeSlot: int
    reqId:    str


class BatchPredictRequest(BaseModel):
    batchId:    str
    reqIndices: List[int]
    timeSlot:   int


class UpdateRewardRequest(BaseModel):
    successfulRequest: int
    timeSlot:          int
    actions:           list         # always empty list (data is in Redis)
    actionIds:         List[str]


# ── Redis matrix helpers ─────────────────────────────────────────────────────
def _pop_matrices(reqId: str):
    ent  = pickle.loads(_r.get(f"reqId_{reqId}_ent_matrix"))
    req  = pickle.loads(_r.get(f"reqId_{reqId}_req_matrix"))
    dist = pickle.loads(_r.get(f"reqId_{reqId}_dist_matrix"))
    _r.delete(f"reqId_{reqId}_ent_matrix",
              f"reqId_{reqId}_req_matrix",
              f"reqId_{reqId}_dist_matrix")
    return ent, req, dist


def _pop_batch_matrices(batchId: str):
    ent  = pickle.loads(_r.get(f"batch_{batchId}_ent"))
    req  = pickle.loads(_r.get(f"batch_{batchId}_req"))
    dist = pickle.loads(_r.get(f"batch_{batchId}_dist"))
    _r.delete(f"batch_{batchId}_ent",
              f"batch_{batchId}_req",
              f"batch_{batchId}_dist")
    return ent, req, dist


async def _get_or_compute_attn(req_matrix, timeSlot: int) -> np.ndarray:
    async with _attn_lock:
        if timeSlot not in _attn_cache:
            feats = get_request_embeddings(req_matrix)
            attn  = await asyncio.to_thread(apply_request_attention, feats)
            _attn_cache[timeSlot] = attn
            for old in [k for k in _attn_cache if k < timeSlot - 2]:
                del _attn_cache[old]
    return _attn_cache[timeSlot]


# ── Single predict ───────────────────────────────────────────────────────────
def _predict_single(req_idx: int, ent_matrix, req_matrix, dist_matrix,
                    timeSlot: int, attn_encoded: np.ndarray):
    req = list(req_matrix[req_idx][:6])
    req[3] = req_matrix[req_idx + len(req_matrix) // 2]
    req[0] = int(req[0]); req[1] = int(req[1])
    req[2] = int(req[2]); req[4] = int(req[4])
    if req[5]:
        return None

    ent_arr = np.array(ent_matrix, dtype=np.float32)
    curr_emb    = attn_encoded[req_idx]
    neighbor_embs = get_neighbor_embeddings(ent_arr, req[2])
    context_vec   = apply_neighbor_attention(curr_emb, neighbor_embs)

    local = np.zeros(SIZE, dtype=np.float32)
    local[req[2]] = 10.0
    local[req[1]] = 10.0

    state = np.concatenate([
        curr_emb, context_vec, local,
        ent_arr.flatten(),
        np.array(dist_matrix, dtype=np.float32).flatten(),
    ]).astype(np.float32)

    mask = get_mask_one_req_schedule_route(req, ent_matrix, req_matrix)
    eps  = get_epsilon_linear(timeSlot)
    if np.random.random() > eps:
        qs = agent.batch_predict(state[np.newaxis, :])[0]
        action = int(np.argmax(np.where(mask == 1, qs, -np.inf)))
    else:
        valid = np.where(mask == 1)[0]
        action = int(np.random.choice(valid))

    return [state.tolist(), action]


@app.post("/learn_predict")
async def call_learn_and_predict(data: LearnPredictRequest):
    t = time.time()
    try:
        ent_matrix, req_matrix, dist_matrix = _pop_matrices(data.reqId)
        attn = await _get_or_compute_attn(req_matrix, data.timeSlot)
        result = await asyncio.wait_for(
            asyncio.to_thread(
                _predict_single,
                data.reqIndex, ent_matrix, req_matrix, dist_matrix,
                data.timeSlot, attn,
            ),
            timeout=30,
        )
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

    if result is None:
        return {"result": None}
    print(f"[predict] single done in {time.time()-t:.3f}s  reqIdx={data.reqIndex}")
    return {"result": [[], result[1]]}


# ── Batch predict ────────────────────────────────────────────────────────────
def _batch_predict_all(req_indices, ent_matrix, req_matrix, dist_matrix,
                       timeSlot: int):
    ent_arr  = np.array(ent_matrix,  dtype=np.float32)
    dist_arr = np.array(dist_matrix, dtype=np.float32)
    ent_flat  = ent_arr.flatten()
    dist_flat = dist_arr.flatten()

    feats        = get_request_embeddings(req_matrix)
    attn_encoded = apply_request_attention(feats)

    valid = []
    for req_idx in req_indices:
        req = list(req_matrix[req_idx][:6])
        req[3] = req_matrix[req_idx + len(req_matrix) // 2]
        req[0] = int(req[0]); req[1] = int(req[1])
        req[2] = int(req[2]); req[4] = int(req[4])
        if req[5]:
            continue
        curr_emb  = attn_encoded[req_idx]
        n_embs    = get_neighbor_embeddings(ent_arr, req[2])
        ctx       = apply_neighbor_attention(curr_emb, n_embs)
        local     = np.zeros(SIZE, dtype=np.float32)
        local[req[2]] = 10.0; local[req[1]] = 10.0
        state = np.concatenate([curr_emb, ctx, local, ent_flat, dist_flat]).astype(np.float32)
        valid.append((req_idx, req, state))

    if not valid:
        return {}

    eps = get_epsilon_linear(timeSlot)
    states_batch = np.stack([v[2] for v in valid])
    if np.random.random() > eps:
        qs_batch = agent.batch_predict(states_batch)
    else:
        qs_batch = np.random.rand(len(valid), SIZE).astype(np.float32)

    results = {}
    for i, (req_idx, req, state) in enumerate(valid):
        mask   = get_mask_one_req_schedule_route(req, ent_matrix, req_matrix)
        action = int(np.argmax(np.where(mask == 1, qs_batch[i], -np.inf)))
        results[req_idx] = [state.tolist(), action]
    return results


@app.post("/learn_predict_batch")
async def call_learn_and_predict_batch(data: BatchPredictRequest):
    t = time.time()
    try:
        ent_matrix, req_matrix, dist_matrix = _pop_batch_matrices(data.batchId)
        results = await asyncio.wait_for(
            asyncio.to_thread(
                _batch_predict_all,
                data.reqIndices, ent_matrix, req_matrix, dist_matrix,
                data.timeSlot,
            ),
            timeout=60,
        )
    except asyncio.TimeoutError:
        return {"error": "Request timed out", "results": {}}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e), "results": {}}

    elapsed = time.time() - t
    print(f"[batch_predict] {len(results)}/{len(data.reqIndices)} in {elapsed:.3f}s")
    return {"results": {str(k): [[], v[1]] for k, v in results.items()}}


# ── Training receiver (/update_reward) ──────────────────────────────────────
def _process_transitions(action_ids: List[str]) -> int:
    """
    Read action tuples from Redis, push to replay buffer.
    Action tuple format (set by DQRL_dist1.py route_schedule_single2):
      [index, curr_node_id, next_node_id, curr_state, done_episode,
       timeSlot, reward, (ent_matrix, req_matrix), dist_matrix, a_id]
    """
    pushed = 0
    for action_id in action_ids:
        raw = _r.get(f"action_{action_id}")
        if raw is None:
            continue
        try:
            data = pickle.loads(raw)
            req_idx    = int(data[0])
            next_node  = int(data[2])           # action = next node id
            curr_state = np.array(data[3], dtype=np.float32)
            done       = bool(data[4])
            reward     = float(data[6])

            # Reconstruct next state from saved matrices
            try:
                ent_m, req_m = data[7]
                dist_m       = data[8]
                curr_req     = req_m[req_idx][:6]
                curr_req[3]  = req_m[req_idx + len(req_m) // 2]
                next_state   = schedule_routing_state_dist(
                    curr_req, ent_m, req_m, dist_m)
            except Exception:
                next_state = curr_state   # fallback

            agent.remember(curr_state, next_node, reward, next_state, done)
            pushed += 1
        except Exception:
            import traceback; traceback.print_exc()
    return pushed


@app.post("/update_reward")
async def update_reward(data: UpdateRewardRequest):
    global _train_step_counter, _replay_call_counter

    if INFERENCE_MODE:
        return {"status": "inference_mode"}

    pushed = await asyncio.to_thread(_process_transitions, data.actionIds)
    _train_step_counter += pushed

    loss: Optional[float] = None
    if _train_step_counter >= STEP_BETWEEN_TRAIN:
        _train_step_counter = 0
        loss = await asyncio.to_thread(agent.replay)
        if loss is not None:
            _loss_history.append(loss)
            _replay_call_counter += 1
            print(f"[worker{WORKER_ID}] ts={data.timeSlot}"
                  f"  replay#{_replay_call_counter}  loss={loss:.5f}"
                  f"  buf={len(agent.single_replay)}")

        # Push worker model to Redis for FedAvg
        if not IS_PREDICT_SERVER and _replay_call_counter % FEDAVG_PUSH_EVERY == 0:
            await asyncio.to_thread(
                save_worker_model_to_redis, agent.model, WORKER_ID,
                GLOBAL_MODEL_NAME, _r)

    return {"status": "ok", "pushed": pushed, "loss": loss}


# ── Health & debug ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status":    "ok",
        "worker_id": WORKER_ID,
        "buf_size":  len(agent.single_replay) if agent else 0,
        "replays":   _replay_call_counter,
    }


@app.get("/debug_memory")
def debug_memory():
    proc = psutil.Process(os.getpid())
    return {
        "rss_MB":     round(proc.memory_info().rss / 1024 / 1024, 2),
        "loss_hist":  _loss_history[-10:],
        "replay_n":   _replay_call_counter,
    }


@app.get("/loss_history")
def loss_history():
    """Smoke-test endpoint: return full loss history for learning verification."""
    return {"losses": _loss_history}


# ── Background FedAvg (predict server only) ───────────────────────────────────
async def _fedavg_loop():
    """Runs on the predict server. Every AGG_INTERVAL_S seconds, aggregate
    all worker models and broadcast the global model back to all workers."""
    await asyncio.sleep(AGG_INTERVAL_S)   # initial wait
    while True:
        try:
            version = await asyncio.to_thread(
                fedavg_aggregate, NUM_WORKERS, GLOBAL_MODEL_NAME,
                GLOBAL_MODEL_NAME, _r)
            if version:
                # Pull the fresh global model into predict server too
                await asyncio.to_thread(
                    load_model_from_redis, agent.model,
                    GLOBAL_MODEL_NAME, _r)
                print(f"[FedAvg] global model v{version} pulled into predict server")
        except Exception as e:
            print(f"[FedAvg] error: {e}")
        await asyncio.sleep(AGG_INTERVAL_S)


# ── Background global model sync (workers) ───────────────────────────────────
async def _sync_global_loop():
    """Workers periodically pull the FedAvg global model."""
    await asyncio.sleep(AGG_INTERVAL_S * 2)
    while True:
        try:
            version = await asyncio.to_thread(
                load_model_from_redis, agent.model, GLOBAL_MODEL_NAME, _r)
            if version:
                agent.target_model.load_state_dict(agent.model.state_dict())
                print(f"[worker{WORKER_ID}] synced global model v{version}")
        except Exception as e:
            print(f"[worker{WORKER_ID}] sync error: {e}")
        await asyncio.sleep(AGG_INTERVAL_S * 2)


# ── Lifecycle ─────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global agent, _r
    _r    = get_redis()
    agent = DQRLAgentDist()
    role  = "PREDICT" if IS_PREDICT_SERVER else f"WORKER-{WORKER_ID}"
    mode  = "INFERENCE" if INFERENCE_MODE else "TRAINING"
    print(f"[PT server PID {os.getpid()}] {role} ({mode}) initializing...")

    if INFERENCE_MODE:
        loaded = load_model_from_disk(agent.model)
        if not loaded:
            loaded = load_model_from_redis(agent.model, GLOBAL_MODEL_NAME, _r) is not None
        print(f"[PT server] INFERENCE mode — weights_loaded={loaded}")

    if not INFERENCE_MODE:
        if IS_PREDICT_SERVER:
            asyncio.create_task(_fedavg_loop())
            print(f"[PT server] FedAvg loop started (every {AGG_INTERVAL_S}s)")
        else:
            asyncio.create_task(_sync_global_loop())
            print(f"[PT server] global sync loop started")

    print(f"Application startup complete")   # ← smoke_test.sh waits for this


@app.on_event("shutdown")
async def shutdown_event():
    train_executor.shutdown(wait=True)
    gc.collect()


if __name__ == "__main__":
    if IS_PREDICT_SERVER:
        port = PREDICT_PORT
    else:
        port = BASE_WORKER_PORT + WORKER_ID
    uvicorn.run(app, host="0.0.0.0", port=port)
