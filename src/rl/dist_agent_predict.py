"""
Predict server — port 8080.

Endpoints
---------
POST /learn_predict        single-request predict (legacy, still supported)
POST /learn_predict_batch  all requests in one timeslot — 1 MHA pass + 1 batched
                           model forward pass  (Improvement 1 + 2)

Improvements vs original
-------------------------
1. Timeslot attention cache: req_matrix MHA is computed once per timeslot and
   reused for every request that shares that timeslot.  Saves (N-1) × MHA cost.

2. Batch predict endpoint: caller sends all reqIndices + a batchId that points
   to the shared matrices already stored in Redis.  One asyncio.to_thread call
   runs agent.batch_predict_all_requests(), which does a single batched forward
   pass for all N states simultaneously.
"""

import asyncio
import gc
import os
import pickle
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List

import numpy as np
import psutil
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from DQRLAgentDist_API import DQRLAgentDist
from dist_agent_helper import (
    mpredis, executor, save_replay_memory, load_replay_memory,
    get_request_embeddings, apply_request_attention,
)

app = FastAPI()
agent: DQRLAgentDist | None = None

train_executor = ThreadPoolExecutor(max_workers=10)

# ── Timeslot attention cache ────────────────────────────────────────────────
# Maps  timeSlot → np.ndarray (num_req_rows, 64)
# Reset when a new timeslot is seen; cleaned for old timeslots.
_attn_cache: Dict[int, np.ndarray] = {}
_attn_lock = asyncio.Lock()   # ensures only one coroutine populates a slot


# ── Pydantic models ──────────────────────────────────────────────────────────
class LearnPredictRequest(BaseModel):
    reqIndex: int
    timeSlot: int
    reqId:    str

class BatchPredictRequest(BaseModel):
    """
    Batch endpoint: caller uploads matrices once under ``batchId``, then
    sends all reqIndices together.  One HTTP round-trip per timeslot.
    """
    batchId:    str
    reqIndices: List[int]
    timeSlot:   int


# ── Helpers ──────────────────────────────────────────────────────────────────
def _pop_matrices(reqId: str):
    """Fetch and delete the three matrices stored for a single-request call."""
    ent  = pickle.loads(mpredis.get(f"reqId_{reqId}_ent_matrix"))
    req  = pickle.loads(mpredis.get(f"reqId_{reqId}_req_matrix"))
    dist = pickle.loads(mpredis.get(f"reqId_{reqId}_dist_matrix"))
    mpredis.delete(f"reqId_{reqId}_ent_matrix",
                   f"reqId_{reqId}_req_matrix",
                   f"reqId_{reqId}_dist_matrix")
    return ent, req, dist


def _pop_batch_matrices(batchId: str):
    """Fetch and delete the three matrices stored for a batch call."""
    ent  = pickle.loads(mpredis.get(f"batch_{batchId}_ent"))
    req  = pickle.loads(mpredis.get(f"batch_{batchId}_req"))
    dist = pickle.loads(mpredis.get(f"batch_{batchId}_dist"))
    mpredis.delete(f"batch_{batchId}_ent",
                   f"batch_{batchId}_req",
                   f"batch_{batchId}_dist")
    return ent, req, dist


async def _get_or_compute_attn(req_matrix, timeSlot: int) -> np.ndarray:
    """
    Return cached MHA attention for ``timeSlot``, computing it if needed.
    The asyncio.Lock ensures only the first coroutine does the work;
    subsequent ones for the same timeslot wait and then reuse the result.
    """
    async with _attn_lock:
        if timeSlot not in _attn_cache:
            req_tensor = get_request_embeddings(req_matrix.tolist()
                                                if hasattr(req_matrix, 'tolist')
                                                else req_matrix)
            # Offload TF computation to thread so we don't block the event loop
            attn = await asyncio.to_thread(
                lambda: apply_request_attention(req_tensor).numpy()
            )
            _attn_cache[timeSlot] = attn
            # Evict entries older than 3 timeslots to bound memory
            for old_ts in [k for k in _attn_cache if k < timeSlot - 2]:
                del _attn_cache[old_ts]
    return _attn_cache[timeSlot]


# ── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/learn_predict")
async def call_learn_and_predict(data: LearnPredictRequest):
    """
    Single-request endpoint (legacy).  Now uses cached MHA attention so the
    N-th request for a timeslot pays no extra MHA cost.
    """
    t = time.time()
    ent_matrix, req_matrix, dist_matrix = _pop_matrices(data.reqId)
    print(f'[predict] matrices loaded in {time.time()-t:.3f}s  reqIdx={data.reqIndex}')

    # Ensure attention is cached for this timeslot
    attn_encoded = await _get_or_compute_attn(req_matrix, data.timeSlot)

    try:
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent.predict_with_attn,
                data.reqIndex,
                ent_matrix.tolist(),
                req_matrix.tolist(),
                dist_matrix.tolist(),
                data.timeSlot,
                attn_encoded,
            ),
            timeout=30,
        )
    except asyncio.TimeoutError:
        print(f"WARNING: /learn_predict timed out after 30s  reqIdx={data.reqIndex}")
        return {"error": "Request timed out"}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e)}

    if result is None:
        return {"result": None}

    print(f'[predict] single done in {time.time()-t:.3f}s  reqIdx={data.reqIndex}')
    return {"result": [[], result[1]]}


@app.post("/learn_predict_batch")
async def call_learn_and_predict_batch(data: BatchPredictRequest):
    """
    Batch endpoint — processes all requests for a timeslot in one call.

    Speed improvements vs N individual /learn_predict calls:
      • MHA computed once (Improvement 1 — attention cache)
      • Single batched model forward pass for N states (Improvement 2)
      • One HTTP round-trip instead of N

    Returns: {"results": {"<reqIndex>": [[], action_int], ...}}
    """
    t = time.time()
    ent_matrix, req_matrix, dist_matrix = _pop_batch_matrices(data.batchId)
    print(f'[batch_predict] matrices loaded in {time.time()-t:.3f}s  '
          f'n={len(data.reqIndices)} ts={data.timeSlot}')

    try:
        results = await asyncio.wait_for(
            asyncio.to_thread(
                agent.batch_predict_all_requests,
                data.reqIndices,
                ent_matrix.tolist(),
                req_matrix.tolist(),
                dist_matrix.tolist(),
                data.timeSlot,
            ),
            timeout=60,
        )
    except asyncio.TimeoutError:
        print(f"WARNING: /learn_predict_batch timed out  n={len(data.reqIndices)}")
        return {"error": "Request timed out", "results": {}}
    except Exception as e:
        import traceback; traceback.print_exc()
        return {"error": str(e), "results": {}}

    elapsed = time.time() - t
    print(f'[batch_predict] {len(results)}/{len(data.reqIndices)} results in {elapsed:.3f}s')
    # Return only the action; state is large and unused by the caller
    return {"results": {str(k): [[], v[1]] for k, v in results.items()}}


# ── Lifecycle ────────────────────────────────────────────────────────────────
process_info = psutil.Process(os.getpid())

@app.get("/debug_memory")
def debug_memory():
    mem = process_info.memory_info().rss / 1024 / 1024
    return {"rss_MB": round(mem, 2)}


@app.on_event("shutdown")
async def shutdown_event():
    print(f"[Predict PID {os.getpid()}] Shutting down...")
    executor.shutdown(wait=True)
    print("Shutdown complete.")


@app.on_event("startup")
async def startup_event():
    global agent
    print(f"[Predict PID {os.getpid()}] Initializing agent...")
    tracemalloc.start()
    agent = DQRLAgentDist()
    agent.initiate()
    # Predict server always reads the global (FedAvg-aggregated) model
    print(f"[Predict PID {os.getpid()}] Agent ready  model_name={agent.model_name}")


if __name__ == "__main__":
    import os as _os
    _port = int(_os.environ.get("PREDICT_PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=_port)
