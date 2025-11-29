import asyncio
from concurrent.futures import ThreadPoolExecutor,wait
import pickle
import time
from fastapi import FastAPI, BackgroundTasks, Request
from pydantic import BaseModel
import uvicorn
from DQRLAgentDist_API import DQRLAgentDist  # replace with your actual module
import numpy as np
import psutil
import os
app = FastAPI()
agent = None  # global agent reference per worker
import gc
from dist_agent_helper import mpredis, executor, save_replay_memory, load_replay_memory

train_executor = ThreadPoolExecutor(max_workers=10)
active_futures = set()

class UpdateActionParams(BaseModel):
    reqIndex: int
    current_node_id: int
    next_node_id: int
    current_state: list
    done_episode: bool
    timeSlot: int
    reward: float
    node_matrix: list
    req_matrix: list
    dist_matrix: list

# Model for a batch
class UpdateActionBatchParams(BaseModel):
    batch: list[UpdateActionParams]

class UpdateRewardRequest(BaseModel):
    successfulRequest: int
    timeSlot: int
    actionIds: list[str]
    actions: list[UpdateActionParams]

class LearnPredictRequest(BaseModel):
    reqIndex: int
    timeSlot: int
    reqId: str

def make_json_safe(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.generic):  # NumPy scalar like int64, float32
        return obj.item()
    elif isinstance(obj, list):
        return [make_json_safe(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    else:
        return obj



@app.post("/learn_predict")
async def call_learn_and_predict(data: LearnPredictRequest):
    print('request received for learn_and_predict_next_req_node_single with reqIndex:')
    t = time.time()
    ent_matrix = pickle.loads(mpredis.get(f"reqId_{data.reqId}_ent_matrix"))
    req_matrix = pickle.loads(mpredis.get(f"reqId_{data.reqId}_req_matrix"))
    dist_matrix = pickle.loads(mpredis.get(f"reqId_{data.reqId}_dist_matrix"))

    mpredis.delete(f"reqId_{data.reqId}_ent_matrix")
    mpredis.delete(f"reqId_{data.reqId}_req_matrix")
    mpredis.delete(f"reqId_{data.reqId}_dist_matrix")
    print('======================got matrices from redis time ' , time.time() - t)
    try:
        # Run CPU-bound function in separate thread with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent.learn_and_predict_next_req_node_single,
                data.reqIndex,
                ent_matrix.tolist(),
                req_matrix.tolist(),
                dist_matrix.tolist(),
                data.timeSlot
            ),
            timeout=5  # timeout in seconds
        )
    except asyncio.TimeoutError:
        return {"error": "Request timed out"}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"error": str(e)}

    print('==============learn_and_predict result time:', time.time() - t, 's')
    return {"result": [[], result[1]]}

process = psutil.Process(os.getpid())

@app.get("/debug_memory")
def debug_memory():
    mem = process.memory_info().rss / 1024 / 1024
    return {"rss_MB": round(mem, 2)}
import tracemalloc

@app.on_event("shutdown")
async def shutdown_event():
    global executor
    """Only called when FastAPI app is shutting down"""
    print(f"[Worker PID {os.getpid()}] Shutting down...")
    print("Shutting down executor...")
    executor.shutdown(wait=True)
    print("Saving replay memory...")
    # save_replay_memory()
    print('replay memory saved.')
    print("Shutdown complete.")

@app.on_event("startup")
async def startup_event():
    """Initialize per-worker agent and start memory tracing."""
    global agent
    print(f"[Worker PID {os.getpid()}] Initializing agent...")
    tracemalloc.start()
    # load_replay_memory()

    agent = DQRLAgentDist()
    agent.initiate()
    print(f"[Worker PID {os.getpid()}] Agent ready.")


@app.get("/snapshot")
def snapshot():
    snapshot = tracemalloc.take_snapshot()
    # Group by traceback instead of just lineno
    top_stats = snapshot.statistics("traceback")

    report = []
    for stat in top_stats[:10]:  # top 10 memory hogs
        block = {
            "size_MB": round(stat.size / 1024 / 1024, 2),
            "count": stat.count,
            "traceback": stat.traceback.format()  # list of stack frames
        }
        report.append(block)

    return {"top": report}

if __name__ == "__main__":
    # agent = DQRLAgentDist()
    # agent.initiate()
    uvicorn.run(app, host="0.0.0.0", port=8000)
