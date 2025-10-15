import asyncio
from concurrent.futures import ThreadPoolExecutor,wait
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
    actions: list[UpdateActionParams]

class LearnPredictRequest(BaseModel):
    reqIndex: int
    timeSlot: int
    ent_matrix: list
    req_matrix: list
    dist_matrix: list

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


# @app.post("/update_action_batch")
# async def update_action_batch(params: UpdateActionBatchParams, background_tasks: BackgroundTasks):

#     results = []
#     print(f"=============Received batch of size: {len(params.batch)}")
#     for p in params.batch:
#     #     # Call your actual function here
#     #     def task(p):
#     #         return agent.update_action(
#     #             p.reqIndex, p.current_node_id, p.next_node_id, p.current_state,
#     #             p.done_episode, p.timeSlot, p.reward, p.node_matrix, p.req_matrix, p.dist_matrix
#     #         )
#     #     background_tasks.add_task(task, p)
#         result = agent.update_action(
#             p.reqIndex, p.current_node_id, p.next_node_id, p.current_state,
#             p.done_episode, p.timeSlot, p.reward, p.node_matrix, p.req_matrix, p.dist_matrix
#         )

#     return {"results": 'success'}


# @app.post("/update_reward")
# async def call_update_reward(data: UpdateRewardRequest, background_tasks: BackgroundTasks):
#     print('In update_reward API with timeSlot:', data.timeSlot)
#     successfulRequest = data.successfulRequest
#     timeSlot = data.timeSlot
#     actions = data.actions

#     async def background_task(successfulRequest, timeSlot, actions):
#         async with semaphore:  # Wait if 10 are already running
#             try:
#                 # Run the blocking function in a thread to not block the event loop
#                 await asyncio.to_thread(agent.update_reward, successfulRequest, timeSlot, actions)
#             except Exception as e:
#                 import traceback
#                 traceback.print_exc()
#                 print(f"[BackgroundTaskError] update_reward failed: {e}")
#             finally:
#                 del successfulRequest, timeSlot, actions
#                 gc.collect()

#     # Add to background task list
#     background_tasks.add_task(asyncio.create_task, background_task(successfulRequest, timeSlot, actions))

#     return {"status": "update_reward started in background"}

def run_async_in_thread( coro):
    global active_futures
    global train_executor
    def target():
        asyncio.run(coro)

    if len(active_futures) >= 10:
        while len(active_futures):
            # Wait for at least one to finish before submitting new one
            print('Waiting for an active future to complete.......................................')
            done, pending = wait(active_futures)
            active_futures -= done



    try:
        # future = train_executor.submit(target)
        # print('Submitted a new background task. Active tasks:', len(active_futures) + 1)
        # active_futures.add(future)
        target()

    except Exception as e:
        import traceback
        traceback.print_exc()

@app.post("/update_reward")
async def call_update_reward(data: UpdateRewardRequest, background_tasks: BackgroundTasks):
    print('In update_reward API with timeSlot:', data.timeSlot)
    successfulRequest = data.successfulRequest
    timeSlot = data.timeSlot
    actions = data.actions

    def task(successfulRequest, timeSlot, actions):
        try:
            agent.update_reward(successfulRequest, timeSlot, actions)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"[BackgroundTaskError] update_reward failed: {e}")
        finally:
            del successfulRequest, timeSlot, actions
            # import gc; 
            gc.collect()
    # run_async_in_thread(task(successfulRequest, timeSlot, actions))
        

    task(successfulRequest, timeSlot, actions)
    # background_tasks.add_task(task, successfulRequest, timeSlot, actions)
    return {"status": "update_reward started in background"}

# @app.post("/update_reward")
# async def call_update_reward(request: Request, background_tasks: BackgroundTasks):
#     raw_body = await request.body()
#     background_tasks.add_task(process_update_reward, raw_body)
#     return {"status": "started"}

# def process_update_reward(raw_body):
#     import orjson
#     data = orjson.loads(raw_body)
#     agent.update_reward(data["successfulRequest"], data["timeSlot"], data["actions"])



@app.post("/learn_predict")
async def call_learn_and_predict(data: LearnPredictRequest):
    print('request received for learn_and_predict_next_req_node_single with reqIndex:')
    t = time.time()
    try:
        # Run CPU-bound function in separate thread with timeout
        result = await asyncio.wait_for(
            asyncio.to_thread(
                agent.learn_and_predict_next_req_node_single,
                data.reqIndex,
                data.ent_matrix,
                data.req_matrix,
                data.dist_matrix,
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

@app.on_event("startup")
async def startup_event():
    """Initialize per-worker agent and start memory tracing."""
    global agent
    print(f"[Worker PID {os.getpid()}] Initializing agent...")
    tracemalloc.start()
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
