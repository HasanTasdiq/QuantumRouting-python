"""
TF-free drop-in for dist_agent_helper.py.

Provides only the constants that DQRL_dist*.py imports at module level:
  INFERENCE_MODE, WORKER_PORTS, BASE_WORKER_PORT, PREDICT_PORT,
  TRAINING_MODE, and a no-op replay_memory placeholder.

Called by the _pt algorithm variants via sys.modules patching.
"""
import os
from collections import deque

# ── Env-var config (mirrors dist_agent_helper.py) ────────────────────────────
REDIS_DB         = int(os.environ.get("REDIS_DB",         "0"))
BASE_WORKER_PORT = int(os.environ.get("BASE_WORKER_PORT", "8000"))
PREDICT_PORT     = int(os.environ.get("PREDICT_PORT",     "8080"))

REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))

INFERENCE_MODE = os.environ.get("INFERENCE_MODE", "0") == "1"

TRAINING_MODE = os.environ.get("TRAINING_MODE", "paper")

NUM_TRAINING_WORKERS = 4
WORKER_PORTS = [BASE_WORKER_PORT + i for i in range(NUM_TRAINING_WORKERS)]

MODEL_SAVE_PATH  = os.environ.get("MODEL_SAVE_PATH",
                                  "/tmp/qrouting_model/trained_weights_pt.pkl")
GLOBAL_MODEL_NAME = os.environ.get("GLOBAL_MODEL_NAME", "dqrl_model")

# Training hyper-params (mirrors dist_agent_helper.py)
if TRAINING_MODE == "paper":
    START_EPSILON_DECAYING = 3000
    END_EPSILON_DECAYING   = 8000
    REPLAY_MEMORY_SIZE     = 50_000
    MIN_REPLAY_MEMORY_SIZE = 512
    MINIBATCH_SIZE         = 512
    UPDATE_TARGET_EVERY    = 100
else:
    START_EPSILON_DECAYING = 10
    END_EPSILON_DECAYING   = 40
    REPLAY_MEMORY_SIZE     = 500
    MIN_REPLAY_MEMORY_SIZE = 20
    MINIBATCH_SIZE         = 8
    UPDATE_TARGET_EVERY    = 10

MAX_REQUESTS_SMOKE = 15
MAX_REQUESTS_PAPER = 100

SIZE = int(os.environ.get("SIZE", "100"))

# replay_memory is managed inside pt/agent.py; provide a stub so any
# import of this name doesn't crash.
replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)

AGGREGATION_EVERY = 5

def worker_model_name(worker_id: int, base: str = "dqrl_model") -> str:
    return f"{base}_worker{worker_id}"
