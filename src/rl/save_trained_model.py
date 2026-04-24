"""
save_trained_model.py — called after training completes.

Reads the FedAvg-aggregated global model from Redis and persists it to disk
so the inference phase can load it even if Redis is restarted.
"""

import sys
import os

# Ensure the rl directory is on the path
_rl_dir = os.path.dirname(os.path.abspath(__file__))
if _rl_dir not in sys.path:
    sys.path.insert(0, _rl_dir)

from dist_agent_helper import (
    load_model_from_redis, save_model_to_disk,
    GLOBAL_MODEL_NAME, MODEL_SAVE_PATH,
)
from DQRLAgentDist_API import DQRLAgentDist

print("[save_trained_model] Initializing agent...")
agent = DQRLAgentDist()
agent.initiate()

version = load_model_from_redis(agent.model, GLOBAL_MODEL_NAME)
if version:
    save_model_to_disk(agent.model, MODEL_SAVE_PATH)
    print(f"[save_trained_model] Done — global model v{version} saved to {MODEL_SAVE_PATH}")
else:
    print("[save_trained_model] WARNING: no global model found in Redis.")
    print("  Training may not have completed or FedAvg never ran.")
    sys.exit(1)
