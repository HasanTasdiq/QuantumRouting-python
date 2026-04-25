"""
save_trained_model_pt.py — PT replacement for src/rl/save_trained_model.py.

Reads the FedAvg-aggregated global model from Redis and persists it to disk.
Called at the end of training phase in smoke_test.sh / deploy_full.sh.
"""
import os, sys

_rl_pt_dir = os.path.dirname(os.path.abspath(__file__))
_rl_dir    = os.path.dirname(_rl_pt_dir)
if _rl_pt_dir not in sys.path: sys.path.insert(0, _rl_pt_dir)
if _rl_dir    not in sys.path: sys.path.insert(0, _rl_dir)

from pt.agent import DQRLAgentDist
from redis_io import get_redis, load_model_from_redis, save_model_to_disk, \
                      MODEL_SAVE_PATH

GLOBAL_MODEL_NAME = os.environ.get("GLOBAL_MODEL_NAME", "dqrl_model")

def main():
    r = get_redis()
    print("[save_model_pt] Initializing agent...")
    agent = DQRLAgentDist()

    version = load_model_from_redis(agent.model, GLOBAL_MODEL_NAME, r)
    if version:
        save_model_to_disk(agent.model, MODEL_SAVE_PATH)
        print(f"[save_model_pt] global model v{version} → {MODEL_SAVE_PATH}")
    else:
        print("[save_model_pt] WARNING: no global model in Redis.")
        print("  Training may not have completed or FedAvg never ran.")
        sys.exit(1)

if __name__ == "__main__":
    main()
