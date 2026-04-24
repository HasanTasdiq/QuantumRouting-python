"""
PT FedAvg aggregator — optional standalone alternative to the built-in
_fedavg_loop() in server.py.  Useful when the predict server is under heavy
load and you want FedAvg on a separate CPU core.

Usage:
    python -u aggregator.py
Env vars:
    NUM_WORKERS         number of training workers (default 4)
    AGGREGATION_EVERY_S interval in seconds        (default 30)
    REDIS_DB / REDIS_HOST / REDIS_PORT
    GLOBAL_MODEL_NAME
"""
import os, sys, time

_rl_pt_dir = os.path.dirname(os.path.abspath(__file__))
if _rl_pt_dir not in sys.path:
    sys.path.insert(0, _rl_pt_dir)

from redis_io import get_redis, fedavg_aggregate

NUM_WORKERS   = int(os.environ.get("NUM_WORKERS",         "4"))
INTERVAL_S    = int(os.environ.get("AGGREGATION_EVERY_S", "30"))
GLOBAL_NAME   = os.environ.get("GLOBAL_MODEL_NAME", "dqrl_model")

def main():
    r = get_redis()
    print(f"[PT aggregator] FedAvg every {INTERVAL_S}s  workers={NUM_WORKERS}")
    while True:
        time.sleep(INTERVAL_S)
        v = fedavg_aggregate(NUM_WORKERS, GLOBAL_NAME, GLOBAL_NAME, r)
        if v is None:
            print("[PT aggregator] not enough workers ready — waiting")

if __name__ == "__main__":
    main()
