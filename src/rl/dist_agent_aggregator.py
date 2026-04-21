"""
FedAvg Aggregator — runs as a standalone daemon alongside the training workers.

Architecture
------------
  Training workers  (dist_agent.py  WORKER_ID=0,1,2,3  ports 8000-8003)
      │  each saves weights to   dqrl_model_worker{id}_weights
      ▼
  Aggregator  (this script)
      │  reads all worker weight blobs
      │  computes element-wise mean  (FedAvg)
      │  writes to  dqrl_model_weights  +  increments  dqrl_model_version
      ▼
  Predict server  (dist_agent_predict.py  port 8080)
      │  loads from  dqrl_model_weights  each timeslot

Usage
-----
    python dist_agent_aggregator.py

Environment variables
---------------------
  NUM_WORKERS          number of training workers (default 4)
  AGGREGATION_EVERY_S  seconds between aggregation rounds (default 30)
  AGGREGATION_EVERY_TS timeslots between aggregation (informational only;
                        actual trigger is time-based)

The aggregator runs forever until killed.  It is safe to start it before the
workers — it simply skips rounds where < 2 workers have published weights.
"""

import os
import time

from dist_agent_helper import (
    mpredis,
    fedavg_aggregate,
    NUM_TRAINING_WORKERS,
    GLOBAL_MODEL_NAME,
)

NUM_WORKERS       = int(os.environ.get("NUM_WORKERS",          NUM_TRAINING_WORKERS))
AGG_INTERVAL_S    = float(os.environ.get("AGGREGATION_EVERY_S", 30))   # seconds


def run():
    print(f"[FedAvg Aggregator] watching {NUM_WORKERS} workers, "
          f"aggregating every {AGG_INTERVAL_S:.0f}s → key '{GLOBAL_MODEL_NAME}_weights'",
          flush=True)

    round_no = 0
    while True:
        time.sleep(AGG_INTERVAL_S)
        round_no += 1
        t = time.time()
        try:
            version = fedavg_aggregate(num_workers=NUM_WORKERS,
                                       global_name=GLOBAL_MODEL_NAME)
            if version is not None:
                print(f"[FedAvg] round {round_no}  version={version}  "
                      f"took {time.time()-t:.2f}s")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            print(f"[FedAvg] round {round_no} ERROR: {exc}")


if __name__ == "__main__":
    run()
