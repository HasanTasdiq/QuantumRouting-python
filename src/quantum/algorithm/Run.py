"""
Main experiment runner.

Algorithms:
  QuRA_DQRL_DIST  (Seq)   — in-process DQN routing     (v2: GAT + edge-score Q)
  QuRA_Flock_DIST         — parallel DQN routing
  QuRA_Guard_DIST         — parallel DQN + b-matching
  QuRA_Hive_DIST          — parallel DQN + b-matching + QMIX
  RELiQ_Adapter           — pre-trained RELiQ DQN baseline (F_min gated)
  EBSPA                   — Dijkstra on -log(fidelity), deterministic baseline

Environment variables
---------------------
  INFERENCE_MODE=1        evaluate frozen weights across all loads (no training)
  TTIME=N                 total timeslots (default 10000)
  TOTAL_REQUESTS=N        optional fixed request budget per load
  STEP=N                  CSV sample stride (default 1000)
  TIMES=N                 independent repetitions (default 1)
  TRAIN_LOAD=N            request load used during training (default 100)
  REQ_LOADS=5,10,25       comma-separated loads for inference (default: 6 paper loads)
  RUN_ALGOS=A,B           optional comma-separated algorithm-name filter
  TRAINING_MODE=smoke|mid|paper
  SEED=N                  optional reproducibility seed
  MODEL_DIR=/path         where QuRA weights are saved/loaded (default /tmp/qrouting_model)
"""
import copy
import gc
import math
import multiprocessing
import multiprocessing.context as ctx
import os
import subprocess
import sys
import time
import random as _random
from random import sample

import numpy as np

sys.setrecursionlimit(2000)
ctx._force_start_method('spawn')

# ── Path setup: make src/rl/pt importable ─────────────────────────────────────
_algo_dir = os.path.dirname(os.path.abspath(__file__))   # src/quantum/algorithm
_src_dir  = os.path.normpath(os.path.join(_algo_dir, '../..'))   # src
_rl_dir   = os.path.join(_src_dir, 'rl')                         # src/rl

# Add src/rl (not src/rl/pt) so local_trainer's relative imports resolve correctly
# inside the `pt` package (from .agent import ..., from .replay import ..., etc.)
for _p in [_algo_dir, _src_dir, _rl_dir]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from AlgorithmBase import AlgorithmResult
from pt.local_trainer_v2 import (
    QuRA_DQRL_DIST,
    QuRA_Flock_DIST,
    QuRA_Guard_DIST,
    QuRA_Hive_DIST,
)
from RELiQ_Adapter import RELiQ_Adapter
from pt.ebspa import EBSPA
from topo.Topo import Topo

_project_root = os.path.normpath(os.path.join(_algo_dir, '../../..'))
from topo.mp_helper import executor as executor

# ── Run configuration ─────────────────────────────────────────────────────────
INFERENCE_MODE  = os.environ.get("INFERENCE_MODE",  "0")    == "1"
SEED_ENV = os.environ.get("SEED")
BASE_SEED = int(SEED_ENV) if SEED_ENV is not None else None

def _seed_everything(seed: int) -> None:
    _random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

if BASE_SEED is not None:
    _seed_everything(BASE_SEED)

# TRAINING_MODE controls the QuRA replay/epsilon schedule (paper | mid | smoke).
# Must be set BEFORE importing local_trainer so replay.py reads the right values.
# Expose a default so the child processes (spawn) see the same value.
_tmode = os.environ.get("TRAINING_MODE", "paper")
os.environ["TRAINING_MODE"] = _tmode   # propagate to spawned child processes

ttime    = int(os.environ.get("TTIME",  "10000"))
TOTAL_REQUESTS = int(os.environ.get("TOTAL_REQUESTS", "0"))
step     = int(os.environ.get("STEP",   "1000"))
times    = int(os.environ.get("TIMES",  "1"))
nodeNo   = int(os.environ.get("SIZE",   "100"))
gridSize = int(math.sqrt(nodeNo))

alpha_  = float(os.environ.get("ALPHA", "0.0002"))
SWAP_PROB = float(os.environ.get("SWAP_PROB", os.environ.get("Q", "0.9")))
degree  = 1

# TRAIN_LOAD=25 is the max-feasible training load: at LOAD=100 the network
# saturates (162 entangled links / 4.2 hops/req ≈ 38 req/slot capacity), so
# p_complete≈0 and the model sees no positive transitions.  Train at 25 (where
# p_complete≈0.8%) and let the load-agnostic state representation generalise
# to higher inference loads.
TRAIN_LOAD           = int(os.environ.get("TRAIN_LOAD", "25"))
_INFER_LOADS_DEFAULT = [5, 10, 25, 50, 75, 100]

# Curriculum: each training timeslot gets a random load in [TRAIN_LOAD_MIN,
# TRAIN_LOAD].  Default min=5 (always trains on at least the lowest infer load).
TRAIN_LOAD_MIN = int(os.environ.get("TRAIN_LOAD_MIN", "5"))
_algos_env = os.environ.get("RUN_ALGOS", "").strip()
RUN_ALGOS = {x.strip() for x in _algos_env.split(",") if x.strip()}

_req_env = os.environ.get("REQ_LOADS", "")
if _req_env:
    numOfRequestPerRound = [int(x.strip()) for x in _req_env.split(",")]
elif INFERENCE_MODE:
    numOfRequestPerRound = _INFER_LOADS_DEFAULT
else:
    numOfRequestPerRound = [TRAIN_LOAD]

print(f"[Run.py] mode={'INFERENCE' if INFERENCE_MODE else 'TRAINING'}"
      f"  training_mode={_tmode}"
      f"  ttime={ttime}  step={step}  times={times}"
      f"  q={SWAP_PROB:g}  alpha={alpha_:g}"
      + (f"  seed={BASE_SEED}" if BASE_SEED is not None else "")
      + (f"  F_MIN={os.environ['F_MIN']}" if "F_MIN" in os.environ else "")
      + (f"  EPS_MIN={os.environ['EPS_MIN']}" if "EPS_MIN" in os.environ else "")
      + (f"  EPS_WARMUP={os.environ['EPS_WARMUP']}" if "EPS_WARMUP" in os.environ else "")
      + (f"  EPS_DECAY_END={os.environ['EPS_DECAY_END']}" if "EPS_DECAY_END" in os.environ else "")
      + (f"  total_requests={TOTAL_REQUESTS}" if TOTAL_REQUESTS > 0 else "")
      + f"  loads={numOfRequestPerRound}"
      + (f"  curriculum=[{TRAIN_LOAD_MIN},{TRAIN_LOAD}]" if not INFERENCE_MODE else ""))


# ── Per-algorithm thread ───────────────────────────────────────────────────────

def runThread(algo, requests, algoIndex, ttime, pid, resultDict, shared_data):
    _t_start = time.time()
    if BASE_SEED is not None:
        _seed_everything(BASE_SEED)
    # Cap PyTorch/OpenMP threads per worker process.  With 6 algorithms running
    # in parallel, the default (all cores) causes severe CPU contention on macOS.
    _n_threads = int(os.environ.get("TORCH_THREADS", "2"))
    try:
        import torch
        torch.set_num_threads(_n_threads)
    except ImportError:
        pass
    os.environ.setdefault("OMP_NUM_THREADS",  str(_n_threads))
    os.environ.setdefault("MKL_NUM_THREADS",  str(_n_threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(_n_threads))

    _log_dir = "/tmp/qrouting_logs"
    os.makedirs(_log_dir, exist_ok=True)
    _req_count = algo.topo.numOfRequestPerRound
    _csv_path  = os.path.join(_log_dir, f"progress_{algo.name}_req{_req_count}.csv")

    with open(_csv_path, "w", buffering=1) as _csv:
        _csv.write("timeslot,successful_requests,reward,wall_ms\n")
        _wall_samples = []
        for i in range(ttime):
            _t0    = time.perf_counter()
            result = algo.work(requests[i], i)
            _wall_ms = (time.perf_counter() - _t0) * 1000.0
            _wall_samples.append(_wall_ms)
            _succ = result.successfulRequestPerRound[i] \
                    if i < len(result.successfulRequestPerRound) else 0
            _rew  = result.rewardPerRound[i] \
                    if i < len(result.rewardPerRound) else 0
            _csv.write(f"{i},{_succ},{_rew},{_wall_ms:.1f}\n")

    _attempted = sum(len(requests.get(i, [])) for i in range(ttime))
    _succ_series = result.successfulRequestPerRound[:ttime]
    _success = sum(_succ_series)
    _fid_series = [f for f in result.fidelityPerRound[:ttime] if f > 0]
    _half = max(ttime // 2, 1)
    _first_attempted = sum(len(requests.get(i, [])) for i in range(_half))
    _second_attempted = max(_attempted - _first_attempted, 0)
    _first_success = sum(_succ_series[:_half])
    _second_success = sum(_succ_series[_half:])
    _summary_path = os.path.join(_log_dir, f"summary_{algo.name}_req{_req_count}.csv")
    with open(_summary_path, "w", buffering=1) as _sum:
        _sum.write(
            "algo,load,total_requests,timeslots,successful,success_rate,"
            "mean_fidelity,mean_wall_ms,total_wall_s,"
            "first_half_success_rate,second_half_success_rate\n")
        _sum.write(
            f"{algo.name},{_req_count},{_attempted},{ttime},{_success},"
            f"{(_success / max(_attempted, 1)):.6f},"
            f"{(sum(_fid_series) / max(len(_fid_series), 1)):.6f},"
            f"{(sum(_wall_samples) / max(len(_wall_samples), 1)):.3f},"
            f"{(sum(_wall_samples) / 1000.0):.3f},"
            f"{(_first_success / max(_first_attempted, 1)):.6f},"
            f"{(_second_success / max(_second_attempted, 1)):.6f}\n")

    # Save trained weights to disk after training run
    if not INFERENCE_MODE and hasattr(algo, '_save_weights'):
        algo._save_weights()

    resultDict[pid] = result

    if executor is not None:
        executor.shutdown(wait=True)

    success_req = sum(result.successfulRequestPerRound[:ttime])
    _wall = time.time() - _t_start
    print(f"{'=' * 52}")
    print(f"  pid={pid}  algo={algo.name}  success={success_req}  wall={_wall:.1f}s")
    print(f"{'=' * 52}")


# ── Main Run function ─────────────────────────────────────────────────────────

def Run(numOfRequestPerRound=20, numOfNode=0, r=7, q=SWAP_PROB,
        alpha=alpha_, SocialNetworkDensity=0.5,
        rtime=ttime, topo=None, FixedRequests=None, results=[]):

    if topo is None:
        topo = Topo.generate(numOfNode, q, 5, alpha, 6, int(math.sqrt(numOfNode)))
    numOfNode = len(topo.nodes)
    topo.setQ(q)
    topo.setAlpha(alpha)
    topo.setNumOfRequestPerRound(numOfRequestPerRound)

    algorithms = [
        QuRA_DQRL_DIST(copy.deepcopy(topo),  name='QuRA_Seq_DIST'),
        QuRA_Flock_DIST(copy.deepcopy(topo),  name='QuRA_Flock_DIST'),
        QuRA_Guard_DIST(copy.deepcopy(topo),  name='QuRA_Guard_DIST'),
        QuRA_Hive_DIST(copy.deepcopy(topo),   name='QuRA_Hive_DIST'),
        RELiQ_Adapter(copy.deepcopy(topo),    name='RELiQ'),
        EBSPA(copy.deepcopy(topo),            name='EBSPA'),
    ]
    if RUN_ALGOS:
        algorithms = [a for a in algorithms if a.name in RUN_ALGOS]
        if not algorithms:
            raise ValueError(f"RUN_ALGOS matched no algorithms: {sorted(RUN_ALGOS)}")

    gc.collect()
    print(f"  algorithms: {[a.name for a in algorithms]}")

    global times
    results      = [[] for _ in range(len(algorithms))]
    rtime        = ttime
    if TOTAL_REQUESTS > 0 and FixedRequests is None:
        rtime = int(math.ceil(TOTAL_REQUESTS / max(numOfRequestPerRound, 1)))
        print(f"  fixed-budget: total_requests={TOTAL_REQUESTS}  timeslots={rtime}")
    resultDicts  = [multiprocessing.Manager().dict() for _ in algorithms]
    shared_data  = multiprocessing.Manager().dict()

    for algo in algorithms:
        key = algo.name + str(len(algo.topo.nodes)) + str(algo.topo.alpha) + str(algo.topo.q) + 'max_success'
        shared_data[key] = 0

    bias_weights = [x % 10 == 0 for x in range(numOfNode)]
    prob = np.array(bias_weights) / np.sum(bias_weights)

    jobs = []
    pid  = 0

    for _ in range(times):
        ids = {i: [] for i in range(rtime)}
        if FixedRequests is not None:
            ids = FixedRequests
        else:
            requests_remaining = TOTAL_REQUESTS if TOTAL_REQUESTS > 0 else None
            for i in range(rtime):
                if i < rtime:
                    # Curriculum: randomly pick a load this timeslot so the
                    # model sees all traffic levels during training.
                    # In inference mode INFERENCE_MODE is True and
                    # numOfRequestPerRound is fixed to the evaluation load.
                    if requests_remaining is not None:
                        if requests_remaining <= 0:
                            slot_load = 0
                        else:
                            slot_load = min(numOfRequestPerRound, requests_remaining)
                            requests_remaining -= slot_load
                    elif INFERENCE_MODE:
                        slot_load = numOfRequestPerRound
                    else:
                        slot_load = _random.randint(TRAIN_LOAD_MIN,
                                                    numOfRequestPerRound)
                    for _ in range(slot_load):
                        while True:
                            a = sample(list(range(numOfNode)), 2)
                            if (a[0], a[1]) not in ids[i]:
                                break
                        ids[i].append((a[0], a[1]))

        for algoIndex in range(len(algorithms)):
            algo = copy.deepcopy(algorithms[algoIndex])
            requests = {i: [] for i in range(rtime)}
            for i in range(rtime):
                for (src, dst) in ids[i]:
                    requests[i].append((algo.topo.nodes[src], algo.topo.nodes[dst]))
            pid += 1
            job = multiprocessing.Process(
                target=runThread,
                args=(algo, requests, algoIndex, rtime, pid,
                      resultDicts[algoIndex], shared_data))
            jobs.append(job)

    for job in jobs:
        job.start()
    for job in jobs:
        job.join()

    for algoIndex in range(len(algorithms)):
        results[algoIndex] = AlgorithmResult.Avg(
            resultDicts[algoIndex].values(),
            numOfRequestPerRound,
            algorithms[0].topo,
        )

    return results


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("[Run.py] starting")
    t1 = time.time()

    topo = Topo.generate(nodeNo, SWAP_PROB, 5, alpha_, degree, gridSize=gridSize)

    for load in numOfRequestPerRound:
        print(f"\n[Run.py] load={load}")
        Run(numOfRequestPerRound=load, topo=copy.deepcopy(topo), rtime=ttime)

    t2 = time.time()
    print(f"\n[Run.py] done  elapsed={(t2 - t1) / 3600:.2f}h")
