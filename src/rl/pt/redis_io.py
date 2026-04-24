"""
Redis I/O helpers for PyTorch model weights.
Mirrors the pickle-based interface in dist_agent_helper.py so that
the same Redis key schema works unchanged.
"""
import gzip
import io
import os
import pickle

import redis
import torch

REDIS_DB         = int(os.environ.get("REDIS_DB",         "0"))
REDIS_HOST       = os.environ.get("REDIS_HOST",           "localhost")
REDIS_PORT       = int(os.environ.get("REDIS_PORT",       "6379"))
MODEL_SAVE_PATH  = os.environ.get("MODEL_SAVE_PATH",
                                  "/tmp/qrouting_model/trained_weights_pt.pkl")


def get_redis() -> redis.Redis:
    return redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)


def save_model_to_redis(model: torch.nn.Module, model_name: str = "dqrl_model",
                        r: redis.Redis = None) -> int | None:
    if r is None:
        r = get_redis()
    try:
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        serialized = buf.getvalue()
        r.set(f"{model_name}_weights", serialized, ex=86400)
        version = r.incr(f"{model_name}_version")
        print(f"[redis_io] saved {model_name} version {version}")
        return version
    except Exception as e:
        print(f"[redis_io] save error: {e}")
        return None


def load_model_from_redis(model: torch.nn.Module, model_name: str = "dqrl_model",
                          r: redis.Redis = None) -> int | None:
    if r is None:
        r = get_redis()
    try:
        raw = r.get(f"{model_name}_weights")
        if raw is None:
            return None
        buf = io.BytesIO(raw)
        state = torch.load(buf, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        version = int(r.get(f"{model_name}_version") or 0)
        print(f"[redis_io] loaded {model_name} version {version}")
        return version
    except Exception as e:
        print(f"[redis_io] load error: {e}")
        return None


def save_worker_model_to_redis(model: torch.nn.Module, worker_id: int,
                               base_name: str = "dqrl_model",
                               r: redis.Redis = None) -> int | None:
    return save_model_to_redis(model, f"{base_name}_worker{worker_id}", r)


def save_model_to_disk(model: torch.nn.Module,
                       path: str = MODEL_SAVE_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    with gzip.open(path, "wb") as f:
        f.write(buf.getvalue())
    print(f"[redis_io] saved weights → {path}")


def load_model_from_disk(model: torch.nn.Module,
                         path: str = MODEL_SAVE_PATH) -> bool:
    if not os.path.exists(path):
        print(f"[redis_io] no file at {path}")
        return False
    try:
        with gzip.open(path, "rb") as f:
            raw = f.read()
        buf = io.BytesIO(raw)
        state = torch.load(buf, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        print(f"[redis_io] loaded weights from {path}")
        return True
    except Exception as e:
        print(f"[redis_io] disk load error: {e}")
        return False


# ── FedAvg (inline, no separate fedavg.py needed) ───────────────────────────
def fedavg_aggregate(num_workers: int = 4,
                     global_name: str = "dqrl_model",
                     base_name:   str = "dqrl_model",
                     r: redis.Redis = None) -> int | None:
    if r is None:
        r = get_redis()
    all_states = []
    for wid in range(num_workers):
        raw = r.get(f"{base_name}_worker{wid}_weights")
        if raw:
            buf = io.BytesIO(raw)
            all_states.append(torch.load(buf, map_location="cpu", weights_only=True))

    if len(all_states) < 2:
        print(f"[FedAvg] only {len(all_states)} worker(s) ready — skipping")
        return None

    import torch as _t
    avg = {}
    keys = list(all_states[0].keys())
    for k in keys:
        stacked = _t.stack([s[k].float() for s in all_states], dim=0)
        avg[k] = stacked.mean(dim=0)

    buf = io.BytesIO()
    _t.save(avg, buf)
    r.set(f"{global_name}_weights", buf.getvalue(), ex=86400)
    version = r.incr(f"{global_name}_version")
    print(f"[FedAvg] aggregated {len(all_states)} workers → global version {version}")
    return version
