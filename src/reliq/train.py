"""
Standalone RELiQ trainer with QuRA physics baked in.

Usage (from QuantumRouting-python/):
    python -m src.reliq.train [--total-steps N] [--output-dir DIR]

Defaults match the RELiQ paper config (CPU, Werner physics, lifetime=10).
Reduce --total-steps for a quick smoke run (e.g. 50_000).
Trained model saved to <output-dir>/RELiQ_QuRAPhysics/model.pt
"""
import os
import sys

# Ensure src/reliq/ is importable as a package
_this_dir = os.path.dirname(os.path.abspath(__file__))
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)
_src_dir = os.path.dirname(_this_dir)
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

import subprocess, shlex

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--total-steps",  type=int, default=500_000)
    p.add_argument("--output-dir",   type=str, default="runs_quantum")
    p.add_argument("--device",       type=str, default="cpu")
    p.add_argument("--comment",      type=str, default="RELiQ_QuRAPhysics")
    args = p.parse_args()

    # NetMon (graph message-passing) is intentionally disabled because the
    # QuRA adapter at inference time does not have access to the per-node
    # graph observations / RNN state propagation that NetMon trains against.
    # A plain DQN over per-request observations gives an inference-compatible
    # checkpoint and matches the obs format the adapter constructs.
    #
    # `--neighbors=6` matches RELiQ_Adapter._NEIGHBOR_COUNT so the per-agent
    # observation dim (`MAX_REQ + 3 + neighbors*9 = 157`) and action space size
    # (`neighbors = 6`, no idle with --no-idle-action) line up with inference.
    step_before = max(1000, min(10000, args.total_steps // 10))
    step_between = max(50, args.total_steps // 250)
    cmd = [
        sys.executable, "-u",
        os.path.join(_this_dir, "main.py"),
        "--use-future-rewards",
        "--no-idle-action",
        "--request-based-observation",
        "--fixed-requests",
        "--action-mask",
        "--disable-progressbar",
        f"--total-steps={args.total_steps}",
        f"--step-between-train={step_between}",
        f"--step-before-train={step_before}",
        "--neighbors=6",
        "--model=dqn",
        f"--device={args.device}",
        f"--capacity={min(50000, max(2000, args.total_steps // 4))}",
        "--min-path-length=1",
        f"--output-dir={args.output_dir}",
        f"--comment={args.comment}",
    ]

    print("[reliq/train.py] launching:", " ".join(shlex.quote(c) for c in cmd))
    result = subprocess.run(cmd, cwd=_this_dir)
    if result.returncode != 0:
        print(f"[reliq/train.py] ERROR: main.py exited with code {result.returncode}")
        sys.exit(result.returncode)

    # Locate the checkpoint and copy to a stable path
    import glob, shutil
    pattern = os.path.join(args.output_dir, f"*{args.comment}*", "model.pt")
    matches = sorted(glob.glob(pattern))
    if matches:
        stable = os.path.join(args.output_dir, "RELiQ_QuRAPhysics", "model.pt")
        os.makedirs(os.path.dirname(stable), exist_ok=True)
        if matches[-1] != stable:
            shutil.copy2(matches[-1], stable)
        print(f"[reliq/train.py] model saved → {stable}")
    else:
        print(f"[reliq/train.py] WARNING: no model.pt found under {pattern}")

if __name__ == "__main__":
    main()
