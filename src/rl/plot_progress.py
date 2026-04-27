"""
plot_progress.py — visualise QuRA-v2 training and inference results.

Reads /tmp/qrouting_logs/progress_<ALGO>_req<LOAD>.csv and plots:
  1. Per-timeslot successful requests with a rolling average (one panel per load)
  2. Summary bar chart: total success per algorithm per load

Usage
-----
  python plot_progress.py                        # auto-detect all CSVs
  python plot_progress.py --load 100             # single load
  python plot_progress.py --log-dir /tmp/...     # custom log dir
  python plot_progress.py --save results.png     # save instead of show
"""
from __future__ import annotations
import argparse
import glob
import os
import re
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg" if "DISPLAY" not in os.environ and sys.platform != "darwin" else "MacOSX")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ── colour palette (colour-blind friendly) ───────────────────────────────────
_COLORS = {
    "QuRA_Seq_DIST":   "#1f77b4",
    "QuRA_Flock_DIST": "#ff7f0e",
    "QuRA_Guard_DIST": "#2ca02c",
    "QuRA_Hive_DIST":  "#d62728",
    "RELiQ":           "#9467bd",
    "EBSPA":           "#8c564b",
    "ShortestPath":    "#7f7f7f",
}
_ALGO_ORDER = list(_COLORS.keys())


def _rolling(arr: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return arr.copy()
    out = np.convolve(arr, np.ones(w) / w, mode="same")
    # fix edge artefacts: use cumsum for leading window
    cs = np.cumsum(arr)
    for i in range(min(w - 1, len(arr))):
        out[i] = cs[i] / (i + 1)
    return out


def load_csv(path: str) -> np.ndarray:
    """Returns (T, 3) array: [timeslot, successful_requests, wall_ms]."""
    rows = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if len(parts) < 3:
                continue
            try:
                rows.append([float(parts[0]), float(parts[1]), float(parts[3]) if len(parts) > 3 else 0.0])
            except ValueError:
                continue
    return np.array(rows) if rows else np.zeros((0, 3))


def discover_csvs(log_dir: str, load_filter: int | None) -> dict[int, dict[str, str]]:
    """Returns {load: {algo_name: csv_path}}."""
    pattern = os.path.join(log_dir, "progress_*_req*.csv")
    result: dict[int, dict[str, str]] = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"progress_(.+)_req(\d+)\.csv$", os.path.basename(path))
        if not m:
            continue
        algo, load = m.group(1), int(m.group(2))
        if load_filter is not None and load != load_filter:
            continue
        result.setdefault(load, {})[algo] = path
    return result


def print_summary(by_load: dict[int, dict[str, str]]) -> None:
    for load in sorted(by_load):
        algos = by_load[load]
        print(f"\n{'─'*60}")
        print(f"  Load = {load} requests/timeslot")
        print(f"{'─'*60}")
        print(f"  {'Algorithm':<22} {'Total succ':>10}  {'Mean/slot':>9}  {'Mean wall ms':>12}")
        for algo in _ALGO_ORDER:
            if algo not in algos:
                continue
            data = load_csv(algos[algo])
            if len(data) == 0:
                continue
            total = int(data[:, 1].sum())
            mean  = data[:, 1].mean()
            wall  = data[:, 2].mean()
            print(f"  {algo:<22} {total:>10}  {mean:>9.2f}  {wall:>12.1f}ms")


def plot_all(by_load: dict[int, dict[str, str]],
             save_path: str | None, window: int) -> None:
    loads = sorted(by_load)
    n     = len(loads)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), squeeze=False)
    fig.suptitle("QuRA-v2: Successful requests per timeslot", fontsize=14, y=1.01)

    for row, load in enumerate(loads):
        ax   = axes[row][0]
        algos = by_load[load]
        ax.set_title(f"Load = {load} req/slot", fontsize=11)

        for algo in _ALGO_ORDER:
            if algo not in algos:
                continue
            data = load_csv(algos[algo])
            if len(data) == 0:
                continue
            ts   = data[:, 0]
            succ = data[:, 1]
            col  = _COLORS.get(algo, "#333333")
            w    = min(window, max(1, len(succ) // 10))
            roll = _rolling(succ, w)

            ax.plot(ts, succ, alpha=0.18, color=col, linewidth=0.7)
            ax.plot(ts, roll, color=col, linewidth=2.0,
                    label=f"{algo}  (avg={succ.mean():.2f}/slot)")

        ax.set_xlabel("Timeslot")
        ax.set_ylabel("Successful requests")
        ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(loc="upper left", fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"\n[plot_progress] figure saved → {save_path}")
    else:
        plt.show()


def plot_bar(by_load: dict[int, dict[str, str]], save_path: str | None) -> None:
    """Bar chart: total successful requests per algo per load."""
    loads = sorted(by_load)
    algos = [a for a in _ALGO_ORDER if any(a in by_load[l] for l in loads)]
    x     = np.arange(len(loads))
    width = 0.8 / max(len(algos), 1)

    fig, ax = plt.subplots(figsize=(max(8, len(loads) * 2), 5))
    ax.set_title("Total successful requests by algorithm and load", fontsize=12)

    for i, algo in enumerate(algos):
        totals = []
        for load in loads:
            path = by_load[load].get(algo)
            if path:
                data = load_csv(path)
                totals.append(data[:, 1].sum() if len(data) else 0)
            else:
                totals.append(0)
        offset = (i - len(algos) / 2 + 0.5) * width
        bars = ax.bar(x + offset, totals, width * 0.9,
                      label=algo, color=_COLORS.get(algo, "#333333"), alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f"load={l}" for l in loads])
    ax.set_ylabel("Total successful requests")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    bar_path = (save_path.replace(".png", "_bar.png")
                if save_path else None)
    if bar_path:
        plt.savefig(bar_path, dpi=150, bbox_inches="tight")
        print(f"[plot_progress] bar chart saved → {bar_path}")
    else:
        plt.show()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir",  default="/tmp/qrouting_logs",
                    help="Directory with progress_*.csv files")
    ap.add_argument("--load",     type=int, default=None,
                    help="Only plot this request load (default: all)")
    ap.add_argument("--window",   type=int, default=500,
                    help="Rolling-average window (default 500; auto-scaled if too large)")
    ap.add_argument("--save",     default=None,
                    help="Save figure to this path instead of showing (e.g. out.png)")
    ap.add_argument("--no-plot",  action="store_true",
                    help="Print summary table only, no figure")
    args = ap.parse_args()

    by_load = discover_csvs(args.log_dir, args.load)
    if not by_load:
        print(f"[plot_progress] No CSVs found in {args.log_dir}")
        sys.exit(1)

    print_summary(by_load)

    if args.no_plot or not HAS_MPL:
        if not HAS_MPL:
            print("\n[plot_progress] matplotlib not installed — summary only.")
        return

    plot_all(by_load, args.save, args.window)
    plot_bar(by_load, args.save)


if __name__ == "__main__":
    main()
