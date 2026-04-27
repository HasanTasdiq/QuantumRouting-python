"""
Unified CSV metrics for all QuRA-v2 algorithms.

Schema per row:
  timeslot, algo, load, requests_in, served, served_F_geq_Fmin,
  mean_F_delivered, conflicts, wall_ms

F_min gate is applied identically to all algorithms (QuRA variants, RELiQ,
EBSPA, ShortestPath) so throughput numbers are comparable.
"""
from __future__ import annotations
import csv
import os
import time
from dataclasses import dataclass, field


@dataclass
class SlotMetrics:
    timeslot:         int
    algo:             str
    load:             int
    requests_in:      int   = 0
    served:           int   = 0       # reached dst (any fidelity)
    served_F_geq_Fmin: int  = 0       # reached dst with F >= F_min
    mean_F_delivered: float = 0.0     # mean Werner fidelity of successful deliveries
    conflicts:        int   = 0       # link assignments refused due to capacity
    wall_ms:          float = 0.0


class MetricsWriter:
    """
    Writes one CSV per algorithm per load level.
    Thread-safe: each process writes its own file (no shared file handle).

    log_dir defaults to /tmp/qrouting_logs/v2
    """

    _HEADER = [
        "timeslot", "algo", "load",
        "requests_in", "served", "served_F_geq_Fmin",
        "mean_F_delivered", "conflicts", "wall_ms",
    ]

    def __init__(self, algo: str, load: int,
                 log_dir: str = "/tmp/qrouting_logs/v2"):
        os.makedirs(log_dir, exist_ok=True)
        safe_name = algo.replace(" ", "_").replace("/", "_")
        path = os.path.join(log_dir, f"{safe_name}_req{load}.csv")
        self._fh  = open(path, "w", newline="", buffering=1)
        self._csv = csv.DictWriter(self._fh, fieldnames=self._HEADER)
        self._csv.writeheader()

    def write(self, m: SlotMetrics) -> None:
        self._csv.writerow({
            "timeslot":          m.timeslot,
            "algo":              m.algo,
            "load":              m.load,
            "requests_in":       m.requests_in,
            "served":            m.served,
            "served_F_geq_Fmin": m.served_F_geq_Fmin,
            "mean_F_delivered":  f"{m.mean_F_delivered:.4f}",
            "conflicts":         m.conflicts,
            "wall_ms":           f"{m.wall_ms:.2f}",
        })

    def close(self) -> None:
        self._fh.close()

    def __del__(self):
        try:
            self._fh.close()
        except Exception:
            pass
