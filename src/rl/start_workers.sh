#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# start_workers.sh  —  launch the full multi-worker training stack
#
# Services started
# ────────────────
#  Training workers   WORKER_ID=0 → port 8000
#                     WORKER_ID=1 → port 8001
#                     WORKER_ID=2 → port 8002
#                     WORKER_ID=3 → port 8003
#  Predict server     port 8080  (reads global FedAvg model from Redis)
#  FedAvg aggregator  background daemon (aggregates every 30 s)
#
# Usage
# ─────
#  Single machine (smoke test):   bash start_workers.sh smoke
#  Full paper run:                bash start_workers.sh paper
#
# Stop everything:   bash start_workers.sh stop
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="/tmp/qrouting_logs"
mkdir -p "$LOG_DIR"
PYTHON="${PYTHON:-python3}"
MODE="${1:-smoke}"

stop_all() {
    echo "[stop] Killing all worker, predict, and aggregator processes..."
    pkill -f "dist_agent.py"     2>/dev/null || true
    pkill -f "dist_agent_predict.py" 2>/dev/null || true
    pkill -f "dist_agent_aggregator.py" 2>/dev/null || true
    sleep 1
    echo "[stop] Done."
    exit 0
}

[[ "${MODE}" == "stop" ]] && stop_all

# ── Number of workers ────────────────────────────────────────────────────────
if [[ "${MODE}" == "smoke" ]]; then
    NUM_WORKERS=2          # smoke: 2 workers (Seq + Flock)
    AGG_INTERVAL=20        # aggregate every 20 s
else
    NUM_WORKERS=4          # paper: one per algorithm variant
    AGG_INTERVAL=30
fi

echo "═══════════════════════════════════════════════════"
echo "  QuRA Multi-Worker Stack  mode=${MODE}  workers=${NUM_WORKERS}"
echo "═══════════════════════════════════════════════════"

cd "$SCRIPT_DIR"

# ── Start training workers ───────────────────────────────────────────────────
for (( wid=0; wid<NUM_WORKERS; wid++ )); do
    PORT=$(( 8000 + wid ))
    LOG="${LOG_DIR}/worker${wid}.log"
    echo "[start] Training worker ${wid}  port=${PORT}  log=${LOG}"
    WORKER_ID=${wid} nohup "$PYTHON" dist_agent.py > "$LOG" 2>&1 &
    echo $! > "${LOG_DIR}/worker${wid}.pid"
done

# Wait for workers to be ready
echo "[wait]  Waiting for training workers to start..."
for (( wid=0; wid<NUM_WORKERS; wid++ )); do
    PORT=$(( 8000 + wid ))
    LOG="${LOG_DIR}/worker${wid}.log"
    until grep -q "Application startup complete" "$LOG" 2>/dev/null; do sleep 1; done
    echo "        worker ${wid} ready on port ${PORT}"
done

# ── Start predict server ─────────────────────────────────────────────────────
PRED_LOG="${LOG_DIR}/predict.log"
echo "[start] Predict server  port=8080  log=${PRED_LOG}"
nohup "$PYTHON" dist_agent_predict.py > "$PRED_LOG" 2>&1 &
echo $! > "${LOG_DIR}/predict.pid"
until grep -q "Application startup complete" "$PRED_LOG" 2>/dev/null; do sleep 1; done
echo "        Predict server ready on port 8080"

# ── Start FedAvg aggregator ──────────────────────────────────────────────────
AGG_LOG="${LOG_DIR}/aggregator.log"
echo "[start] FedAvg aggregator  every=${AGG_INTERVAL}s  log=${AGG_LOG}"
NUM_WORKERS=${NUM_WORKERS} AGGREGATION_EVERY_S=${AGG_INTERVAL} \
    nohup "$PYTHON" dist_agent_aggregator.py > "$AGG_LOG" 2>&1 &
echo $! > "${LOG_DIR}/aggregator.pid"

echo ""
echo "All services running.  To stop: bash start_workers.sh stop"
echo "Logs: ${LOG_DIR}/"
