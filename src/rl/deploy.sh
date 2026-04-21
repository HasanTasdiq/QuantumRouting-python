#!/usr/bin/env bash
# =============================================================================
# deploy.sh — Launch (or restart) the full QuRA multi-worker training stack.
#
# Usage
# -----
#   bash deploy.sh          # start / restart everything
#   bash deploy.sh stop     # graceful stop
#   bash deploy.sh status   # show running services
#   bash deploy.sh logs     # tail all logs
#
# Services managed
# ----------------
#   Redis          (started if not already running)
#   Training workers  WORKER_ID=0..3  ports 8000-8003
#   Predict server    port 8080
#   FedAvg aggregator (background daemon)
#   Run.py            main experiment (started last)
#
# Linux 64-core tuning
# --------------------
#   Each training worker gets OMP/TF_NUM_INTRAOP = CORES_PER_WORKER cores.
#   Predict server gets its own thread budget.
#   Run.py routing executor uses remaining cores.
# =============================================================================

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/../../quantum/algorithm" && pwd)"
LOG_DIR="/tmp/qrouting_logs"
PID_DIR="${LOG_DIR}/pids"
PYTHON="${PYTHON:-python3}"
MODE="${1:-start}"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

# ── Config ────────────────────────────────────────────────────────────────────
NUM_WORKERS=4
AGG_INTERVAL=30          # FedAvg every 30 s
REDIS_PORT=6379

# CPU tuning — divide 64 cores across services
TOTAL_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
CORES_PER_WORKER=$(( TOTAL_CORES / (NUM_WORKERS + 2) ))   # +2 for predict + slack
CORES_PER_WORKER=$(( CORES_PER_WORKER < 2 ? 2 : CORES_PER_WORKER ))
PREDICT_CORES=$(( CORES_PER_WORKER * 2 ))   # predict server gets 2× share

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Stop helpers ──────────────────────────────────────────────────────────────
kill_by_pid_file() {
    local f="$1"
    if [[ -f "$f" ]]; then
        local pid; pid=$(<"$f")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && log "Killed PID $pid ($f)"
        fi
        rm -f "$f"
    fi
}

kill_by_port() {
    local port="$1"
    # try fuser first (Linux), fall back to lsof (macOS)
    local pids
    if command -v fuser &>/dev/null; then
        pids=$(fuser "${port}/tcp" 2>/dev/null | tr ' ' '\n' | grep -v '^$' || true)
    else
        pids=$(lsof -ti tcp:"${port}" 2>/dev/null || true)
    fi
    for pid in $pids; do
        kill "$pid" 2>/dev/null && log "Killed port ${port} owner PID $pid"
    done
}

kill_by_name() {
    pkill -f "$1" 2>/dev/null && log "Killed processes matching: $1" || true
}

stop_all() {
    log "Stopping all QuRA services..."
    kill_by_pid_file "${PID_DIR}/run.pid"
    kill_by_pid_file "${PID_DIR}/aggregator.pid"
    kill_by_pid_file "${PID_DIR}/predict.pid"
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        kill_by_pid_file "${PID_DIR}/worker${wid}.pid"
    done
    # Belt-and-suspenders: kill by process name too
    kill_by_name "dist_agent_aggregator.py"
    kill_by_name "dist_agent_predict.py"
    kill_by_name "dist_agent.py"
    kill_by_name "Run.py"
    # Force-free ports
    for port in 8000 8001 8002 8003 8080; do
        kill_by_port "$port"
    done
    sleep 2
    log "All services stopped."
}

status_all() {
    echo ""
    echo "══════════════════ QuRA Service Status ══════════════════"
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        local f="${PID_DIR}/worker${wid}.pid"
        local port=$(( 8000 + wid ))
        if [[ -f "$f" ]] && kill -0 "$(<"$f")" 2>/dev/null; then
            echo "  Worker ${wid}    (port ${port})  PID=$(<"$f")  ✓ running"
        else
            echo "  Worker ${wid}    (port ${port})  ✗ not running"
        fi
    done
    local pf="${PID_DIR}/predict.pid"
    if [[ -f "$pf" ]] && kill -0 "$(<"$pf")" 2>/dev/null; then
        echo "  Predict        (port 8080)  PID=$(<"$pf")  ✓ running"
    else
        echo "  Predict        (port 8080)  ✗ not running"
    fi
    local af="${PID_DIR}/aggregator.pid"
    if [[ -f "$af" ]] && kill -0 "$(<"$af")" 2>/dev/null; then
        echo "  Aggregator                  PID=$(<"$af")  ✓ running"
    else
        echo "  Aggregator                  ✗ not running"
    fi
    local rf="${PID_DIR}/run.pid"
    if [[ -f "$rf" ]] && kill -0 "$(<"$rf")" 2>/dev/null; then
        echo "  Run.py                      PID=$(<"$rf")  ✓ running"
    else
        echo "  Run.py                      ✗ not running"
    fi
    echo "  Logs: ${LOG_DIR}/"
    echo "═════════════════════════════════════════════════════════"
    echo ""
}

tail_logs() {
    tail -f \
        "${LOG_DIR}/worker0.log" \
        "${LOG_DIR}/worker1.log" \
        "${LOG_DIR}/worker2.log" \
        "${LOG_DIR}/worker3.log" \
        "${LOG_DIR}/predict.log" \
        "${LOG_DIR}/aggregator.log" \
        "${LOG_DIR}/run.log" 2>/dev/null
}

[[ "${MODE}" == "stop"   ]] && { stop_all;    exit 0; }
[[ "${MODE}" == "status" ]] && { status_all;  exit 0; }
[[ "${MODE}" == "logs"   ]] && { tail_logs;   exit 0; }

# ── START ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  QuRA Paper Run — ${NUM_WORKERS} workers — ${TOTAL_CORES} cores detected"
echo "  Cores per worker: ${CORES_PER_WORKER}   Predict cores: ${PREDICT_CORES}"
echo "═══════════════════════════════════════════════════════════"

# Always stop first — clear any stale state
stop_all

# ── Redis ─────────────────────────────────────────────────────────────────────
if redis-cli -p "${REDIS_PORT}" ping &>/dev/null; then
    log "Redis already running on port ${REDIS_PORT}"
else
    log "Starting Redis on port ${REDIS_PORT}..."
    redis-server --port "${REDIS_PORT}" --daemonize yes \
        --logfile "${LOG_DIR}/redis.log" \
        --maxmemory 50gb \
        --maxmemory-policy allkeys-lru
    sleep 1
    if redis-cli -p "${REDIS_PORT}" ping &>/dev/null; then
        log "Redis started OK"
    else
        echo "ERROR: Redis failed to start. Check ${LOG_DIR}/redis.log"
        exit 1
    fi
fi

# Flush stale model keys so FedAvg starts fresh
log "Flushing stale model keys from Redis..."
redis-cli -p "${REDIS_PORT}" --scan --pattern 'dqrl_model*' \
    | xargs -r redis-cli -p "${REDIS_PORT}" del > /dev/null 2>&1 || true

cd "${SCRIPT_DIR}"

# ── Training workers ──────────────────────────────────────────────────────────
for (( wid=0; wid<NUM_WORKERS; wid++ )); do
    PORT=$(( 8000 + wid ))
    LOG="${LOG_DIR}/worker${wid}.log"
    > "${LOG}"   # truncate log
    log "Starting training worker ${wid}  port=${PORT}"
    WORKER_ID=${wid} \
    OMP_NUM_THREADS=${CORES_PER_WORKER} \
    TF_NUM_INTRAOP_THREADS=${CORES_PER_WORKER} \
    TF_NUM_INTEROP_THREADS=2 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    nohup "${PYTHON}" -u dist_agent.py > "${LOG}" 2>&1 &
    echo $! > "${PID_DIR}/worker${wid}.pid"
done

# Wait for all workers to be ready
log "Waiting for training workers..."
for (( wid=0; wid<NUM_WORKERS; wid++ )); do
    PORT=$(( 8000 + wid ))
    LOG="${LOG_DIR}/worker${wid}.log"
    local_timeout=120
    elapsed=0
    until grep -q "Application startup complete" "${LOG}" 2>/dev/null; do
        sleep 2; elapsed=$(( elapsed + 2 ))
        if (( elapsed >= local_timeout )); then
            echo "ERROR: Worker ${wid} did not start within ${local_timeout}s"
            echo "Last log lines:"
            tail -20 "${LOG}"
            exit 1
        fi
    done
    log "  Worker ${wid} ready on port ${PORT}"
done

# ── Predict server ────────────────────────────────────────────────────────────
PRED_LOG="${LOG_DIR}/predict.log"
> "${PRED_LOG}"
log "Starting predict server  port=8080"
OMP_NUM_THREADS=${PREDICT_CORES} \
TF_NUM_INTRAOP_THREADS=${PREDICT_CORES} \
TF_NUM_INTEROP_THREADS=2 \
TF_CPP_MIN_LOG_LEVEL=2 \
nohup "${PYTHON}" -u dist_agent_predict.py > "${PRED_LOG}" 2>&1 &
echo $! > "${PID_DIR}/predict.pid"

local_timeout=120; elapsed=0
until grep -q "Application startup complete" "${PRED_LOG}" 2>/dev/null; do
    sleep 2; elapsed=$(( elapsed + 2 ))
    if (( elapsed >= local_timeout )); then
        echo "ERROR: Predict server did not start within ${local_timeout}s"
        tail -20 "${PRED_LOG}"
        exit 1
    fi
done
log "  Predict server ready on port 8080"

# ── FedAvg aggregator ─────────────────────────────────────────────────────────
AGG_LOG="${LOG_DIR}/aggregator.log"
> "${AGG_LOG}"
log "Starting FedAvg aggregator  every=${AGG_INTERVAL}s"
NUM_WORKERS=${NUM_WORKERS} \
AGGREGATION_EVERY_S=${AGG_INTERVAL} \
TF_CPP_MIN_LOG_LEVEL=2 \
nohup "${PYTHON}" -u dist_agent_aggregator.py > "${AGG_LOG}" 2>&1 &
echo $! > "${PID_DIR}/aggregator.pid"
sleep 1

# ── Run.py ────────────────────────────────────────────────────────────────────
RUN_LOG="${LOG_DIR}/run.log"
> "${RUN_LOG}"
log "Starting Run.py (paper experiment)"
cd "${ALGO_DIR}"
OMP_NUM_THREADS=4 \
TF_CPP_MIN_LOG_LEVEL=2 \
nohup "${PYTHON}" -u Run.py > "${RUN_LOG}" 2>&1 &
echo $! > "${PID_DIR}/run.pid"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  All services launched successfully."
echo ""
echo "  Monitor progress:"
echo "    tail -f ${LOG_DIR}/run.log"
echo "    tail -f ${LOG_DIR}/worker0.log"
echo "    tail -f ${LOG_DIR}/aggregator.log"
echo ""
echo "  Check status:  bash deploy.sh status"
echo "  Stop all:      bash deploy.sh stop"
echo "  Tail all logs: bash deploy.sh logs"
echo "═══════════════════════════════════════════════════════════"
echo ""
