#!/usr/bin/env bash
# =============================================================================
# deploy_full.sh — Full paper run: train one model, then evaluate on all loads.
#
# Two-phase pipeline
# ------------------
#  Phase 1 — TRAINING  (single request load = 100, 10000 timeslots)
#    4 training workers + FedAvg aggregator + predict server + Run.py
#    Model checkpointed to Redis + /tmp/qrouting_model/trained_weights.pkl
#    Expected duration: ~36 hours on 64-core CPU server
#
#  Phase 2 — INFERENCE  (all 6 loads in parallel, frozen model, epsilon=0)
#    1 predict server (INFERENCE_MODE=1) + 6 concurrent Run.py processes
#    Each Run.py handles one req load: 5, 10, 25, 50, 75, 100
#    All share the same predict server (no training, read-only model)
#    Expected duration: ~4-6 hours (inference is fast — no gradient updates)
#
# Usage
# -----
#   bash deploy_full.sh             # run both phases sequentially
#   bash deploy_full.sh train       # phase 1 only (training)
#   bash deploy_full.sh infer       # phase 2 only (requires trained weights)
#   bash deploy_full.sh stop        # stop everything
#   bash deploy_full.sh status      # show all process statuses
#   bash deploy_full.sh logs        # tail run logs live
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/../quantum/algorithm" && pwd)"
LOG_BASE="/tmp/qrouting_logs"
PID_DIR="${LOG_BASE}/pids"
PYTHON="${PYTHON:-python3}"
MODE="${1:-all}"

# ── Paper run config ──────────────────────────────────────────────────────────
NUM_WORKERS=4           # training workers (FedAvg)
TRAIN_LOAD=100          # highest req load for training
TTIME=10000             # timeslots for training
STEP=1000               # CSV sample stride (10 plot points)
TIMES=3                 # independent runs per load (averaged for paper results)
INFER_LOADS=(5 10 25 50 75 100)
PREDICT_PORT=8080
BASE_WORKER_PORT=8000
REDIS_DB=0
AGG_INTERVAL=30         # FedAvg every 30 s

TOTAL_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
CORES_PER_WORKER=$(( TOTAL_CORES / (NUM_WORKERS + 4) ))
CORES_PER_WORKER=$(( CORES_PER_WORKER < 2 ? 2 : CORES_PER_WORKER ))

PT_SERVER="${SCRIPT_DIR}/pt/server.py"

BASE_ENV="REDIS_DB=${REDIS_DB} BASE_WORKER_PORT=${BASE_WORKER_PORT} \
PREDICT_PORT=${PREDICT_PORT} \
TTIME=${TTIME} STEP=${STEP} TIMES=${TIMES}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────
kill_by_pid_file() {
    local f="$1"
    [[ -f "$f" ]] || return
    local pid; pid=$(<"$f")
    kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null && log "  Killed PID $pid"
    rm -f "$f"
}

kill_by_port() {
    local port="$1"
    local pids
    if command -v fuser &>/dev/null; then
        pids=$(fuser "${port}/tcp" 2>/dev/null | tr ' ' '\n' | grep -v '^$' || true)
    else
        pids=$(lsof -ti tcp:"${port}" 2>/dev/null || true)
    fi
    for pid in $pids; do kill "$pid" 2>/dev/null || true; done
}

wait_for_log() {
    local logfile="$1" pattern="$2" timeout="${3:-180}" label="${4:-service}"
    local elapsed=0
    until grep -q "$pattern" "$logfile" 2>/dev/null; do
        sleep 2; elapsed=$(( elapsed + 2 ))
        if (( elapsed >= timeout )); then
            echo "ERROR: $label failed to start"
            tail -15 "$logfile"; exit 1
        fi
    done
}

stop_all() {
    log "Stopping all processes..."
    kill_by_pid_file "${PID_DIR}/run_train.pid"
    kill_by_pid_file "${PID_DIR}/aggregator.pid"
    kill_by_pid_file "${PID_DIR}/predict.pid"
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        kill_by_pid_file "${PID_DIR}/worker${wid}.pid"
    done
    for load in "${INFER_LOADS[@]}"; do
        kill_by_pid_file "${PID_DIR}/run_infer_req${load}.pid"
    done
    pkill -f "pt/server.py"             2>/dev/null || true
    pkill -f "dist_agent_aggregator.py" 2>/dev/null || true
    pkill -f "dist_agent_predict.py"    2>/dev/null || true
    pkill -f "dist_agent.py"            2>/dev/null || true
    pkill -f "Run.py"                   2>/dev/null || true
    sleep 2; log "Done."
}

status_all() {
    echo ""; echo "══════════════════════ Full Run Status ══════════════════════"
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        local f="${PID_DIR}/worker${wid}.pid"
        [[ -f "$f" ]] && kill -0 "$(<"$f")" 2>/dev/null \
            && echo "  Worker ${wid}: ✓  PID=$(<"$f")" || echo "  Worker ${wid}: ✗"
    done
    local pf="${PID_DIR}/predict.pid"
    [[ -f "$pf" ]] && kill -0 "$(<"$pf")" 2>/dev/null \
        && echo "  Predict:  ✓  PID=$(<"$pf")" || echo "  Predict:  ✗"
    local rf="${PID_DIR}/run_train.pid"
    [[ -f "$rf" ]] && kill -0 "$(<"$rf")" 2>/dev/null \
        && echo "  Training Run: ✓  PID=$(<"$rf")" || echo "  Training Run: ✗"
    echo ""
    for load in "${INFER_LOADS[@]}"; do
        local irf="${PID_DIR}/run_infer_req${load}.pid"
        [[ -f "$irf" ]] && kill -0 "$(<"$irf")" 2>/dev/null \
            && echo "  Infer req=${load}: ✓  PID=$(<"$irf")" \
            || echo "  Infer req=${load}: ✗"
    done
    echo ""; echo "  Logs: ${LOG_BASE}/"
    echo "═══════════════════════════════════════════════════════════"
}

tail_logs() {
    local files=()
    [[ -f "${LOG_BASE}/run_train.log"  ]] && files+=("${LOG_BASE}/run_train.log")
    for load in "${INFER_LOADS[@]}"; do
        [[ -f "${LOG_BASE}/run_infer_req${load}.log" ]] && \
            files+=("${LOG_BASE}/run_infer_req${load}.log")
    done
    [[ ${#files[@]} -gt 0 ]] && tail -f "${files[@]}" || echo "No log files found."
}

[[ "$MODE" == "stop"   ]] && { stop_all;   exit 0; }
[[ "$MODE" == "status" ]] && { status_all; exit 0; }
[[ "$MODE" == "logs"   ]] && { tail_logs;  exit 0; }

mkdir -p "${LOG_BASE}" "${PID_DIR}"

# ── Redis ─────────────────────────────────────────────────────────────────────
if ! redis-cli ping &>/dev/null; then
    log "Starting Redis..."
    redis-server --daemonize yes \
        --logfile "${LOG_BASE}/redis.log" \
        --maxmemory 150gb \
        --maxmemory-policy allkeys-lru \
        --databases 16
    sleep 2
fi
log "Redis OK"

# =============================================================================
# PHASE 1 — TRAINING
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  PHASE 1: TRAINING"
    echo "  req=${TRAIN_LOAD}  TS=${TTIME}  times=${TIMES}  workers=${NUM_WORKERS}"
    echo "  Total cores: ${TOTAL_CORES}  cores/worker: ${CORES_PER_WORKER}"
    echo "═══════════════════════════════════════════════════════════"

    stop_all 2>/dev/null || true
    redis-cli -n "${REDIS_DB}" FLUSHDB > /dev/null && log "Redis DB ${REDIS_DB} flushed"

    # Start training workers (PyTorch)
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        wlog="${LOG_BASE}/worker${wid}.log"; > "$wlog"
        env $BASE_ENV WORKER_ID=${wid} \
            OMP_NUM_THREADS=${CORES_PER_WORKER} \
            nohup "${PYTHON}" -u "${PT_SERVER}" > "$wlog" 2>&1 &
        echo $! > "${PID_DIR}/worker${wid}.pid"
    done
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        wait_for_log "${LOG_BASE}/worker${wid}.log" "Application startup complete" 180 "Worker ${wid}"
    done
    log "  Workers ready"

    # Start predict server (training mode, PyTorch)
    plog="${LOG_BASE}/predict_train.log"; > "$plog"
    env $BASE_ENV \
        OMP_NUM_THREADS=$(( CORES_PER_WORKER * 2 )) \
        nohup "${PYTHON}" -u "${PT_SERVER}" > "$plog" 2>&1 &
    echo $! > "${PID_DIR}/predict.pid"
    wait_for_log "$plog" "Application startup complete" 180 "Predict server"
    log "  Predict server ready (training mode)"

    # Start FedAvg aggregator
    alog="${LOG_BASE}/aggregator.log"; > "$alog"
    env $BASE_ENV NUM_WORKERS=${NUM_WORKERS} AGGREGATION_EVERY_S=${AGG_INTERVAL} \
        nohup "${PYTHON}" -u "${SCRIPT_DIR}/dist_agent_aggregator.py" > "$alog" 2>&1 &
    echo $! > "${PID_DIR}/aggregator.pid"
    log "  Aggregator started (FedAvg every ${AGG_INTERVAL}s)"

    # Run training (blocks until complete)
    rlog="${LOG_BASE}/run_train.log"; > "$rlog"
    log "  Launching Run.py (training)..."
    ( cd "${ALGO_DIR}" && env $BASE_ENV REQ_LOADS=${TRAIN_LOAD} OMP_NUM_THREADS=4 \
        nohup "${PYTHON}" -u Run.py > "$rlog" 2>&1 ) &
    echo $! > "${PID_DIR}/run_train.pid"
    log "  Training PID=$!  — waiting for completion..."

    # Wait for training to finish (monitor run log for completion marker)
    wait_for_log "$rlog" "EXIT" $(( TTIME * 60 )) "Training run"
    log "  Training complete"

    # Stop aggregator and workers (no longer needed)
    kill_by_pid_file "${PID_DIR}/aggregator.pid"
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        kill_by_pid_file "${PID_DIR}/worker${wid}.pid"
    done
    # Stop training predict server
    kill_by_pid_file "${PID_DIR}/predict.pid"
    sleep 2

    # Persist final model from Redis to disk
    log "  Saving trained model to disk..."
    env $BASE_ENV PYTHONPATH="${SCRIPT_DIR}" \
        "${PYTHON}" -u "${SCRIPT_DIR}/save_trained_model.py" | tee "${LOG_BASE}/save_model.log"
fi

# =============================================================================
# PHASE 2 — INFERENCE
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "infer" ]]; then
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  PHASE 2: INFERENCE"
    echo "  loads=${INFER_LOADS[*]}  TS=${TTIME}  times=${TIMES}"
    echo "  Frozen model, epsilon=0, no training"
    echo "═══════════════════════════════════════════════════════════"

    # Start single shared predict server in inference mode (PyTorch)
    iplog="${LOG_BASE}/predict_infer.log"; > "$iplog"
    env $BASE_ENV INFERENCE_MODE=1 \
        OMP_NUM_THREADS=$(( CORES_PER_WORKER * 2 )) \
        nohup "${PYTHON}" -u "${PT_SERVER}" > "$iplog" 2>&1 &
    echo $! > "${PID_DIR}/predict.pid"
    wait_for_log "$iplog" "Application startup complete" 180 "Predict server (inference)"
    log "  Predict server ready (INFERENCE mode, epsilon=0)"

    # Launch one Run.py per req load in parallel
    # All share the same predict server
    log "  Launching ${#INFER_LOADS[@]} inference processes..."
    for load in "${INFER_LOADS[@]}"; do
        irlog="${LOG_BASE}/run_infer_req${load}.log"; > "$irlog"
        ( cd "${ALGO_DIR}" && env $BASE_ENV INFERENCE_MODE=1 REQ_LOADS=${load} \
            OMP_NUM_THREADS=4 \
            nohup "${PYTHON}" -u Run.py > "$irlog" 2>&1 ) &
        echo $! > "${PID_DIR}/run_infer_req${load}.pid"
        log "    req=${load}  PID=$!"
    done

    # Wait for all inference runs to finish
    log "  Waiting for all inference runs to complete..."
    for load in "${INFER_LOADS[@]}"; do
        pid_file="${PID_DIR}/run_infer_req${load}.pid"
        if [[ -f "$pid_file" ]]; then
            pid=$(<"$pid_file")
            wait "$pid" 2>/dev/null && log "    req=${load} done" || log "    req=${load} exited"
        fi
    done

    kill_by_pid_file "${PID_DIR}/predict.pid"
    log "  All inference runs complete"
fi

# ── Final summary ─────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Full run complete."
echo ""
echo "  Progress CSVs (per-timeslot metrics, all algorithms):"
for load in "${INFER_LOADS[@]}"; do
    csv="${LOG_BASE}/progress_QuRA_Seq_DIST_req${load}.csv"
    [[ -f "$csv" ]] && echo "    ${csv}"
done
echo ""
echo "  Paper data files:  ../../plot/data/Timeslot_#successRequest*.txt"
echo ""
echo "  Check status:  bash deploy_full.sh status"
echo "═══════════════════════════════════════════════════════════"
