#!/usr/bin/env bash
# =============================================================================
# smoke_mid.sh — Mid-weight smoke test: training + learning verification only.
#
# Lighter than smoke_test_200.sh (50 timeslots, 1 worker, DB=2, no inference).
# Use this to quickly confirm the PyTorch stack is wired up correctly.
#
# Usage
# -----
#   bash smoke_mid.sh          # run training then verify
#   bash smoke_mid.sh verify   # re-run verification against existing logs
#   bash smoke_mid.sh stop     # kill servers
#   bash smoke_mid.sh status   # show process status
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/../quantum/algorithm" && pwd)"
PT_DIR="${SCRIPT_DIR}/pt"
LOG_DIR="/tmp/qrouting_logs/mid"
PID_DIR="${LOG_DIR}/pids"
PYTHON="${PYTHON:-python3}"
MODE="${1:-train}"

# ── Mid config ────────────────────────────────────────────────────────────────
NUM_WORKERS=1
TRAIN_LOAD=10
TTIME=50
STEP=5
TIMES=1
PREDICT_PORT=8080
BASE_WORKER_PORT=8000
REDIS_DB=2          # DB=2: isolated from paper (DB=0) and full smoke (DB=1)
AGG_INTERVAL_S=8
TRAINING_MODE=mid

BASE_ENV="REDIS_DB=${REDIS_DB} \
BASE_WORKER_PORT=${BASE_WORKER_PORT} \
PREDICT_PORT=${PREDICT_PORT} \
NUM_WORKERS=${NUM_WORKERS} \
AGG_INTERVAL_S=${AGG_INTERVAL_S} \
TRAINING_MODE=${TRAINING_MODE} \
TTIME=${TTIME} STEP=${STEP} TIMES=${TIMES}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────
kill_by_pid_file() {
    local f="$1"; [[ -f "$f" ]] || return
    local pid; pid=$(<"$f")
    kill -0 "$pid" 2>/dev/null && kill "$pid" 2>/dev/null && log "  Killed PID $pid"
    rm -f "$f"
}

wait_for_log() {
    local logfile="$1" pattern="$2" timeout="${3:-90}" label="${4:-service}"
    local elapsed=0
    until grep -q "$pattern" "$logfile" 2>/dev/null; do
        sleep 2; elapsed=$(( elapsed + 2 ))
        if (( elapsed >= timeout )); then
            log "ERROR: $label did not start (waited ${timeout}s)"
            tail -20 "$logfile"; exit 1
        fi
    done
}

stop_all() {
    log "Stopping mid-smoke processes..."
    kill_by_pid_file "${PID_DIR}/predict.pid"
    kill_by_pid_file "${PID_DIR}/worker0.pid"
    kill_by_pid_file "${PID_DIR}/run_train.pid"
    pkill -f "pt/server.py"  2>/dev/null || true
    pkill -f "Run.py"        2>/dev/null || true
    sleep 1; log "Done."
}

status_all() {
    echo ""; echo "══════════ Mid Smoke Status ══════════"
    local wf="${PID_DIR}/worker0.pid"
    [[ -f "$wf" ]] && kill -0 "$(<"$wf")" 2>/dev/null \
        && echo "  Worker 0: ✓  PID=$(<"$wf")" || echo "  Worker 0: ✗"
    local pf="${PID_DIR}/predict.pid"
    [[ -f "$pf" ]] && kill -0 "$(<"$pf")" 2>/dev/null \
        && echo "  Predict:  ✓  PID=$(<"$pf")" || echo "  Predict:  ✗"
    local rf="${PID_DIR}/run_train.pid"
    [[ -f "$rf" ]] && kill -0 "$(<"$rf")" 2>/dev/null \
        && echo "  Run(train): ✓  PID=$(<"$rf")" || echo "  Run(train): ✗"
    echo "══════════════════════════════════════"
}

verify_results() {
    echo ""; echo "══════════ Mid Smoke Verification ══════════"
    local ok=0 fail=0
    local algos=("QuRA_Seq_DIST" "QuRA_Flock_DIST" "QuRA_Guard_DIST" "QuRA_Hive_DIST" "RELiQ")

    for algo in "${algos[@]}"; do
        local csv="/tmp/qrouting_logs/progress_${algo}_req${TRAIN_LOAD}.csv"
        if [[ ! -f "$csv" ]]; then
            echo "  [MISSING] ${algo}"
            (( fail++ )) || true; continue
        fi
        local rows; rows=$(( $(wc -l < "$csv") - 1 ))
        if (( rows < 1 )); then
            echo "  [EMPTY]   ${algo}  (0 data rows)"
            (( fail++ )) || true; continue
        fi
        local max_succ; max_succ=$(tail -n +2 "$csv" | cut -d',' -f2 | sort -n | tail -1)
        local last_row; last_row=$(tail -1 "$csv")
        if [[ "${max_succ:-0}" -gt 0 ]]; then
            echo "  [PASS]    ${algo}  rows=${rows}  max_succ=${max_succ}  last=${last_row}"
            (( ok++ )) || true
        else
            echo "  [WARN]    ${algo}  rows=${rows}  max_succ=0 (no routing yet)"
            (( fail++ )) || true
        fi
    done

    # Learning signal: worker0 should have fired replay steps
    echo ""
    local wlog="${LOG_DIR}/worker0.log"
    if [[ -f "$wlog" ]]; then
        local replay_lines; replay_lines=$(grep -c "replay#\|loss=" "$wlog" 2>/dev/null || true)
        local update_lines; update_lines=$(grep -c "/update_reward" "$wlog" 2>/dev/null || true)
        echo "  Worker-0 /update_reward hits : ${update_lines}"
        if (( replay_lines > 0 )); then
            echo "  [PASS]    Worker-0 ran ${replay_lines} replay/loss lines (learning confirmed)"
            (( ok++ )) || true
        else
            echo "  [WARN]    No replay steps seen in worker0.log — buffer may not have filled yet"
            echo "            (needs ~20 samples; with ${TTIME} TS and req=${TRAIN_LOAD} this should fire)"
        fi
        echo ""
        echo "  Last 10 lines of worker0.log:"
        tail -10 "$wlog" | sed 's/^/    /'
    else
        echo "  [MISSING] ${wlog}"
        (( fail++ )) || true
    fi

    # Predict server: confirm FedAvg actually aggregated (not just "skipping")
    local plog="${LOG_DIR}/predict_train.log"
    if [[ -f "$plog" ]]; then
        local fedavg_real; fedavg_real=$(grep -c "global model v" "$plog" 2>/dev/null || true)
        local fedavg_skip; fedavg_skip=$(grep -c "skipping" "$plog" 2>/dev/null || true)
        if (( fedavg_real > 0 )); then
            echo "  [PASS]    FedAvg aggregated ${fedavg_real} times (skipped ${fedavg_skip})"
            (( ok++ )) || true
        else
            echo "  [WARN]    FedAvg only skipped (${fedavg_skip}x) — worker weights not in Redis yet"
            echo "            (FEDAVG_PUSH_EVERY=20 replays; need >20 replay steps before FedAvg fires)"
        fi
    fi

    echo ""
    echo "  Results: ${ok} PASS  ${fail} FAIL"
    echo "  Logs:    ${LOG_DIR}/"
    echo "  CSVs:    /tmp/qrouting_logs/progress_*_req${TRAIN_LOAD}.csv"
    echo "══════════════════════════════════════════"
    [[ $fail -eq 0 ]]
}

[[ "$MODE" == "stop"   ]] && { stop_all;       exit 0; }
[[ "$MODE" == "status" ]] && { status_all;     exit 0; }
[[ "$MODE" == "verify" ]] && { verify_results; exit $?; }

mkdir -p "${LOG_DIR}" "${PID_DIR}"

# ── Redis ─────────────────────────────────────────────────────────────────────
if ! redis-cli ping &>/dev/null; then
    log "Starting Redis..."
    redis-server --daemonize yes --logfile "${LOG_DIR}/redis.log" \
        --maxmemory 4gb --maxmemory-policy allkeys-lru
    sleep 2
fi
redis-cli -n "${REDIS_DB}" FLUSHDB > /dev/null
log "Redis ready (DB ${REDIS_DB} flushed)"

# =============================================================================
# TRAINING
# =============================================================================
echo ""
log "════════════════════════════════════════════"
log "  MID SMOKE: TRAINING"
log "  req=${TRAIN_LOAD}  TS=${TTIME}  workers=${NUM_WORKERS}  times=${TIMES}  DB=${REDIS_DB}"
log "════════════════════════════════════════════"

stop_all 2>/dev/null || true

# Start single training worker (port 8000)
wlog="${LOG_DIR}/worker0.log"; > "$wlog"
env $BASE_ENV WORKER_ID=0 PREDICT_PORT=${BASE_WORKER_PORT} \
    OMP_NUM_THREADS=2 \
    nohup "${PYTHON}" -u "${PT_DIR}/server.py" > "$wlog" 2>&1 &
echo $! > "${PID_DIR}/worker0.pid"
log "  Worker 0 started on port ${BASE_WORKER_PORT} (PID=$!)"
wait_for_log "$wlog" "Application startup complete" 90 "Worker 0"
log "  Worker 0 ready"

# Start predict server (port 8080, runs FedAvg loop)
plog="${LOG_DIR}/predict_train.log"; > "$plog"
env $BASE_ENV WORKER_ID=-1 PREDICT_PORT=${PREDICT_PORT} \
    OMP_NUM_THREADS=2 \
    nohup "${PYTHON}" -u "${PT_DIR}/server.py" > "$plog" 2>&1 &
echo $! > "${PID_DIR}/predict.pid"
wait_for_log "$plog" "Application startup complete" 90 "Predict server"
log "  Predict server ready (FedAvg every ${AGG_INTERVAL_S}s)"

# Run training (synchronous)
rlog="${LOG_DIR}/run_train.log"; > "$rlog"
log "  Launching Run.py..."
( cd "${ALGO_DIR}" && env $BASE_ENV REQ_LOADS=${TRAIN_LOAD} TIMES=${TIMES} \
    "${PYTHON}" -u Run.py ) 2>&1 | tee "$rlog"
log "  Training run complete"

# Save weights
save_log="${LOG_DIR}/save_model.log"
log "  Saving global model to disk..."
env $BASE_ENV "${PYTHON}" -u "${PT_DIR}/save_trained_model_pt.py" \
    > "$save_log" 2>&1 || true
cat "$save_log"

# Stop servers
kill_by_pid_file "${PID_DIR}/predict.pid"
kill_by_pid_file "${PID_DIR}/worker0.pid"

# =============================================================================
# VERIFY
# =============================================================================
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Mid smoke done. Verifying..."
echo "═══════════════════════════════════════════════════════"
verify_results || true
