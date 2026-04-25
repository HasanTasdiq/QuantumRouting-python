#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — Quick smoke test for the PyTorch QuRA + RELiQ stack.
#
# Two-phase pipeline
# ------------------
#  Phase 1 — TRAINING  (req=10, 50 timeslots, 2 workers)
#    2 worker servers  (ports 8000–8001)  receive /update_reward, train locally
#    1 predict server  (port  8080)       receives /learn_predict_batch, runs FedAvg
#    Run.py (5 algorithms: QuRA-Seq/Flock/Guard/Hive + RELiQ)
#
#  Phase 2 — INFERENCE  (req=5,10,25, 50 timeslots, frozen model, epsilon=0)
#    1 predict server in INFERENCE_MODE=1 + Run.py
#
# Usage
# -----
#   bash smoke_test.sh             # both phases
#   bash smoke_test.sh train       # phase 1 only
#   bash smoke_test.sh infer       # phase 2 only (requires saved weights)
#   bash smoke_test.sh verify      # check CSVs for learning signal
#   bash smoke_test.sh stop        # kill everything
#   bash smoke_test.sh status      # show process status
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/../quantum/algorithm" && pwd)"
PT_DIR="${SCRIPT_DIR}/pt"
LOG_DIR="/tmp/qrouting_logs/smoke"
PID_DIR="${LOG_DIR}/pids"
PYTHON="${PYTHON:-python3}"
MODE="${1:-all}"

# ── Config — override any of these via env before calling the script ──────────
# Quick smoke (default):
#   bash smoke_test.sh
# Medium run (~10 min on a server):
#   NUM_WORKERS=4 TRAIN_LOAD=50 TTIME=2000 STEP=200 TIMES=3 \
#   TRAINING_MODE=mid RELIQ_STEPS=50000 bash smoke_test.sh
# Full-scale (mirrors deploy_full.sh):
#   see deploy_full.sh
NUM_WORKERS="${NUM_WORKERS:-2}"
TRAIN_LOAD="${TRAIN_LOAD:-10}"
TTIME="${TTIME:-50}"
STEP="${STEP:-5}"
TIMES="${TIMES:-2}"
INFER_LOADS="${INFER_LOADS:-5,10,25}"
PREDICT_PORT="${PREDICT_PORT:-8080}"
BASE_WORKER_PORT="${BASE_WORKER_PORT:-8000}"
REDIS_DB="${REDIS_DB:-1}"        # DB=1 keeps smoke isolated from paper run (DB=0)
AGG_INTERVAL_S="${AGG_INTERVAL_S:-10}"
TRAINING_MODE="${TRAINING_MODE:-smoke}"
RELIQ_STEPS="${RELIQ_STEPS:-5000}"

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
    local logfile="$1" pattern="$2" timeout="${3:-120}" label="${4:-service}"
    local elapsed=0
    until grep -q "$pattern" "$logfile" 2>/dev/null; do
        sleep 2; elapsed=$(( elapsed + 2 ))
        if (( elapsed >= timeout )); then
            log "ERROR: $label did not start (waited ${timeout}s for '$pattern')"
            tail -20 "$logfile"; exit 1
        fi
    done
}

stop_all() {
    log "Stopping all smoke processes..."
    kill_by_pid_file "${PID_DIR}/predict.pid"
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        kill_by_pid_file "${PID_DIR}/worker${wid}.pid"
    done
    kill_by_pid_file "${PID_DIR}/run_train.pid"
    kill_by_pid_file "${PID_DIR}/run_infer.pid"
    pkill -f "pt/server.py"  2>/dev/null || true
    pkill -f "Run.py"        2>/dev/null || true
    sleep 1; log "Done."
}

status_all() {
    echo ""; echo "══════════ Smoke Test Status ══════════"
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        local f="${PID_DIR}/worker${wid}.pid"
        [[ -f "$f" ]] && kill -0 "$(<"$f")" 2>/dev/null \
            && echo "  Worker ${wid}: ✓  PID=$(<"$f")" || echo "  Worker ${wid}: ✗"
    done
    local pf="${PID_DIR}/predict.pid"
    [[ -f "$pf" ]] && kill -0 "$(<"$pf")" 2>/dev/null \
        && echo "  Predict:  ✓  PID=$(<"$pf")" || echo "  Predict:  ✗"
    for phase in train infer; do
        local rf="${PID_DIR}/run_${phase}.pid"
        [[ -f "$rf" ]] && kill -0 "$(<"$rf")" 2>/dev/null \
            && echo "  Run(${phase}): ✓  PID=$(<"$rf")" || echo "  Run(${phase}): ✗"
    done
    echo "═══════════════════════════════════════"
}

verify_results() {
    echo ""; echo "══════════ Smoke Verification ══════════"
    local ok=0 fail=0
    local algos=("QuRA_Seq_DIST" "QuRA_Flock_DIST" "QuRA_Guard_DIST" "QuRA_Hive_DIST" "RELiQ")

    for load in $(echo "${INFER_LOADS}" | tr ',' ' '); do
        for algo in "${algos[@]}"; do
            local csv="/tmp/qrouting_logs/progress_${algo}_req${load}.csv"
            if [[ ! -f "$csv" ]]; then
                echo "  [MISSING] ${algo} req=${load}"
                (( fail++ )) || true; continue
            fi
            local rows; rows=$(( $(wc -l < "$csv") - 1 ))  # minus header
            if (( rows < 1 )); then
                echo "  [EMPTY]   ${algo} req=${load}  (0 data rows)"
                (( fail++ )) || true; continue
            fi
            # Check at least one successful request was routed
            local max_succ; max_succ=$(tail -n +2 "$csv" | cut -d',' -f2 | \
                sort -n | tail -1)
            if [[ "$max_succ" -gt 0 ]]; then
                echo "  [PASS]    ${algo} req=${load}  rows=${rows}  max_succ=${max_succ}"
                (( ok++ )) || true
            else
                echo "  [WARN]    ${algo} req=${load}  rows=${rows}  max_succ=0 (no routing)"
                (( fail++ )) || true
            fi
        done
    done

    # Check learning happened on EVERY worker: each algo round-robins to its
    # own worker, so all NUM_WORKERS logs must show replay steps.
    echo ""
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        local wlog="${LOG_DIR}/worker${wid}.log"
        if [[ ! -f "$wlog" ]]; then
            echo "  [WARN]    Worker-${wid} log missing"
            continue
        fi
        local replay_lines; replay_lines=$(grep -c "replay#" "$wlog" 2>/dev/null || echo 0)
        if (( replay_lines > 0 )); then
            echo "  [PASS]    Worker-${wid} ran ${replay_lines} QMIX replay steps"
            (( ok++ )) || true
        else
            echo "  [WARN]    Worker-${wid} log shows no replay steps — training may not have fired"
        fi
    done

    # RELiQ baseline checkpoint
    local reliq_model="${SCRIPT_DIR}/../../runs_quantum/RELiQ_QuRAPhysics/model.pt"
    if [[ -f "$reliq_model" ]]; then
        echo "  [PASS]    RELiQ checkpoint present at ${reliq_model}"
        (( ok++ )) || true
    else
        echo "  [WARN]    RELiQ checkpoint missing — adapter ran in greedy fallback"
    fi

    echo ""
    echo "  Results: ${ok} PASS  ${fail} FAIL"
    echo "  Logs:    ${LOG_DIR}/"
    echo "═══════════════════════════════════════"
    [[ $fail -eq 0 ]]   # exit 0 if all pass
}

[[ "$MODE" == "stop"   ]] && { stop_all;      exit 0; }
[[ "$MODE" == "status" ]] && { status_all;    exit 0; }
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
# PHASE 1 — TRAINING
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
    echo ""
    log "════════════════════════════════════════════"
    log "  PHASE 1: TRAINING"
    log "  req=${TRAIN_LOAD}  TS=${TTIME}  workers=${NUM_WORKERS}  times=${TIMES}"
    log "════════════════════════════════════════════"

    stop_all 2>/dev/null || true

    # Start training worker servers (ports 8000, 8001)
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        wport=$(( BASE_WORKER_PORT + wid ))
        wlog="${LOG_DIR}/worker${wid}.log"; > "$wlog"
        env $BASE_ENV WORKER_ID=${wid} PREDICT_PORT=${wport} \
            OMP_NUM_THREADS=2 \
            nohup "${PYTHON}" -u "${PT_DIR}/server.py" > "$wlog" 2>&1 &
        echo $! > "${PID_DIR}/worker${wid}.pid"
        log "  Worker ${wid} started on port ${wport} (PID=$!)"
    done

    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        wait_for_log "${LOG_DIR}/worker${wid}.log" \
            "Application startup complete" 120 "Worker ${wid}"
    done
    log "  Workers ready"

    # Start predict server (port 8080, IS_PREDICT_SERVER=true → runs FedAvg loop)
    plog="${LOG_DIR}/predict_train.log"; > "$plog"
    env $BASE_ENV WORKER_ID=-1 PREDICT_PORT=${PREDICT_PORT} \
        OMP_NUM_THREADS=4 \
        nohup "${PYTHON}" -u "${PT_DIR}/server.py" > "$plog" 2>&1 &
    echo $! > "${PID_DIR}/predict.pid"
    wait_for_log "$plog" "Application startup complete" 120 "Predict server"
    log "  Predict server ready (training mode, FedAvg every ${AGG_INTERVAL_S}s)"

    # Run training (synchronous — blocks until Run.py exits)
    rlog="${LOG_DIR}/run_train.log"; > "$rlog"
    log "  Launching Run.py (training)..."
    ( cd "${ALGO_DIR}" && env $BASE_ENV REQ_LOADS=${TRAIN_LOAD} TIMES=${TIMES} \
        "${PYTHON}" -u Run.py ) 2>&1 | tee "$rlog"
    log "  Training complete"

    # Save trained weights from Redis to disk
    save_log="${LOG_DIR}/save_model.log"
    log "  Saving global model to disk..."
    env $BASE_ENV "${PYTHON}" -u "${PT_DIR}/save_trained_model_pt.py" \
        > "$save_log" 2>&1 || true
    cat "$save_log"

    # Train (or skip if already present) the RELiQ baseline.
    RELIQ_MODEL="${SCRIPT_DIR}/../../runs_quantum/RELiQ_QuRAPhysics/model.pt"
    if [[ -f "${RELIQ_MODEL}" ]]; then
        log "  RELiQ checkpoint already at ${RELIQ_MODEL} — skipping smoke RELiQ train"
    else
        rlog="${LOG_DIR}/reliq_train.log"; > "$rlog"
        log "  Training RELiQ baseline (${RELIQ_STEPS} steps) — see ${rlog}"
        ( cd "${SCRIPT_DIR}/../.." && \
          "${PYTHON}" -u -m src.reliq.train \
            --total-steps "${RELIQ_STEPS}" \
            --output-dir runs_quantum \
            --device cpu \
            --comment RELiQ_QuRAPhysics ) > "$rlog" 2>&1 || \
          log "  WARNING: RELiQ smoke train failed — adapter will run greedy fallback."
    fi
fi

# =============================================================================
# PHASE 2 — INFERENCE
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "infer" ]]; then
    echo ""
    log "════════════════════════════════════════════"
    log "  PHASE 2: INFERENCE  loads=${INFER_LOADS}  TS=${TTIME}  times=${TIMES}"
    log "════════════════════════════════════════════"

    # Stop training servers
    kill_by_pid_file "${PID_DIR}/predict.pid"
    for (( wid=0; wid<NUM_WORKERS; wid++ )); do
        kill_by_pid_file "${PID_DIR}/worker${wid}.pid"
    done
    sleep 1

    # Start single predict server in inference mode (loads disk weights, epsilon=0)
    iplog="${LOG_DIR}/predict_infer.log"; > "$iplog"
    env $BASE_ENV INFERENCE_MODE=1 WORKER_ID=-1 PREDICT_PORT=${PREDICT_PORT} \
        OMP_NUM_THREADS=4 \
        nohup "${PYTHON}" -u "${PT_DIR}/server.py" > "$iplog" 2>&1 &
    echo $! > "${PID_DIR}/predict.pid"
    wait_for_log "$iplog" "Application startup complete" 120 "Predict server (inference)"
    log "  Predict server ready (INFERENCE mode, epsilon=0)"

    # Run inference across all eval loads
    irlog="${LOG_DIR}/run_infer.log"; > "$irlog"
    ( cd "${ALGO_DIR}" && env $BASE_ENV INFERENCE_MODE=1 \
        REQ_LOADS="${INFER_LOADS}" TIMES=${TIMES} \
        "${PYTHON}" -u Run.py ) 2>&1 | tee "$irlog"
    log "  Inference complete"

    kill_by_pid_file "${PID_DIR}/predict.pid"
fi

# =============================================================================
# RESULTS
# =============================================================================
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Smoke test done.  Running verification..."
echo "═══════════════════════════════════════════════════════"
verify_results || true
