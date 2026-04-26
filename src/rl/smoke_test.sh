#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — Smoke / medium / full-scale test for local PyTorch QuRA.
#
# No HTTP servers. No Redis. No FedAvg. All training is in-process.
#
# Two-phase pipeline
# ------------------
#  Phase 1 — TRAINING  : Run.py trains all algorithms on TRAIN_LOAD
#  Phase 2 — INFERENCE : Run.py evaluates frozen weights on INFER_LOADS
#
# Usage
# -----
#   bash smoke_test.sh             # both phases (default quick-smoke config)
#   bash smoke_test.sh train       # phase 1 only
#   bash smoke_test.sh infer       # phase 2 only (requires saved weights)
#   bash smoke_test.sh verify      # check CSVs and model files
#   bash smoke_test.sh stop        # kill any stray Run.py processes
#
# Config overrides (env vars)
# ---------------------------
#   Quick smoke (default):
#     bash smoke_test.sh
#
#   Medium run (~10 min on a server):
#     TRAIN_LOAD=50 TTIME=2000 STEP=200 TIMES=3 TRAINING_MODE=mid \
#     RELIQ_STEPS=50000 bash smoke_test.sh
#
#   Full paper run:
#     see deploy_full.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/../quantum/algorithm" && pwd)"
LOG_DIR="/tmp/qrouting_logs/smoke"
PYTHON="${PYTHON:-python3}"
MODE="${1:-all}"

# ── Config (all overridable via env) ─────────────────────────────────────────
TRAIN_LOAD="${TRAIN_LOAD:-10}"
TTIME="${TTIME:-50}"
STEP="${STEP:-5}"
TIMES="${TIMES:-2}"
INFER_LOADS="${INFER_LOADS:-5,10,25}"
TRAINING_MODE="${TRAINING_MODE:-smoke}"
RELIQ_STEPS="${RELIQ_STEPS:-5000}"
MODEL_DIR="${MODEL_DIR:-/tmp/qrouting_model}"

BASE_ENV="TRAINING_MODE=${TRAINING_MODE} TTIME=${TTIME} STEP=${STEP} \
TIMES=${TIMES} MODEL_DIR=${MODEL_DIR}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

stop_all() {
    log "Stopping any stray Run.py processes..."
    pkill -f "Run.py" 2>/dev/null || true
    sleep 1
    log "Done."
}

verify_results() {
    echo ""
    echo "══════════ Smoke Verification ══════════"
    local ok=0 fail=0

    # Check per-algorithm CSVs
    local algos=("QuRA_Seq_DIST" "QuRA_Flock_DIST" "QuRA_Guard_DIST"
                 "QuRA_Hive_DIST" "RELiQ" "ShortestPath")

    for load in $(echo "${INFER_LOADS}" | tr ',' ' '); do
        for algo in "${algos[@]}"; do
            local csv="/tmp/qrouting_logs/progress_${algo}_req${load}.csv"
            if [[ ! -f "$csv" ]]; then
                echo "  [MISSING] ${algo} req=${load}"
                (( fail++ )) || true; continue
            fi
            local rows; rows=$(( $(wc -l < "$csv") - 1 ))
            if (( rows < 1 )); then
                echo "  [EMPTY]   ${algo} req=${load}"
                (( fail++ )) || true; continue
            fi
            local max_succ; max_succ=$(tail -n +2 "$csv" | cut -d',' -f2 | sort -n | tail -1)
            if [[ "${max_succ:-0}" -gt 0 ]]; then
                echo "  [PASS]    ${algo} req=${load}  rows=${rows}  max_succ=${max_succ}"
                (( ok++ )) || true
            else
                echo "  [WARN]    ${algo} req=${load}  rows=${rows}  max_succ=0"
                (( fail++ )) || true
            fi
        done
    done

    # Check QuRA model checkpoints
    echo ""
    local qura_algos=("qura_seq_dist" "qura_flock_dist" "qura_guard_dist" "qura_hive_dist")
    for name in "${qura_algos[@]}"; do
        local pt="${MODEL_DIR}/${name}.pt"
        if [[ -f "$pt" ]]; then
            echo "  [PASS]    model: ${pt}"
            (( ok++ )) || true
        else
            echo "  [WARN]    model missing: ${pt}"
            (( fail++ )) || true
        fi
    done

    # Check RELiQ checkpoint
    local reliq_model="${SCRIPT_DIR}/../../runs_quantum/RELiQ_QuRAPhysics/model.pt"
    if [[ -f "$reliq_model" ]]; then
        echo "  [PASS]    RELiQ checkpoint present"
        (( ok++ )) || true
    else
        echo "  [WARN]    RELiQ checkpoint missing — adapter ran greedy fallback"
    fi

    echo ""
    echo "  Results: ${ok} PASS  ${fail} FAIL"
    echo "  Logs:    ${LOG_DIR}/"
    echo "  Models:  ${MODEL_DIR}/"
    echo "═══════════════════════════════════════"
    [[ $fail -eq 0 ]]
}

[[ "$MODE" == "stop"   ]] && { stop_all;      exit 0; }
[[ "$MODE" == "verify" ]] && { verify_results; exit $?; }

mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

# ── Train RELiQ baseline (skip if checkpoint already exists) ─────────────────
RELIQ_MODEL="${SCRIPT_DIR}/../../runs_quantum/RELiQ_QuRAPhysics/model.pt"
if [[ -f "${RELIQ_MODEL}" ]]; then
    log "RELiQ checkpoint already at ${RELIQ_MODEL} — skipping train"
else
    rlog="${LOG_DIR}/reliq_train.log"; > "$rlog"
    log "Training RELiQ baseline (${RELIQ_STEPS} steps) — see ${rlog}"
    ( cd "${SCRIPT_DIR}/../.." && \
      "${PYTHON}" -u -m src.reliq.train \
        --total-steps "${RELIQ_STEPS}" \
        --output-dir runs_quantum \
        --device cpu \
        --comment RELiQ_QuRAPhysics ) > "$rlog" 2>&1 || \
      log "WARNING: RELiQ train failed — adapter will use greedy fallback."
fi

# =============================================================================
# PHASE 1 — TRAINING
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "train" ]]; then
    echo ""
    log "════════════════════════════════════════════"
    log "  PHASE 1: TRAINING"
    log "  req=${TRAIN_LOAD}  TS=${TTIME}  times=${TIMES}  mode=${TRAINING_MODE}"
    log "════════════════════════════════════════════"

    rlog="${LOG_DIR}/run_train.log"; > "$rlog"
    ( cd "${ALGO_DIR}" && env $BASE_ENV REQ_LOADS=${TRAIN_LOAD} TIMES=${TIMES} \
        "${PYTHON}" -u Run.py ) 2>&1 | tee "$rlog"
    log "Training complete"
fi

# =============================================================================
# PHASE 2 — INFERENCE
# =============================================================================
if [[ "$MODE" == "all" || "$MODE" == "infer" ]]; then
    echo ""
    log "════════════════════════════════════════════"
    log "  PHASE 2: INFERENCE  loads=${INFER_LOADS}  TS=${TTIME}"
    log "════════════════════════════════════════════"

    irlog="${LOG_DIR}/run_infer.log"; > "$irlog"
    ( cd "${ALGO_DIR}" && env $BASE_ENV INFERENCE_MODE=1 \
        REQ_LOADS="${INFER_LOADS}" TIMES=${TIMES} \
        "${PYTHON}" -u Run.py ) 2>&1 | tee "$irlog"
    log "Inference complete"
fi

# =============================================================================
# RESULTS
# =============================================================================
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done.  Running verification..."
echo "═══════════════════════════════════════════════════════"
verify_results || true
