#!/usr/bin/env bash
# =============================================================================
# smoke_test.sh — Smoke / medium / full-scale test for local PyTorch QuRA.
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
#   Medium run (~10 min):
#     TRAIN_LOAD=50 TTIME=2000 STEP=200 TIMES=3 TRAINING_MODE=mid \
#     RELIQ_STEPS=50000 bash smoke_test.sh
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/../quantum/algorithm" && pwd)"
LOG_DIR="/tmp/qrouting_logs/smoke"
PYTHON="${PYTHON:-python3}"
MODE="${1:-all}"

# ── Config (all overridable via env) ─────────────────────────────────────────
TRAIN_LOAD="${TRAIN_LOAD:-10}"
TTIME="${TTIME:-2000}"        # theory: need ≥322 slots for enough +ve transitions
STEP="${STEP:-200}"
TIMES="${TIMES:-1}"
INFER_LOADS="${INFER_LOADS:-5,10,25}"
TRAINING_MODE="${TRAINING_MODE:-smoke}"
RELIQ_STEPS="${RELIQ_STEPS:-5000}"
MODEL_DIR="${MODEL_DIR:-/tmp/qrouting_model}"

BASE_ENV="TRAINING_MODE=${TRAINING_MODE} TTIME=${TTIME} STEP=${STEP} \
TIMES=${TIMES} MODEL_DIR=${MODEL_DIR}"

ALGOS=("QuRA_Seq_DIST" "QuRA_Flock_DIST" "QuRA_Guard_DIST"
       "QuRA_Hive_DIST" "RELiQ" "EBSPA")

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────

stop_all() {
    log "Stopping any stray Run.py processes..."
    pkill -f "Run.py" 2>/dev/null || true
    sleep 1
    log "Done."
}

# check_csv ALGO LOAD PHASE
# Returns 0 on pass, 1 on failure. Prints one-line status.
check_csv() {
    local algo="$1" load="$2" phase="$3"
    local csv="/tmp/qrouting_logs/progress_${algo}_req${load}.csv"
    local label="[${phase}] ${algo} req=${load}"

    if [[ ! -f "$csv" ]]; then
        echo "  [MISSING]  ${label}"
        return 1
    fi

    local rows; rows=$(( $(wc -l < "$csv") - 1 ))
    if (( rows < 1 )); then
        echo "  [EMPTY]    ${label}  rows=0"
        return 1
    fi

    # Max successful requests in any timeslot
    local max_succ; max_succ=$(tail -n +2 "$csv" | cut -d',' -f2 | sort -n | tail -1)

    # Mean wall_ms (4th column)
    local mean_wall; mean_wall=$(tail -n +2 "$csv" | awk -F',' '{s+=$4;n++} END {printf "%.1f",s/n}')

    # Training-progress check: is sum(last 20%) >= sum(first 20%) ?
    local progress_ok="ok"
    if (( rows >= 10 )); then
        local fifth=$(( rows / 5 ))
        local early_sum; early_sum=$(tail -n +2 "$csv" | head -n "$fifth" | awk -F',' '{s+=$2} END {print int(s)}')
        local late_sum;  late_sum=$(tail -n +2 "$csv" | tail -n "$fifth" | awk -F',' '{s+=$2} END {print int(s)}')
        if (( late_sum < early_sum )); then
            progress_ok="regress(early=${early_sum},late=${late_sum})"
        fi
    fi

    if [[ "${max_succ:-0}" -gt 0 ]]; then
        echo "  [PASS]     ${label}  rows=${rows}  max_succ=${max_succ}  wall_ms=${mean_wall}  progress=${progress_ok}"
        return 0
    else
        echo "  [WARN]     ${label}  rows=${rows}  max_succ=0  wall_ms=${mean_wall}"
        return 1
    fi
}

verify_results() {
    local ok=0 fail=0

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Phase 1 — Training CSVs (load=${TRAIN_LOAD})"
    echo "══════════════════════════════════════════════════════"
    for algo in "${ALGOS[@]}"; do
        if check_csv "$algo" "$TRAIN_LOAD" "train"; then (( ok++ )) || true
        else                                              (( fail++ )) || true; fi
    done

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Phase 2 — Inference CSVs (loads=${INFER_LOADS})"
    echo "══════════════════════════════════════════════════════"
    for load in $(echo "${INFER_LOADS}" | tr ',' ' '); do
        for algo in "${ALGOS[@]}"; do
            if check_csv "$algo" "$load" "infer"; then (( ok++ )) || true
            else                                        (( fail++ )) || true; fi
        done
    done

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Model checkpoints"
    echo "══════════════════════════════════════════════════════"

    local qura_algos=("qura_seq_dist" "qura_flock_dist" "qura_guard_dist" "qura_hive_dist")
    for name in "${qura_algos[@]}"; do
        local pt="${MODEL_DIR}/${name}.pt"
        if [[ -f "$pt" ]]; then
            local sz; sz=$(du -sh "$pt" 2>/dev/null | cut -f1)
            echo "  [PASS]     model: ${name}.pt  (${sz})"
            (( ok++ )) || true
        else
            echo "  [WARN]     model missing: ${pt}"
            (( fail++ )) || true
        fi
    done

    local reliq_model="${SCRIPT_DIR}/../../runs_quantum/RELiQ_QuRAPhysics/model.pt"
    if [[ -f "$reliq_model" ]]; then
        local sz; sz=$(du -sh "$reliq_model" 2>/dev/null | cut -f1)
        echo "  [PASS]     RELiQ checkpoint (${sz})"
        (( ok++ )) || true
    else
        echo "  [WARN]     RELiQ checkpoint missing — adapter ran greedy fallback"
    fi

    # Sanity-check: verify no NaN in loss column using Python
    echo ""
    echo "  Checking for NaN/Inf in CSVs..."
    "${PYTHON}" - <<'PYEOF'
import os, glob, csv, math, sys
log_dir = "/tmp/qrouting_logs"
nan_found = False
for path in glob.glob(os.path.join(log_dir, "progress_*.csv")):
    with open(path) as f:
        for i, row in enumerate(csv.reader(f)):
            if i == 0:
                continue
            for val in row:
                try:
                    v = float(val)
                    if math.isnan(v) or math.isinf(v):
                        print(f"  [NaN/Inf]  {os.path.basename(path)} row {i}: {row}")
                        nan_found = True
                except ValueError:
                    pass
if not nan_found:
    print("  [PASS]     No NaN/Inf found in any CSV")
PYEOF

    echo ""
    echo "══════════════════════════════════════════════════════"
    echo "  Summary: ${ok} PASS  ${fail} FAIL"
    echo "  Logs:    ${LOG_DIR}/"
    echo "  Models:  ${MODEL_DIR}/"
    echo "══════════════════════════════════════════════════════"
    [[ $fail -eq 0 ]]
}

[[ "$MODE" == "stop"   ]] && { stop_all;      exit 0; }
[[ "$MODE" == "verify" ]] && { verify_results; exit $?; }

mkdir -p "${LOG_DIR}" "${MODEL_DIR}"

# Clear stale CSVs from previous runs so verify doesn't see ghost passes
log "Clearing stale progress CSVs from /tmp/qrouting_logs/..."
rm -f /tmp/qrouting_logs/progress_*.csv 2>/dev/null || true

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

    # Verify training CSVs immediately
    echo ""
    log "Verifying training output..."
    local_ok=0; local_fail=0
    for algo in "${ALGOS[@]}"; do
        if check_csv "$algo" "$TRAIN_LOAD" "train"; then (( local_ok++ )) || true
        else                                              (( local_fail++ )) || true; fi
    done
    log "Training check: ${local_ok} PASS  ${local_fail} FAIL"
    if (( local_fail > 0 )); then
        log "WARNING: Some training CSVs failed checks — see above."
    fi
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
# FULL VERIFICATION
# =============================================================================
echo ""
echo "═══════════════════════════════════════════════════════"
echo "  Done.  Running full verification..."
echo "═══════════════════════════════════════════════════════"
verify_results || true
