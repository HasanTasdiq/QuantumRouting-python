#!/usr/bin/env bash
# =============================================================================
# repro_paper.sh — Full QuRA-v2 paper reproduction matrix.
#
# Reproduces the 3-topology × 8-load × 7-algorithm experiment from the
# QuRA paper (Fig. 5–7 equivalent).
#
# Configurations
# --------------
#   small   : 10q/4l  N=16  grid=4x4
#   medium  : 25q/4l  N=25  grid=5x5  (default paper grid)
#   paper   : 10q/4l  N=100 grid=10x10
#
# Each configuration trains on TRAIN_LOAD, then evaluates frozen weights
# across REQ_LOADS in inference mode.
#
# Usage
# -----
#   bash repro_paper.sh [config] [--dry-run]
#
#   config   : small | medium | paper  (default: paper)
#   --dry-run: print commands without executing (safe to call any time)
#
# Environment overrides
# ---------------------
#   MODEL_DIR=/path   where trained weights are saved (default /tmp/qrouting_model)
#   RESULTS_DIR=/path where CSV outputs are collected (default runs_quantum/repro)
#   TORCH_THREADS=N   threads per worker process (default 2)
#   RELIQ_STEPS=N     RELiQ pretraining steps (default 500000)
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/../quantum/algorithm" && pwd)"
PYTHON="${PYTHON:-python3}"

CONFIG="${1:-paper}"
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

MODEL_DIR="${MODEL_DIR:-/tmp/qrouting_model}"
RESULTS_DIR="${RESULTS_DIR:-${SCRIPT_DIR}/../../runs_quantum/repro}"
TORCH_THREADS="${TORCH_THREADS:-2}"
RELIQ_STEPS="${RELIQ_STEPS:-500000}"

case "${CONFIG}" in
  small)
    TTIME=8000; TRAIN_LOAD=10       # network capacity ≈ 38; train at 10 for high p_complete
    REQ_LOADS="5,10,15,20,25,30,40,50"
    TRAINING_MODE="mid"; TTL_W=25
    LABEL="small_16n"
    ;;
  medium)
    TTIME=8000; TRAIN_LOAD=15       # train under capacity for positive signal
    REQ_LOADS="5,10,15,20,25,30,40,50"
    TRAINING_MODE="mid"; TTL_W=50
    LABEL="medium_25n"
    ;;
  paper)
    TTIME=20000; TRAIN_LOAD=25      # max-feasible training load (network saturates ~38)
    REQ_LOADS="5,10,25,50,75,100"   # eval generalises to congested loads
    TRAINING_MODE="paper"; TTL_W=75
    LABEL="paper_100n"
    ;;
  *)
    echo "Unknown config: ${CONFIG}. Use small|medium|paper."
    exit 1
    ;;
esac

STEP=$((TTIME / 100))   # 100 CSV checkpoints per run
TIMES=5                   # 5 seeds for statistical significance (mean ± std)

log() { echo "[$(date '+%H:%M:%S')] $*"; }

run_cmd() {
    if [[ $DRY_RUN -eq 1 ]]; then
        echo "  [DRY-RUN] $*"
    else
        eval "$*"
    fi
}

mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}"

log "════════════════════════════════════════════════════"
log "  QuRA-v2 paper reproduction: config=${CONFIG}"
log "  TTIME=${TTIME}  TRAIN_LOAD=${TRAIN_LOAD}  TIMES=${TIMES}"
log "  REQ_LOADS=${REQ_LOADS}  TTL_W=${TTL_W}"
log "  MODEL_DIR=${MODEL_DIR}"
log "  RESULTS_DIR=${RESULTS_DIR}"
log "════════════════════════════════════════════════════"

# =============================================================================
# RELiQ pre-training (skip if checkpoint exists)
# =============================================================================
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RELIQ_CKPT="${PROJECT_ROOT}/runs_quantum/RELiQ_QuRAPhysics/model.pt"

if [[ -f "${RELIQ_CKPT}" ]]; then
    log "RELiQ checkpoint already at ${RELIQ_CKPT} — skipping."
else
    log "Pre-training RELiQ (${RELIQ_STEPS} steps)…"
    run_cmd "(cd '${PROJECT_ROOT}' && \
        '${PYTHON}' -m src.reliq.train \
        --total-steps '${RELIQ_STEPS}' \
        --output-dir runs_quantum \
        --device cpu \
        --comment RELiQ_QuRAPhysics)"
fi

# =============================================================================
# PHASE 1 — TRAINING
# =============================================================================
log ""
log "── PHASE 1: TRAINING  load=${TRAIN_LOAD} ──────────────────────────"

BASE_ENV="TRAINING_MODE=${TRAINING_MODE} \
TTIME=${TTIME} STEP=${STEP} TIMES=${TIMES} \
TTL_W=${TTL_W} MODEL_DIR=${MODEL_DIR} \
TORCH_THREADS=${TORCH_THREADS}"

run_cmd "(cd '${ALGO_DIR}' && env ${BASE_ENV} \
    REQ_LOADS=${TRAIN_LOAD} TIMES=${TIMES} \
    '${PYTHON}' -u Run.py)" 2>&1 | tee "${RESULTS_DIR}/train_${LABEL}.log"

log "Training complete — weights in ${MODEL_DIR}"

# =============================================================================
# PHASE 2 — INFERENCE (frozen weights, sweep all loads)
# =============================================================================
log ""
log "── PHASE 2: INFERENCE  loads=${REQ_LOADS} ───────────────────────────"

run_cmd "(cd '${ALGO_DIR}' && env ${BASE_ENV} \
    INFERENCE_MODE=1 REQ_LOADS=${REQ_LOADS} TIMES=${TIMES} \
    '${PYTHON}' -u Run.py)" 2>&1 | tee "${RESULTS_DIR}/infer_${LABEL}.log"

# =============================================================================
# COLLECT RESULTS
# =============================================================================
log ""
log "── RESULTS ───────────────────────────────────────────────────────"

ALGOS="QuRA_Seq_DIST QuRA_Flock_DIST QuRA_Guard_DIST QuRA_Hive_DIST RELiQ EBSPA"
RESULT_DIR_FULL="${RESULTS_DIR}/${LABEL}"
mkdir -p "${RESULT_DIR_FULL}"

for load in $(echo "${REQ_LOADS}" | tr ',' ' '); do
    for algo in ${ALGOS}; do
        src="/tmp/qrouting_logs/progress_${algo}_req${load}.csv"
        dst="${RESULT_DIR_FULL}/${algo}_req${load}.csv"
        if [[ -f "${src}" ]]; then
            run_cmd "cp '${src}' '${dst}'"
            log "  collected: ${algo} req=${load}"
        else
            log "  MISSING:   ${algo} req=${load}"
        fi
    done
done

log ""
log "All done.  Results in ${RESULT_DIR_FULL}/"
log "Next step: run plot_results.py (not yet written) to generate paper figures."
