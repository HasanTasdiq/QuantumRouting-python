#!/usr/bin/env bash
# =============================================================================
# run.sh — QuRA-v2 experiment runner
#
# Runs all 6 algorithms on a 10×10 grid (100 nodes):
#   QuRA_Seq_DIST  QuRA_Flock_DIST  QuRA_Guard_DIST  QuRA_Hive_DIST
#   RELiQ  EBSPA
#
# Phases
# ------
#   train  : train QuRA weights (TTIME timeslots, all algorithms in parallel)
#   infer  : load frozen weights, evaluate across REQ_LOADS
#   both   : train then infer (default)
#
# Configs
# -------
#   long   : 100 nodes, 10×10 grid, TTIME=1000000 (full 1M-slot training, server)
#   paper  : 100 nodes, 10×10 grid, TTIME=20000   (paper reproduction)
#   mid    : 25 nodes,  5×5  grid,  TTIME=8000    (medium quality, ~30 min)
#   smoke  : 25 nodes,  5×5  grid,  TTIME=2000    (quick sanity check, ~5 min)
#
# Usage
# -----
#   bash run.sh [config] [phase] [--dry-run]
#
#   config  : long | paper | mid | smoke   (default: paper)
#   phase   : train | infer | both  (default: both)
#   --dry-run: print commands without executing
#
# Environment overrides
# ---------------------
#   MODEL_DIR=/path    where trained weights are saved  (default: runs_quantum/models)
#   RESULTS_DIR=/path  where CSV outputs land           (default: runs_quantum/results)
#   TORCH_THREADS=N    PyTorch threads per worker       (default: 2)
#   RELIQ_STEPS=N      RELiQ pre-training steps        (default: 500000)
#   TIMES=N            number of independent seeds      (default: 5)
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="${ROOT}/src/quantum/algorithm"
PYTHON="${PYTHON:-python3}"

# ── Args ──────────────────────────────────────────────────────────────────────
CONFIG="${1:-paper}"
PHASE="${2:-both}"
DRY_RUN=0
for arg in "$@"; do [[ "$arg" == "--dry-run" ]] && DRY_RUN=1; done

# ── Per-config settings ───────────────────────────────────────────────────────
case "${CONFIG}" in
  smoke)
    TTIME=2000;    TRAIN_LOAD=10; TRAINING_MODE=smoke; TTL_W=25; SIZE=25
    REQ_LOADS_INFER="5,10,15,20"
    LABEL="smoke_25n"
    ;;
  mid)
    TTIME=8000;    TRAIN_LOAD=15; TRAINING_MODE=mid;   TTL_W=50; SIZE=25
    REQ_LOADS_INFER="5,10,15,20,25,30"
    LABEL="mid_25n"
    ;;
  paper)
    TTIME=20000;   TRAIN_LOAD=25; TRAINING_MODE=paper; TTL_W=75; SIZE=100
    REQ_LOADS_INFER="5,10,25,50,75,100"
    LABEL="paper_100n"
    ;;
  long)
    TTIME=1000000; TRAIN_LOAD=25; TRAINING_MODE=long;  TTL_W=75; SIZE=100
    REQ_LOADS_INFER="5,10,25,50,75,100"
    LABEL="long_100n"
    ;;
  *)
    echo "Unknown config: ${CONFIG}. Use smoke|mid|paper|long." >&2; exit 1 ;;
esac

STEP=$((TTIME / 100))
TIMES="${TIMES:-5}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/runs_quantum/models/${LABEL}}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/runs_quantum/results/${LABEL}}"
TORCH_THREADS="${TORCH_THREADS:-2}"
RELIQ_STEPS="${RELIQ_STEPS:-500000}"

RELIQ_CKPT="${ROOT}/runs_quantum/RELiQ_QuRAPhysics/model.pt"

# ── Helpers ───────────────────────────────────────────────────────────────────
log()     { echo "[$(date '+%H:%M:%S')] $*"; }
run_cmd() {
  if [[ $DRY_RUN -eq 1 ]]; then echo "  [DRY] $*"; else eval "$*"; fi
}

mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}"

log "══════════════════════════════════════════════════════"
log "  config=${CONFIG}  phase=${PHASE}  TTIME=${TTIME}"
log "  TRAIN_LOAD=${TRAIN_LOAD}  SIZE=${SIZE}  TIMES=${TIMES}"
log "  TTL_W=${TTL_W}  TRAINING_MODE=${TRAINING_MODE}"
log "  MODEL_DIR=${MODEL_DIR}"
log "  RESULTS_DIR=${RESULTS_DIR}"
log "══════════════════════════════════════════════════════"

# ── RELiQ pre-training (skipped during train-only phase) ─────────────────────
if [[ "${PHASE}" == "train" ]]; then
  log "RELiQ pre-training skipped (train-only phase)."
elif [[ -f "${RELIQ_CKPT}" ]]; then
  log "RELiQ checkpoint found — skipping pre-training."
else
  log "Pre-training RELiQ (${RELIQ_STEPS} steps)…"
  run_cmd "(cd '${ROOT}' && \
      '${PYTHON}' -m src.reliq.train \
      --total-steps '${RELIQ_STEPS}' \
      --output-dir runs_quantum \
      --comment RELiQ_QuRAPhysics)"
fi

# ── Shared env for Run.py (exported so subshells inherit, handles spaces in paths)
export TRAINING_MODE TTIME STEP TTL_W SIZE MODEL_DIR TORCH_THREADS

# ── Phase 1: Training ─────────────────────────────────────────────────────────
if [[ "${PHASE}" == "train" || "${PHASE}" == "both" ]]; then
  log ""
  log "── PHASE 1: TRAINING  load=${TRAIN_LOAD}  TIMES=${TIMES} ──"
  run_cmd "(cd '${ALGO_DIR}' && TIMES=${TIMES} REQ_LOADS=${TRAIN_LOAD} \
      '${PYTHON}' -u Run.py)" \
    2>&1 | tee "${RESULTS_DIR}/train_${LABEL}.log"
  log "Training complete — weights in ${MODEL_DIR}"
fi

# ── Phase 2: Inference ────────────────────────────────────────────────────────
if [[ "${PHASE}" == "infer" || "${PHASE}" == "both" ]]; then
  log ""
  log "── PHASE 2: INFERENCE  loads=${REQ_LOADS_INFER}  TIMES=${TIMES} ──"
  run_cmd "(cd '${ALGO_DIR}' && INFERENCE_MODE=1 TIMES=${TIMES} REQ_LOADS=${REQ_LOADS_INFER} \
      '${PYTHON}' -u Run.py)" \
    2>&1 | tee "${RESULTS_DIR}/infer_${LABEL}.log"
fi

# ── Collect CSVs ──────────────────────────────────────────────────────────────
log ""
log "── COLLECTING RESULTS ──"

ALGOS="QuRA_Seq_DIST QuRA_Flock_DIST QuRA_Guard_DIST QuRA_Hive_DIST RELiQ EBSPA"

if [[ "${PHASE}" == "train" ]]; then
  COLLECT_LOADS="${TRAIN_LOAD}"
else
  COLLECT_LOADS="${REQ_LOADS_INFER}"
fi

for load in $(echo "${COLLECT_LOADS}" | tr ',' ' '); do
  for algo in ${ALGOS}; do
    src_csv="/tmp/qrouting_logs/progress_${algo}_req${load}.csv"
    dst_csv="${RESULTS_DIR}/${algo}_req${load}.csv"
    if [[ -f "${src_csv}" ]]; then
      run_cmd "cp '${src_csv}' '${dst_csv}'"
      log "  saved: ${algo}  req=${load}"
    else
      log "  MISSING: ${algo}  req=${load}"
    fi
  done
done

log ""
log "Done.  Results in ${RESULTS_DIR}/"
log "Run plot_results.py (TODO) to generate paper figures."
