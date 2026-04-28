#!/usr/bin/env bash
# =============================================================================
# train_reliq_parallel.sh — train RELiQ across N parallel seeds, pick the best
#
# Usage:
#   bash train_reliq_parallel.sh [--jobs N] [--steps S] [--device D]
#
#   --jobs   N  number of parallel workers  (default: 16, adjust to core count)
#   --steps  S  training steps per worker   (default: 500000)
#   --device D  cpu | cuda | mps | auto     (default: cpu for servers)
#
# On a 64-core server: --jobs 16 leaves 4 threads per job.
# Best checkpoint is copied to runs_quantum/RELiQ_QuRAPhysics/model.pt
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python3}"

N_JOBS=16
STEPS=500000
DEVICE="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --jobs)   N_JOBS="$2";  shift 2 ;;
    --steps)  STEPS="$2";   shift 2 ;;
    --device) DEVICE="$2";  shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

# Threads per job: leave 2 cores for OS/IO
TOTAL_CORES=$(nproc 2>/dev/null || sysctl -n hw.logicalcpu 2>/dev/null || echo 8)
THREADS_PER_JOB=$(( (TOTAL_CORES - 2) / N_JOBS ))
THREADS_PER_JOB=$(( THREADS_PER_JOB < 1 ? 1 : THREADS_PER_JOB ))

OUT_BASE="${ROOT}/runs_quantum/reliq_parallel"
FINAL_MODEL="${ROOT}/runs_quantum/RELiQ_QuRAPhysics/model.pt"

mkdir -p "${OUT_BASE}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Launching ${N_JOBS} parallel RELiQ workers"
log "  steps=${STEPS}  device=${DEVICE}  threads_per_job=${THREADS_PER_JOB}"
log "  outputs in ${OUT_BASE}/seed_*/"

# ── Launch all workers in background ─────────────────────────────────────────
PIDS=()
for seed in $(seq 0 $((N_JOBS - 1))); do
  seed_dir="${OUT_BASE}/seed_${seed}"
  mkdir -p "${seed_dir}"
  (
    export OMP_NUM_THREADS="${THREADS_PER_JOB}"
    export MKL_NUM_THREADS="${THREADS_PER_JOB}"
    export OPENBLAS_NUM_THREADS="${THREADS_PER_JOB}"
    cd "${ROOT}" && "${PYTHON}" -m src.reliq.train \
      --total-steps "${STEPS}" \
      --output-dir  "${seed_dir}" \
      --device      "${DEVICE}" \
      --comment     "seed_${seed}"
  ) > "${seed_dir}/train.log" 2>&1 &
  PIDS+=($!)
  log "  started seed=${seed}  pid=${PIDS[-1]}"
done

# ── Wait for all workers ──────────────────────────────────────────────────────
log "Waiting for ${#PIDS[@]} workers…"
FAILED=0
for i in "${!PIDS[@]}"; do
  if wait "${PIDS[$i]}"; then
    log "  seed=${i} finished OK"
  else
    log "  seed=${i} FAILED (exit ${?})"
    FAILED=$((FAILED + 1))
  fi
done

if [[ $FAILED -gt 0 ]]; then
  log "WARNING: ${FAILED} workers failed — check logs in ${OUT_BASE}/seed_*/train.log"
fi

# ── Pick the best checkpoint by eval metric ───────────────────────────────────
log ""
log "Selecting best checkpoint…"

BEST_MODEL=""
BEST_SCORE="-9999"

for seed in $(seq 0 $((N_JOBS - 1))); do
  model_path="${OUT_BASE}/seed_${seed}/seed_${seed}/model.pt"
  metrics_path="${OUT_BASE}/seed_${seed}/eval/metrics.json"

  if [[ ! -f "${model_path}" ]]; then
    log "  seed=${seed}: no checkpoint — skipping"
    continue
  fi

  if [[ -f "${metrics_path}" ]]; then
    score=$(python3 -c "
import json, sys
try:
    m = json.load(open('${metrics_path}'))
    # Use average_episode_packets_mean as the selection metric
    print(m.get('average_episode_packets_mean', -1))
except:
    print(-1)
" 2>/dev/null)
  else
    score="-1"
  fi

  log "  seed=${seed}: score=${score}  (${model_path})"

  if python3 -c "exit(0 if float('${score}') > float('${BEST_SCORE}') else 1)" 2>/dev/null; then
    BEST_SCORE="${score}"
    BEST_MODEL="${model_path}"
  fi
done

if [[ -n "${BEST_MODEL}" ]]; then
  mkdir -p "$(dirname "${FINAL_MODEL}")"
  cp "${BEST_MODEL}" "${FINAL_MODEL}"
  log ""
  log "Best model: seed with score=${BEST_SCORE}"
  log "Copied → ${FINAL_MODEL}"
else
  log "ERROR: no valid checkpoint found in any seed run"
  exit 1
fi

log ""
log "Done. RELiQ model ready at ${FINAL_MODEL}"
