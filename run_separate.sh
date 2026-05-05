#!/usr/bin/env bash
# =============================================================================
# run_separate.sh — launch each algorithm as an independent background process
#
# Usage
# -----
#   bash run_separate.sh [start|status|stop|tail <algo>|log <algo>]
#
#   start         : (default) launch all 6 algorithms in the background
#   status        : show running / done / exit-code for each algo
#   stop          : kill all running algo processes
#   tail <algo>   : live-stream log for one algo (Ctrl-C to stop)
#   log  <algo>   : print the full log for one algo
#
# Algorithms
# ----------
#   QuRA_Seq_DIST   QuRA_Flock_DIST   QuRA_Guard_DIST
#   QuRA_Hive_DIST  RELiQ             EBSPA
#
# Config (edit below or override via env)
# ----------------------------------------
#   SIZE=100  TRAIN_LOAD=25  TTIME=20000  TRAINING_MODE=paper  TTL_W=75
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="${ROOT}/src/quantum/algorithm"
PYTHON="${PYTHON:-python3}"

# ── Config (override via env) ─────────────────────────────────────────────────
SIZE="${SIZE:-100}"
TRAIN_LOAD="${TRAIN_LOAD:-25}"
TTIME="${TTIME:-20000}"
TRAINING_MODE="${TRAINING_MODE:-paper}"
TTL_W="${TTL_W:-75}"
STEP="${STEP:-$((TTIME / 100))}"
TORCH_THREADS="${TORCH_THREADS:-2}"
MODEL_DIR="${MODEL_DIR:-${ROOT}/runs_quantum/models/paper_100n}"
RESULTS_DIR="${RESULTS_DIR:-${ROOT}/runs_quantum/results/paper_100n}"

LOG_DIR="${ROOT}/runs_quantum/logs"
PID_DIR="${ROOT}/runs_quantum/pids"

ALGOS=(
  QuRA_Seq_DIST
  QuRA_Flock_DIST
  QuRA_Guard_DIST
  QuRA_Hive_DIST
  RELiQ
  EBSPA
)

# ── Helpers ───────────────────────────────────────────────────────────────────
log()     { echo "[$(date '+%H:%M:%S')] $*"; }
pidfile() { echo "${PID_DIR}/${1}.pid"; }
logfile() { echo "${LOG_DIR}/${1}.log"; }

# ── start ─────────────────────────────────────────────────────────────────────
cmd_start() {
  mkdir -p "${LOG_DIR}" "${PID_DIR}" "${MODEL_DIR}" "${RESULTS_DIR}"

  log "══════════════════════════════════════════════════════"
  log "  SIZE=${SIZE}  TRAIN_LOAD=${TRAIN_LOAD}  TTIME=${TTIME}"
  log "  TRAINING_MODE=${TRAINING_MODE}  TTL_W=${TTL_W}"
  log "  MODEL_DIR=${MODEL_DIR}"
  log "  LOG_DIR=${LOG_DIR}"
  log "══════════════════════════════════════════════════════"
  log ""

  for algo in "${ALGOS[@]}"; do
    local pid_f; pid_f="$(pidfile "${algo}")"
    local log_f; log_f="$(logfile "${algo}")"

    # Skip if already running
    if [[ -f "${pid_f}" ]]; then
      local old_pid; old_pid="$(cat "${pid_f}")"
      if kill -0 "${old_pid}" 2>/dev/null; then
        log "  SKIP  ${algo}  (already running, pid=${old_pid})"
        continue
      fi
    fi

    # Launch in background, redirect stdout+stderr to log file
    (
      cd "${ALGO_DIR}"
      export SIZE TTIME STEP TTL_W TRAINING_MODE MODEL_DIR TORCH_THREADS
      export TRAIN_LOAD REQ_LOADS="${TRAIN_LOAD}"
      export RUN_ALGOS="${algo}"
      exec "${PYTHON}" -u Run.py \
           2>&1
    ) > "${log_f}" &

    local pid=$!
    echo "${pid}" > "${pid_f}"
    log "  START ${algo}  pid=${pid}  log=${log_f}"
  done

  log ""
  log "── Monitor commands ──────────────────────────────────"
  log "  Status of all  : bash run_separate.sh status"
  log "  Tail one algo  : bash run_separate.sh tail QuRA_Hive_DIST"
  log "  Stop all       : bash run_separate.sh stop"
  log ""
}

# ── status ────────────────────────────────────────────────────────────────────
cmd_status() {
  printf "%-22s  %-8s  %-10s  %s\n" "ALGORITHM" "PID" "STATE" "LAST LOG LINE"
  printf "%-22s  %-8s  %-10s  %s\n" "---------" "---" "-----" "-------------"

  for algo in "${ALGOS[@]}"; do
    local pid_f; pid_f="$(pidfile "${algo}")"
    local log_f; log_f="$(logfile "${algo}")"
    local state="not started"
    local pid_str="—"
    local last=""

    if [[ -f "${pid_f}" ]]; then
      local pid; pid="$(cat "${pid_f}")"
      pid_str="${pid}"
      if kill -0 "${pid}" 2>/dev/null; then
        state="RUNNING"
      else
        # Try to get exit code via wait — won't work for external pids; show done
        state="done"
      fi
    fi

    if [[ -f "${log_f}" ]]; then
      last="$(grep -E '(ts=|success=|pid=|ERROR|Traceback)' "${log_f}" \
              | tail -1 | sed 's/^[[:space:]]*//')"
    fi

    printf "%-22s  %-8s  %-10s  %s\n" "${algo}" "${pid_str}" "${state}" "${last}"
  done
}

# ── tail ──────────────────────────────────────────────────────────────────────
cmd_tail() {
  local algo="${1:-}"
  if [[ -z "${algo}" ]]; then
    echo "Usage: bash run_separate.sh tail <algo>" >&2
    echo "Algos: ${ALGOS[*]}" >&2
    exit 1
  fi
  local log_f; log_f="$(logfile "${algo}")"
  if [[ ! -f "${log_f}" ]]; then
    echo "No log yet for ${algo}: ${log_f}" >&2; exit 1
  fi
  echo "=== tailing ${log_f} (Ctrl-C to stop) ==="
  tail -f "${log_f}"
}

# ── log ───────────────────────────────────────────────────────────────────────
cmd_log() {
  local algo="${1:-}"
  if [[ -z "${algo}" ]]; then
    echo "Usage: bash run_separate.sh log <algo>" >&2
    exit 1
  fi
  local log_f; log_f="$(logfile "${algo}")"
  if [[ ! -f "${log_f}" ]]; then
    echo "No log for ${algo}: ${log_f}" >&2; exit 1
  fi
  cat "${log_f}"
}

# ── stop ──────────────────────────────────────────────────────────────────────
cmd_stop() {
  for algo in "${ALGOS[@]}"; do
    local pid_f; pid_f="$(pidfile "${algo}")"
    if [[ -f "${pid_f}" ]]; then
      local pid; pid="$(cat "${pid_f}")"
      if kill -0 "${pid}" 2>/dev/null; then
        kill "${pid}" && log "  STOPPED ${algo}  pid=${pid}"
      else
        log "  SKIP    ${algo}  (pid=${pid} not running)"
      fi
      rm -f "${pid_f}"
    else
      log "  SKIP    ${algo}  (no pid file)"
    fi
  done
}

# ── dispatch ──────────────────────────────────────────────────────────────────
CMD="${1:-start}"
shift || true

case "${CMD}" in
  start)  cmd_start ;;
  status) cmd_status ;;
  tail)   cmd_tail "$@" ;;
  log)    cmd_log  "$@" ;;
  stop)   cmd_stop ;;
  *)
    echo "Unknown command: ${CMD}" >&2
    echo "Usage: bash run_separate.sh [start|status|stop|tail <algo>|log <algo>]" >&2
    exit 1
    ;;
esac
