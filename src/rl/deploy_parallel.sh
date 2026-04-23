#!/usr/bin/env bash
# =============================================================================
# deploy_parallel.sh — Run all 6 request-load experiments SIMULTANEOUSLY.
#
# Each experiment is a fully isolated pipeline:
#   4 training workers + predict server + FedAvg aggregator + Run.py
#
# Isolation achieved via:
#   REDIS_DB          — separate Redis database per experiment (no key conflicts)
#   BASE_WORKER_PORT  — separate port range per experiment
#   PREDICT_PORT      — separate predict server port
#   REQ_LOADS         — single request load per experiment
#
# Layout (64-core server)
# -----------------------
#   Exp 0  req=5    workers 8000-8003  predict 8080  redis db=0
#   Exp 1  req=10   workers 8010-8013  predict 8090  redis db=1
#   Exp 2  req=25   workers 8020-8023  predict 8100  redis db=2
#   Exp 3  req=50   workers 8030-8033  predict 8110  redis db=3
#   Exp 4  req=75   workers 8040-8043  predict 8120  redis db=4
#   Exp 5  req=100  workers 8050-8053  predict 8130  redis db=5
#
# Total processes: 6 × (4 workers + 1 predict + 1 aggregator + 1 Run.py) = 42
# Total cores used: ~48-56 out of 64
#
# Speedup: ~3× vs sequential (107 hrs → ~36 hrs = time for heaviest load)
#
# Usage
# -----
#   bash deploy_parallel.sh          # launch all 6 experiments
#   bash deploy_parallel.sh stop     # stop everything
#   bash deploy_parallel.sh status   # show all experiment statuses
#   bash deploy_parallel.sh logs     # tail run logs for all experiments
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ALGO_DIR="$(cd "${SCRIPT_DIR}/../quantum/algorithm" && pwd)"
LOG_BASE="/tmp/qrouting_logs"
PID_BASE="${LOG_BASE}/pids"
PYTHON="${PYTHON:-python3}"
MODE="${1:-start}"

# ── Experiment definitions ────────────────────────────────────────────────────
# Format: "REQ_LOAD BASE_WORKER_PORT PREDICT_PORT REDIS_DB"
EXPERIMENTS=(
    "5   8000 8080 0"
    "10  8010 8090 1"
    "25  8020 8100 2"
    "50  8030 8110 3"
    "75  8040 8120 4"
    "100 8050 8130 5"
)

NUM_WORKERS_PER_EXP=4
AGG_INTERVAL=30
TOTAL_CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8)
CORES_PER_WORKER=$(( TOTAL_CORES / (${#EXPERIMENTS[@]} * (NUM_WORKERS_PER_EXP + 2)) ))
CORES_PER_WORKER=$(( CORES_PER_WORKER < 2 ? 2 : CORES_PER_WORKER ))

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ── Helpers ───────────────────────────────────────────────────────────────────
kill_by_pid_file() {
    local f="$1"
    if [[ -f "$f" ]]; then
        local pid; pid=$(<"$f")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null && log "  Killed PID $pid"
        fi
        rm -f "$f"
    fi
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

exp_log_dir()  { echo "${LOG_BASE}/exp_req${1}"; }
exp_pid_dir()  { echo "${PID_BASE}/exp_req${1}"; }

stop_experiment() {
    local req="$1" base_port="$2"
    local pid_dir; pid_dir=$(exp_pid_dir "$req")
    kill_by_pid_file "${pid_dir}/run.pid"
    kill_by_pid_file "${pid_dir}/aggregator.pid"
    kill_by_pid_file "${pid_dir}/predict.pid"
    for (( wid=0; wid<NUM_WORKERS_PER_EXP; wid++ )); do
        kill_by_pid_file "${pid_dir}/worker${wid}.pid"
    done
    for (( wid=0; wid<NUM_WORKERS_PER_EXP; wid++ )); do
        kill_by_port $(( base_port + wid ))
    done
}

stop_all() {
    log "Stopping all parallel experiments..."
    for exp_def in "${EXPERIMENTS[@]}"; do
        read -r req base_port predict_port redis_db <<< "$exp_def"
        stop_experiment "$req" "$base_port"
        kill_by_port "$predict_port"
    done
    pkill -f "dist_agent_aggregator.py" 2>/dev/null || true
    pkill -f "dist_agent_predict.py"    2>/dev/null || true
    pkill -f "dist_agent.py"            2>/dev/null || true
    pkill -f "Run.py"                   2>/dev/null || true
    sleep 2
    log "Done."
}

status_all() {
    echo ""
    echo "══════════════════════ Parallel Experiment Status ══════════════════════"
    for exp_def in "${EXPERIMENTS[@]}"; do
        read -r req base_port predict_port redis_db <<< "$exp_def"
        local pid_dir; pid_dir=$(exp_pid_dir "$req")
        local log_dir; log_dir=$(exp_log_dir "$req")
        echo ""
        echo "  ── Experiment req=${req}  ports ${base_port}-$(( base_port + 3 ))  predict ${predict_port}"
        for (( wid=0; wid<NUM_WORKERS_PER_EXP; wid++ )); do
            local f="${pid_dir}/worker${wid}.pid"
            if [[ -f "$f" ]] && kill -0 "$(<"$f")" 2>/dev/null; then
                echo "     Worker ${wid}: ✓  (PID=$(<"$f"))"
            else
                echo "     Worker ${wid}: ✗"
            fi
        done
        local pf="${pid_dir}/predict.pid"
        [[ -f "$pf" ]] && kill -0 "$(<"$pf")" 2>/dev/null \
            && echo "     Predict : ✓  (PID=$(<"$pf"))" || echo "     Predict : ✗"
        local rf="${pid_dir}/run.pid"
        [[ -f "$rf" ]] && kill -0 "$(<"$rf")" 2>/dev/null \
            && echo "     Run.py  : ✓  (PID=$(<"$rf"))" || echo "     Run.py  : ✗"
        # Show latest progress if CSV exists
        local csv="${LOG_BASE}/progress_QuRA_Seq_DIST_req${req}.csv"
        if [[ -f "$csv" ]]; then
            local last_ts; last_ts=$(tail -1 "$csv" | cut -d, -f1)
            echo "     Progress: TS ${last_ts} / 10000"
        fi
    done
    echo ""
    echo "  Logs: ${LOG_BASE}/exp_req<N>/"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""
}

tail_logs() {
    local files=()
    for exp_def in "${EXPERIMENTS[@]}"; do
        read -r req _ _ _ <<< "$exp_def"
        local log_dir; log_dir=$(exp_log_dir "$req")
        files+=("${log_dir}/run.log")
    done
    tail -f "${files[@]}" 2>/dev/null
}

[[ "${MODE}" == "stop"   ]] && { stop_all;   exit 0; }
[[ "${MODE}" == "status" ]] && { status_all; exit 0; }
[[ "${MODE}" == "logs"   ]] && { tail_logs;  exit 0; }

# ── START ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  QuRA Parallel Paper Run — ${#EXPERIMENTS[@]} experiments × ${NUM_WORKERS_PER_EXP} workers"
echo "  ${TOTAL_CORES} cores detected — ${CORES_PER_WORKER} cores/worker"
echo "═══════════════════════════════════════════════════════════════"

stop_all

# Ensure Redis is running
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

mkdir -p "${LOG_BASE}" "${PID_BASE}"

# ── Launch each experiment ────────────────────────────────────────────────────
for exp_def in "${EXPERIMENTS[@]}"; do
    read -r req base_port predict_port redis_db <<< "$exp_def"

    log_dir=$(exp_log_dir "$req")
    pid_dir=$(exp_pid_dir "$req")
    mkdir -p "$log_dir" "$pid_dir"

    log "━━━ Experiment req=${req}  ports ${base_port}-$(( base_port+3 ))  predict ${predict_port}  db=${redis_db}"

    # Common env for all processes in this experiment
    EXP_ENV="REDIS_DB=${redis_db} BASE_WORKER_PORT=${base_port} PREDICT_PORT=${predict_port} TF_CPP_MIN_LOG_LEVEL=2"

    # Start training workers
    for (( wid=0; wid<NUM_WORKERS_PER_EXP; wid++ )); do
        wlog="${log_dir}/worker${wid}.log"
        > "$wlog"
        env $EXP_ENV \
            WORKER_ID=${wid} \
            OMP_NUM_THREADS=${CORES_PER_WORKER} \
            TF_NUM_INTRAOP_THREADS=${CORES_PER_WORKER} \
            TF_NUM_INTEROP_THREADS=2 \
            nohup "${PYTHON}" -u "${SCRIPT_DIR}/dist_agent.py" > "$wlog" 2>&1 &
        echo $! > "${pid_dir}/worker${wid}.pid"
    done

    # Wait for workers
    for (( wid=0; wid<NUM_WORKERS_PER_EXP; wid++ )); do
        wlog="${log_dir}/worker${wid}.log"
        elapsed=0
        until grep -q "Application startup complete" "$wlog" 2>/dev/null; do
            sleep 2; elapsed=$(( elapsed + 2 ))
            if (( elapsed >= 120 )); then
                echo "ERROR: Worker ${wid} (req=${req}) failed to start"
                tail -10 "$wlog"; exit 1
            fi
        done
    done
    log "  Workers ready"

    # Start predict server
    plog="${log_dir}/predict.log"; > "$plog"
    env $EXP_ENV \
        OMP_NUM_THREADS=$(( CORES_PER_WORKER * 2 )) \
        TF_NUM_INTRAOP_THREADS=$(( CORES_PER_WORKER * 2 )) \
        TF_NUM_INTEROP_THREADS=2 \
        nohup "${PYTHON}" -u "${SCRIPT_DIR}/dist_agent_predict.py" > "$plog" 2>&1 &
    echo $! > "${pid_dir}/predict.pid"
    elapsed=0
    until grep -q "Application startup complete" "$plog" 2>/dev/null; do
        sleep 2; elapsed=$(( elapsed + 2 ))
        if (( elapsed >= 120 )); then echo "ERROR: Predict (req=${req}) failed"; exit 1; fi
    done
    log "  Predict ready on port ${predict_port}"

    # Start aggregator
    alog="${log_dir}/aggregator.log"; > "$alog"
    env $EXP_ENV \
        NUM_WORKERS=${NUM_WORKERS_PER_EXP} \
        AGGREGATION_EVERY_S=${AGG_INTERVAL} \
        nohup "${PYTHON}" -u "${SCRIPT_DIR}/dist_agent_aggregator.py" > "$alog" 2>&1 &
    echo $! > "${pid_dir}/aggregator.pid"

    # Start Run.py (must run from its own dir for relative imports)
    rlog="${log_dir}/run.log"; > "$rlog"
    ( cd "${ALGO_DIR}" && env $EXP_ENV REQ_LOADS=${req} OMP_NUM_THREADS=4 \
        nohup "${PYTHON}" -u Run.py > "$rlog" 2>&1 ) &
    echo $! > "${pid_dir}/run.pid"

    log "  Run.py launched (req=${req})"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  All ${#EXPERIMENTS[@]} experiments running in parallel."
echo ""
echo "  Monitor progress (live learning curves):"
for exp_def in "${EXPERIMENTS[@]}"; do
    read -r req _ _ _ <<< "$exp_def"
    echo "    tail -f ${LOG_BASE}/progress_QuRA_Seq_DIST_req${req}.csv"
done
echo ""
echo "  Check status:   bash deploy_parallel.sh status"
echo "  Stop all:       bash deploy_parallel.sh stop"
echo "  Tail run logs:  bash deploy_parallel.sh logs"
echo "═══════════════════════════════════════════════════════════════"
echo ""
