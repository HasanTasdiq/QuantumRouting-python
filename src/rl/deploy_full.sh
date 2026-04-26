#!/usr/bin/env bash
# =============================================================================
# deploy_full.sh — Full paper run (~36 hours).
#
# Delegates to smoke_test.sh with paper-scale parameters.
# No HTTP servers, no Redis — all training is in-process.
#
# Usage:  bash deploy_full.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

TRAIN_LOAD=100 \
TTIME=10000 \
STEP=1000 \
TIMES=3 \
TRAINING_MODE=paper \
RELIQ_STEPS=500000 \
INFER_LOADS="5,10,25,50,75,100" \
MODEL_DIR="/tmp/qrouting_model" \
bash "${SCRIPT_DIR}/smoke_test.sh" "$@"
