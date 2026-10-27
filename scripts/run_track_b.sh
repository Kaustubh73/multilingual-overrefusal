#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Track B Full Pipeline — Hindi & Hinglish OR Replication
# ============================================================
# Steps:
#   0. Pre-translate benchmark source texts (Hindi/Hinglish)
#   1. Build 810-row catalog (uses pre-translated texts)
#   2. Inference on Hindi/Hinglish rows (Ollama LLaMA)
#   3. Translate-back non-English responses to English
#   4. Ensemble judge (3-call majority vote)
#   5. Generate report
#
# Usage:
#   bash scripts/run_track_b.sh              # Full pipeline
#   bash scripts/run_track_b.sh --step 3     # Resume from step 3
#   bash scripts/run_track_b.sh --steps 0,1  # Run only steps 0 and 1
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="$PROJECT_DIR/logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/track_b_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

# Parse args
START_STEP=0
STEPS=""
for arg in "$@"; do
    case "$arg" in
        --step) shift; START_STEP=$1; shift;;
        --steps) shift; STEPS=$1; shift;;
    esac
done

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

run_step() {
    local step_num=$1
    local step_name=$2
    local script=$3

    log ""
    log "============================================================"
    log "  TRACK B — STEP $step_num: $step_name"
    log "============================================================"
    log "  Script: $script"
    log "  Log:    $LOG_FILE"
    log "============================================================"
    log ""

    python "$PROJECT_DIR/$script" 2>&1 | tee -a "$LOG_FILE"
    local exit_code=${PIPESTATUS[0]}

    if [ $exit_code -ne 0 ]; then
        log ""
        log "!!! STEP $step_num FAILED (exit code $exit_code) !!!"
        log "!!! Check log: $LOG_FILE !!!"
        log "!!! Fix the issue and re-run with: bash scripts/run_track_b.sh --step $step_num !!!"
        exit $exit_code
    fi

    log ""
    log "Step $step_num completed successfully."
}

log "============================================================"
log "  TRACK B FULL PIPELINE"
log "============================================================"
log "  Project:  $PROJECT_DIR"
log "  Log file: $LOG_FILE"
log "  Started:  $(date)"
log "============================================================"

if [ -n "$STEPS" ]; then
    IFS=',' read -ra STEP_LIST <<< "$STEPS"
    for s in "${STEP_LIST[@]}"; do
        case "$s" in
            0) run_step 0 "TRANSLATE BENCHMARK" "scripts/track_b_step0_translate_benchmark.py";;
            1) run_step 1 "BUILD CATALOG" "scripts/track_b_step1_catalog.py";;
            2) run_step 2 "INFERENCE" "scripts/track_b_step2_inference.py";;
            3) run_step 3 "TRANSLATE-BACK" "scripts/track_b_step3_translate_back.py";;
            4) run_step 4 "ENSEMBLE JUDGE" "scripts/track_b_step4_judge.py";;
            5) run_step 5 "REPORT" "scripts/track_b_step5_report.py";;
        esac
    done
else
    for step in $(seq $START_STEP 5); do
        case $step in
            0) run_step 0 "TRANSLATE BENCHMARK" "scripts/track_b_step0_translate_benchmark.py";;
            1) run_step 1 "BUILD CATALOG" "scripts/track_b_step1_catalog.py";;
            2) run_step 2 "INFERENCE" "scripts/track_b_step2_inference.py";;
            3) run_step 3 "TRANSLATE-BACK" "scripts/track_b_step3_translate_back.py";;
            4) run_step 4 "ENSEMBLE JUDGE" "scripts/track_b_step4_judge.py";;
            5) run_step 5 "REPORT" "scripts/track_b_step5_report.py";;
        esac
    done
fi

log ""
log "============================================================"
log "  ALL STEPS COMPLETE"
log "============================================================"
log "  Log saved: $LOG_FILE"
log "  Finished:  $(date)"
log ""
log "  Results:"
log "    data/track_b/track_b_benchmark_translated.csv"
log "    results/track_b_catalog.csv"
log "    results/track_b_raw.csv"
log "    results/track_b_raw_translated.csv"
log "    results/track_b_judgments.csv"
log "    results/track_b_metrics.csv"
log "    results/track_b_delta_or.csv"
log "    results/track_b_report.md"
log "============================================================"
