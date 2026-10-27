#!/usr/bin/env bash
set -euo pipefail

# Track B Test Split — Overnight Runner
# Runs: Inference (Hindi/Hinglish) → Single-call Judge → Report
# Resume: re-run same command to continue from last saved row

SPLIT="test"
LOG_DIR="logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/track_b_${SPLIT}_${TIMESTAMP}.log"

mkdir -p "$LOG_DIR"

log() {
    local msg="[$(date '+%H:%M:%S')] $1"
    echo "$msg" | tee -a "$LOG_FILE"
}

run() {
    local name=$1; shift
    log ""
    log "=== $name ==="
    python "$@" 2>&1 | tee -a "$LOG_FILE"
    local code=${PIPESTATUS[0]}
    if [ $code -ne 0 ]; then
        log "!!! $name FAILED (exit $code) — re-run to resume !!!"
        exit $code
    fi
    log "$name done."
}

log "=== TRACK B TEST SPLIT — OVERNIGHT RUN ==="
log "Split: $SPLIT | Log: $LOG_FILE | Started: $(date)"

run "STEP 2: Inference (Hindi/Hinglish, ~4.5h)" \
    scripts/track_b_step2_inference.py --split "$SPLIT"

run "STEP 4: Single-call Judge (no translate-back, ~3.5h)" \
    scripts/track_b_step4_judge.py --split "$SPLIT" --no-translate-back

run "STEP 5: Report" \
    scripts/track_b_step5_report.py --split "$SPLIT"

log ""
log "=== ALL DONE ==="
log "Results: results/track_b_raw_${SPLIT}.csv, results/track_b_judgments_${SPLIT}.csv, results/track_b_report_${SPLIT}.md"
log "Finished: $(date)"