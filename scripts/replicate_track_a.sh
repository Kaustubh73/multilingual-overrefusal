#!/usr/bin/env bash
set -euo pipefail

# Track A replication pipeline
# Usage: bash scripts/replicate_track_a.sh [flags]
#
# Steps:
#   1. Calibrate judge against golden set (scripts/calibrate_judge.py)
#   2. Run HF inference on 270 test rows → results/paper_raw.csv
#   3. Ensemble-judge all rows → results/paper_judgments.csv
#   4. Compute OR metrics → results/replication_report.md

EXTRA_FLAGS=("$@")

echo "=== Step 1: Calibrate ensemble judge ==="
python scripts/calibrate_judge.py "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"

echo ""
echo "=== Step 2: Run replicate pipeline (inference + ensemble judge + report) ==="
python main.py replicate --ensemble "${EXTRA_FLAGS[@]+"${EXTRA_FLAGS[@]}"}"

echo ""
echo "=== Done ==="
echo "Outputs:"
echo "  results/paper_raw.csv          - HF model responses"
echo "  results/paper_judgments.csv    - Ensemble judge labels"
echo "  results/replication_report.md  - Full replication report"
echo "  results/paper_metrics.csv      - Per-slice OR metrics"
