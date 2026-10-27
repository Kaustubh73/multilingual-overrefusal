#!/bin/bash
# Mac half of stage 2: Qwen3, Aya, Gemma on MLX, then the policy-v1 gpt-oss
# judge on every landed row (Mac 1,500, then minty's 500 once that file hits 500).
# Llama stage 2 runs on minty at the same time (scripts/run_stage2_minty.sh).
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
RUN="$ROOT/runs/harmful-screen-2026-09"
LOG="$RUN/stage2_mac.log"
MINTY="${MINTY_HOST:?set MINTY_HOST, e.g. user@host}"
REMOTE="Desktop/Speshal/projects/multilingual-overrefusal/runs/harmful-screen-2026-09/stage2_minty.jsonl"
cd "$ROOT/scripts"
mkdir -p "$RUN"

echo "=== mac stage2 start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
"$PY" -u screen_harmful.py stage2 \
  --models Qwen/Qwen3-8B CohereLabs/aya-23-8B google/gemma-2-9b-it \
  --top 50 --samples 10 --out stage2_mac.jsonl 2>&1 | tee -a "$LOG"

echo "=== mac generations done; judging mac rows while minty may still be running ===" | tee -a "$LOG"
"$PY" -u judge_stage2.py stage2_mac.jsonl 2>&1 | tee -a "$LOG"

echo "=== waiting for minty llama stage2 (500 rows) ===" | tee -a "$LOG"
until ssh -o BatchMode=yes -o ConnectTimeout=10 "$MINTY" \
    "test \$(wc -l < ~/$REMOTE) -ge 500" 2>/dev/null; do
  sleep 60
done
rsync -az "$MINTY:$REMOTE" "$RUN/stage2_minty.jsonl"
echo "=== judging minty rows ===" | tee -a "$LOG"
"$PY" -u judge_stage2.py stage2_mac.jsonl stage2_minty.jsonl 2>&1 | tee -a "$LOG"
echo "=== STAGE2 COMPLETE $(date -u +%FT%TZ) ===" | tee -a "$LOG"
