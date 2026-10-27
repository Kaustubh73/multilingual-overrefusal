#!/usr/bin/env bash
# Overnight qualitative pilot on minty (16 prompts × 3 GGUF instruct models).
set -euo pipefail
ROOT="/home/kaustubhg73/Desktop/Speshal/projects/multilingual-overrefusal"
PY="/home/kaustubhg73/Desktop/Speshal/projects/indoml-datathon-2026/.venv/bin/python"
OUT="$ROOT/results/qualitative_pilot"
LOG="$OUT/overnight.log"
mkdir -p "$OUT"
cd "$ROOT"
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER=0
# Prefer HF token from cache
if [[ -f "$HOME/.cache/huggingface/token" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi
echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" >>"$LOG"
"$PY" scripts/run_qualitative_pilot.py \
  --n-gpu-layers 0 \
  --n-threads 4 \
  --max-new-tokens 256 \
  >>"$LOG" 2>&1
echo "=== end $(date -u +%Y-%m-%dT%H:%M:%SZ) exit=$? ===" >>"$LOG"
