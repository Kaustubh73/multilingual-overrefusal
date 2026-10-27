#!/usr/bin/env bash
# Harmful-seed screen on minty (llama.cpp, CPU). Resumable: re-run to continue.
# Launch: tmux new -d -s harmful-screen scripts/run_harmful_screen_minty.sh
set -uo pipefail
ROOT="/home/kaustubhg73/Desktop/Speshal/projects/multilingual-overrefusal"
PY="/home/kaustubhg73/Desktop/Speshal/projects/indoml-datathon-2026/.venv/bin/python"
LOG="$ROOT/runs/harmful-screen-2026-09/minty.log"
MODELS=(meta-llama/Llama-3.1-8B-Instruct Qwen/Qwen3-8B CohereLabs/aya-23-8B google/gemma-2-9b-it)
# All GGUFs are pre-cached; never touch the network mid-run.
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_PROGRESS_BARS=1 TOKENIZERS_PARALLELISM=false
mkdir -p "$(dirname "$LOG")"
cd "$ROOT/scripts"

for stage in stage1 stage2; do
  echo "=== $stage start $(date -u +%FT%TZ) ===" | tee -a "$LOG"
  "$PY" -u screen_harmful.py "$stage" --backend llamacpp --models "${MODELS[@]}" 2>&1 \
    | grep --line-buffered -vE '^(llama_|load|print_info|init_tokenizer|llm_load|ggml_)' | tee -a "$LOG"
  rc=${PIPESTATUS[0]}
  echo "=== $stage end $(date -u +%FT%TZ) exit=$rc ===" | tee -a "$LOG"
  [[ $rc -ne 0 ]] && exit "$rc"
done
touch "$ROOT/runs/harmful-screen-2026-09/DONE_minty"
