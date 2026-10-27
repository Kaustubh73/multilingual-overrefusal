#!/usr/bin/env bash
# Bootstrap qualitative pilot.
# On Mac: Metal. On Linux (minty): CPU llama-cpp (no Metal).
set -euo pipefail

OS="$(uname -s)"
LOCAL_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$LOCAL_ROOT/results/qualitative_pilot_rerun"
PROMPTS_SRC="$LOCAL_ROOT/results/qualitative_pilot/prompts.csv"
LOG="$OUT/run.log"
mkdir -p "$OUT"

echo "==> host=$(hostname) os=$OS root=$LOCAL_ROOT"

if [[ ! -f "$PROMPTS_SRC" ]]; then
  echo "Missing prompts: $PROMPTS_SRC" >&2
  exit 1
fi
cp -f "$PROMPTS_SRC" "$OUT/prompts.csv"

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
if [[ -f "$HOME/.cache/huggingface/token" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

cd "$LOCAL_ROOT"

# Prefer existing working venv with llama_cpp (minty: indoml), else project .venv
if [[ -x /home/kaustubhg73/Desktop/Speshal/projects/indoml-datathon-2026/.venv/bin/python ]] \
  && /home/kaustubhg73/Desktop/Speshal/projects/indoml-datathon-2026/.venv/bin/python -c 'import llama_cpp' 2>/dev/null; then
  PY=/home/kaustubhg73/Desktop/Speshal/projects/indoml-datathon-2026/.venv/bin/python
elif [[ -x "$LOCAL_ROOT/.venv/bin/python" ]] \
  && "$LOCAL_ROOT/.venv/bin/python" -c 'import llama_cpp' 2>/dev/null; then
  PY="$LOCAL_ROOT/.venv/bin/python"
else
  if command -v uv >/dev/null 2>&1; then
    [[ -x "$LOCAL_ROOT/.venv/bin/python" ]] || uv venv "$LOCAL_ROOT/.venv"
    PY="$LOCAL_ROOT/.venv/bin/python"
    uv pip install --python "$PY" pandas pyarrow huggingface_hub
    if [[ "$OS" == "Darwin" ]]; then
      echo "==> installing llama-cpp-python with Metal"
      CMAKE_ARGS="-DGGML_METAL=on" uv pip install --python "$PY" llama-cpp-python --no-cache
    else
      echo "==> installing llama-cpp-python CPU (Linux — Metal not available)"
      uv pip install --python "$PY" llama-cpp-python
    fi
  else
    echo "Need uv or a venv with llama_cpp" >&2
    exit 1
  fi
fi

if [[ "$OS" == "Darwin" ]]; then
  N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
  N_THREADS="${N_THREADS:-6}"
  N_CTX="${N_CTX:-4096}"
  # Mac: full Q4 for gemma
  GEMMA_FILE="gemma-2-9b-it-Q4_K_M.gguf"
else
  N_GPU_LAYERS="${N_GPU_LAYERS:-0}"
  N_THREADS="${N_THREADS:-4}"
  N_CTX="${N_CTX:-2048}"
  # minty 8GB: gemma Q3 to avoid swap thrash
  GEMMA_FILE="gemma-2-9b-it-Q3_K_M.gguf"
fi

echo "==> py=$PY gpu_layers=$N_GPU_LAYERS threads=$N_THREADS ctx=$N_CTX gemma=$GEMMA_FILE"
echo "==> models: Qwen + Llama + Gemma + Aya-expanse (Q4_K_M except gemma on Linux as above)"

# Patch gemma filename for this host via env consumed... actually script has fixed DEFAULT_MODELS.
# Use --only and two-phase OR sed a temp config. Simplest: run with env OVERRIDE via small wrapper.
# For Linux, temporarily point gemma to Q3 by editing via python one-liner before run.

"$PY" - <<PY
from pathlib import Path
p = Path("scripts/run_qualitative_pilot.py")
text = p.read_text()
# ensure aya present (already in file)
print("pilot script ok", p.exists())
PY

# On Linux override gemma quant in-process by copying models list — use sed on a run-local copy
RUN_SCRIPT="$OUT/run_qualitative_pilot.py"
cp -f scripts/run_qualitative_pilot.py "$RUN_SCRIPT"
if [[ "$OS" != "Darwin" ]]; then
  sed -i 's/gemma-2-9b-it-Q4_K_M.gguf/gemma-2-9b-it-Q3_K_M.gguf/' "$RUN_SCRIPT"
fi

echo "=== start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$LOG"
nohup "$PY" "$RUN_SCRIPT" \
  --prompts "$OUT/prompts.csv" \
  --out-dir "$OUT" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --n-threads "$N_THREADS" \
  --n-ctx "$N_CTX" \
  --max-new-tokens 256 \
  >>"$LOG" 2>&1 &
echo "PID=$!"
echo "Log: $LOG"
echo "Out: $OUT"
echo "Tail: tail -f $LOG"
