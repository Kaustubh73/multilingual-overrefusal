#!/usr/bin/env bash
# Mac qualitative pilot: 16 prompts × Qwen + Llama + Gemma + Aya (Q4_K_M, Metal).
# Fresh out dir: results/qualitative_pilot_mac (does not overwrite minty run).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$ROOT/results/qualitative_pilot_mac"
PROMPTS_SRC="$ROOT/results/qualitative_pilot/prompts.csv"
LOG="$OUT/mac_run.log"
mkdir -p "$OUT"
cd "$ROOT"

if [[ ! -f "$PROMPTS_SRC" ]]; then
  echo "Missing $PROMPTS_SRC — export prompts first" >&2
  exit 1
fi
cp -n "$PROMPTS_SRC" "$OUT/prompts.csv" 2>/dev/null || cp "$PROMPTS_SRC" "$OUT/prompts.csv"

# Prefer project venv, else create with uv
if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PY="$ROOT/.venv/bin/python"
elif command -v uv >/dev/null 2>&1; then
  uv venv "$ROOT/.venv"
  PY="$ROOT/.venv/bin/python"
  uv pip install --python "$PY" pandas pyarrow huggingface_hub llama-cpp-python
else
  PY="$(command -v python3)"
  "$PY" -c 'import llama_cpp, pandas, huggingface_hub' 2>/dev/null || {
    echo "Need llama-cpp-python + pandas. On Mac:" >&2
    echo "  brew install cmake" >&2
    echo "  CMAKE_ARGS=\"-DGGML_METAL=on\" pip install llama-cpp-python pandas pyarrow huggingface_hub" >&2
    exit 1
  }
fi

export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
if [[ -f "$HOME/.cache/huggingface/token" && -z "${HF_TOKEN:-}" ]]; then
  export HF_TOKEN
  HF_TOKEN="$(tr -d '\n' < "$HOME/.cache/huggingface/token")"
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
fi

# Metal: all layers on GPU (-1). Threads leave headroom for UI.
N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
N_THREADS="${N_THREADS:-6}"
N_CTX="${N_CTX:-4096}"

echo "=== mac pilot start $(date -u +%Y-%m-%dT%H:%M:%SZ) py=$PY gpu_layers=$N_GPU_LAYERS ===" | tee -a "$LOG"
nohup "$PY" scripts/run_qualitative_pilot.py \
  --prompts "$OUT/prompts.csv" \
  --out-dir "$OUT" \
  --n-gpu-layers "$N_GPU_LAYERS" \
  --n-threads "$N_THREADS" \
  --n-ctx "$N_CTX" \
  --max-new-tokens 256 \
  >>"$LOG" 2>&1 &
echo "PID=$! log=$LOG out=$OUT"
