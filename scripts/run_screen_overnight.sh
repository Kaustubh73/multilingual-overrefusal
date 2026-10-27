#!/bin/bash
# Sequential E3 chain: finish stage 1 -> rank -> stage 2 (Gemma3/4, T=1.0) -> Qwen3Guard judge.
# Each step starts only after the previous one exits; stops on first hard failure. Resumable throughout.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/scripts"
LOG="$ROOT/results/e3/overnight.log"
say() { echo "=== $* $(date '+%F %T %Z')" | tee -a "$LOG"; }

say "wait for running stage 1 (pid ${1:-none})"
[ -n "${1:-}" ] && while kill -0 "$1" 2>/dev/null; do sleep 30; done

s1_done() { python3 - <<'E'
import json,collections
c=collections.Counter(json.loads(l)["model_id"] for l in open("../results/e3/stage1.jsonl") if l.strip())
raise SystemExit(0 if c["gemma3:12b"]>=861 and c["gemma4:e4b"]>=861 else 1)
E
}
for i in 1 2 3; do s1_done && break; say "stage 1 incomplete, resume try $i"; python3 ollama_screen.py >> "$LOG" 2>&1; done
s1_done || { say "FAIL stage 1 incomplete"; exit 1; }

say "rank"; python3 rank_seeds.py >> "$LOG" 2>&1 || { say "FAIL rank"; exit 1; }
[ -s ../results/e3/ranked_E_6model.csv ] || { say "FAIL no ranked_E_6model.csv"; exit 1; }

say "stage 2 generation"
for i in 1 2 3; do python3 ollama_stage2.py >> "$LOG" 2>&1 && break; say "stage 2 retry $i"; done
[ "$(wc -l < ../results/e3/stage2.jsonl)" -ge 1000 ] || { say "FAIL stage 2 incomplete"; exit 1; }

say "judge (Qwen3Guard-Gen-4B, MLX)"
for i in 1 2 3; do ../.venv/bin/python judge_qwen3guard.py e3stage2 >> "$LOG" 2>&1 && break; say "judge retry $i"; done
say "E3 CHAIN COMPLETE"
