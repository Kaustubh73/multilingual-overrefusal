#!/usr/bin/env python3
"""E3 stage 1: screen the 861 English harmful seeds on new models via Ollama (greedy).

Same seeds, 128 tokens and keyword refusal screen as scripts/screen_harmful.py stage1,
so results merge with runs/harmful-screen-2026-09/stage1.jsonl (Llama, Qwen3, Aya, Gemma-2).
Backend differs (Ollama Q4_K_M GGUF vs earlier MLX 4-bit): noted in run.json.
Outputs go to results/e3/, resumable.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import ollama

from screen_harmful import is_refusal, load_seeds

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "e3"
MODELS = ["gemma3:12b", "gemma4:e4b"]


def done(path: Path) -> set[tuple[str, str]]:
    if not path.exists():
        return set()
    return {(o["model_id"], o["content_id"]) for o in map(json.loads, filter(str.strip, path.open(encoding="utf-8")))}


def gen(client, model: str, prompt: str, max_tokens: int) -> str:
    kw = dict(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0, "seed": 0, "num_predict": max_tokens, "top_k": 1},
    )
    try:
        r = client.chat(think=False, **kw)
    except ollama.ResponseError:
        r = client.chat(**kw)
    return r.message.content or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = load_seeds()
    if args.limit:
        seeds = seeds[: args.limit]
    client = ollama.Client()
    out = OUT_DIR / ("stage1_test.jsonl" if args.limit else "stage1.jsonl")
    seen = done(out)

    digests = {m.model: m.digest[:12] for m in client.list().models}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT_DIR / "run.json").write_text(json.dumps({
        "stage": "stage1 greedy", "seed": 0, "n_seeds": len(seeds), "max_tokens": args.max_tokens,
        "options": {"temperature": 0, "seed": 0, "top_k": 1}, "think": False,
        "models": {m: digests.get(m) for m in args.models}, "backend": "ollama Q4_K_M GGUF",
        "ollama_version": ollama.__version__ if hasattr(ollama, "__version__") else None,
        "git_hash": git, "command": f"scripts/ollama_screen.py {' '.join(__import__('sys').argv[1:])}",
        "ts": datetime.now(timezone.utc).isoformat(),
    }, indent=1), encoding="utf-8")

    for model in args.models:
        todo = [s for s in seeds if (model, s["content_id"]) not in seen]
        print(f"=== {model}: {len(todo)} generations", flush=True)
        with out.open("a", encoding="utf-8") as f:
            for i, s in enumerate(todo, 1):
                t0 = time.time()
                text = gen(client, model, s["prompt"], args.max_tokens)
                f.write(json.dumps({
                    "content_id": s["content_id"], "model_id": model, "sample": None, "backend": "ollama",
                    # Markdown bold ("I **cannot**") breaks the regex, so strip asterisks for "refusal".
                    "prompt": s["prompt"], "response": text, "refusal": is_refusal(text.replace("*", "")),
                    "refusal_raw": is_refusal(text),
                    "latency_s": round(time.time() - t0, 3),
                }, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"{model} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
