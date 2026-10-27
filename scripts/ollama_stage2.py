#!/usr/bin/env python3
"""E3 stage 2: resample the top-N ranked English seeds on the Ollama Gemma models, T=1.0.

Same seeds/ranking as rank_seeds.py (default results/e3/ranked_E_6model.csv). Sample i uses Ollama seed i.
Resumable. Outputs results/e3/stage2.jsonl (same row schema as the old stage2 + judge_qwen3guard e3stage2).
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import ollama

from ollama_screen import OUT_DIR, ROOT, MODELS
from screen_harmful import is_refusal, load_seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranked", default=str(OUT_DIR / "ranked_E_6model.csv"))
    ap.add_argument("--top", type=int, default=50)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--models", nargs="*", default=MODELS)
    args = ap.parse_args()
    with open(args.ranked, encoding="utf-8") as f:
        top = [r["content_id"] for r in csv.DictReader(f)][: args.top]
    by_id = {s["content_id"]: s for s in load_seeds()}
    seeds = [by_id[c] for c in top]
    client = ollama.Client()
    out = OUT_DIR / "stage2.jsonl"
    seen = set()
    if out.exists():
        for l in out.open(encoding="utf-8"):
            if l.strip():
                o = json.loads(l)
                seen.add((o["model_id"], o["content_id"], o["sample"]))
    digests = {m.model: m.digest[:12] for m in client.list().models}
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT_DIR / "stage2_run.json").write_text(json.dumps({
        "stage": "stage2 sampled", "ranked_from": args.ranked, "top_n": args.top, "samples": args.samples,
        "sample_seeds": list(range(args.samples)), "max_tokens": args.max_tokens,
        "options": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": "= sample index"}, "think": False,
        "models": {m: digests.get(m) for m in args.models}, "backend": "ollama Q4_K_M GGUF",
        "git_hash": git, "ts": datetime.now(timezone.utc).isoformat(),
    }, indent=1), encoding="utf-8")

    for model in args.models:
        todo = [(s, k) for s in seeds for k in range(args.samples) if (model, s["content_id"], k) not in seen]
        print(f"=== {model}: {len(todo)} generations", flush=True)
        with out.open("a", encoding="utf-8") as f:
            for i, (s, k) in enumerate(todo, 1):
                t0 = time.time()
                kw = dict(model=model, messages=[{"role": "user", "content": s["prompt"]}],
                          options={"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": k, "num_predict": args.max_tokens})
                try:
                    r = client.chat(think=False, **kw)
                except ollama.ResponseError:
                    r = client.chat(**kw)
                text = r.message.content or ""
                f.write(json.dumps({
                    "content_id": s["content_id"], "model_id": model, "sample": k, "backend": "ollama",
                    "prompt": s["prompt"], "response": text, "refusal": is_refusal(text.replace("*", "")),
                    "latency_s": round(time.time() - t0, 3),
                }, ensure_ascii=False) + "\n")
                f.flush()
                if i % 25 == 0 or i == len(todo):
                    print(f"{model} {i}/{len(todo)}", flush=True)


if __name__ == "__main__":
    main()
