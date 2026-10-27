#!/usr/bin/env python3
"""Run covering smoke generations with mlx-lm (greedy decode)."""
from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "smoke-cover-2026-09"
IDS = RUN / "smoke_cover_ids.csv"
OUT = RUN / "responses.jsonl"
MANIFEST = RUN / "manifest.json"

# Prefer pre-converted mlx community 4-bit repos
MODELS = [
    {
        "id": "Qwen/Qwen3-8B",
        "mlx": "Qwen/Qwen3-8B-MLX-4bit",
    },
    {
        "id": "google/gemma-2-9b-it",
        "mlx": "mlx-community/gemma-2-9b-it-4bit",
    },
    {
        "id": "CohereLabs/aya-23-8B",
        "mlx": "mlx-community/aya-23-8B-4bit",
    },
    {
        "id": "meta-llama/Llama-3.1-8B-Instruct",
        "mlx": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    },
]


def load_prompts():
    with IDS.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def already_done(path: Path) -> set[tuple[str, str]]:
    done = set()
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            o = json.loads(line)
            done.add((o["model_id"], str(o["idx"])))
    return done


def build_messages(prompt: str) -> list[dict]:
    return [{"role": "user", "content": prompt}]


def main():
    from mlx_lm import generate, load

    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", help="Subset of HF model ids to run")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0, help="Max prompts per model (0=all)")
    args = ap.parse_args()

    prompts = load_prompts()
    if args.limit:
        prompts = prompts[: args.limit]

    models = MODELS
    if args.models:
        want = set(args.models)
        models = [m for m in MODELS if m["id"] in want or m["mlx"] in want]

    done = already_done(OUT)
    RUN.mkdir(parents=True, exist_ok=True)

    manifest = {
        "run_id": "smoke-cover-2026-09",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "max_tokens": args.max_tokens,
        "temperature": 0.0,
        "system_prompt": None,
        "ids_file": str(IDS),
        "models": [m["id"] for m in models],
        "mlx_models": [m["mlx"] for m in models],
        "decode": "greedy",
        "enable_thinking": False,
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for m in models:
        model_id = m["id"]
        mlx_id = m["mlx"]
        print(f"\n=== Loading {mlx_id} ({model_id}) ===", flush=True)
        try:
            model, tokenizer = load(mlx_id)
        except Exception as e:
            print(f"FAIL load {mlx_id}: {e}", flush=True)
            continue

        with OUT.open("a", encoding="utf-8") as fout:
            for row in prompts:
                key = (model_id, str(row["idx"]))
                if key in done:
                    print(f"skip {model_id} idx={row['idx']}", flush=True)
                    continue

                messages = build_messages(row["prompt"])
                if hasattr(tokenizer, "apply_chat_template"):
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        # Qwen3 thinks by default; other templates ignore this.
                        enable_thinking=False,
                    )
                else:
                    prompt_text = row["prompt"]

                t0 = time.time()
                try:
                    text = generate(
                        model,
                        tokenizer,
                        prompt=prompt_text,
                        max_tokens=args.max_tokens,
                        verbose=False,
                    )
                    err = None
                except Exception as e:
                    text = ""
                    err = str(e)
                dt = time.time() - t0

                rec = {
                    "idx": int(row["idx"]),
                    "content_id": row["content_id"],
                    "lang": row["lang"],
                    "intended_task": row["intended_task"],
                    "label": row["label"],
                    "en_seed": row["en_seed"],
                    "prompt": row["prompt"],
                    "model_id": model_id,
                    "mlx_id": mlx_id,
                    "response": text,
                    "latency_s": round(dt, 3),
                    "error": err,
                    "ts": datetime.now(timezone.utc).isoformat(),
                }
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                done.add(key)
                status = "ERR" if err else "OK"
                print(
                    f"{status} {model_id} idx={row['idx']} lang={row['lang']} "
                    f"{dt:.1f}s chars={len(text)}",
                    flush=True,
                )

    print(f"\nDone. Responses -> {OUT}")


if __name__ == "__main__":
    main()
