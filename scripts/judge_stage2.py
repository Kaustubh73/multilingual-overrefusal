#!/usr/bin/env python3
"""Judge stage-2 responses with the local gpt-oss model. Mac / MLX only.

Same label set and prompt as scripts/judge_gpt_oss.py. Resumable. This is the
judge used for reported numbers; the regex in screen_harmful.py only screens.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from mlx_lm import generate, load

from judge_gpt_oss import (
    DEFAULT_MAX_TOKENS,
    MODEL,
    PROMPT_REFS,
    PROMPT_VERSION,
    REASONING_EFFORT,
    SYSTEM,
    parse_tag,
    user_prompt,
)

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "harmful-screen-2026-09"


def load_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                o = json.loads(line)
                o["row_id"] = f"{o['model_id']}|{o['content_id']}|{o['sample']}"
                rows.append(o)
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="*", help="stage2 jsonl files (default: mac + minty)")
    ap.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    args = ap.parse_args()
    paths = [Path(p) if Path(p).is_absolute() else RUN / p for p in args.inputs] or [
        RUN / "stage2_mac.jsonl", RUN / "stage2_minty.jsonl",
    ]
    out = RUN / "stage2_judge.jsonl"
    rows = load_rows(paths)
    # done_ids double-parses; fine at this size
    seen: set[str] = set()
    if out.exists():
        with out.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if rec.get("tag"):
                        seen.add(rec["row_id"])
    pending = [r for r in rows if r["row_id"] not in seen]
    print(f"rows={len(rows)} done={len(seen)} pending={len(pending)} model={MODEL}", flush=True)
    if not pending:
        return
    model, tokenizer = load(MODEL)
    manifest = {
        "model": MODEL,
        "prompt_version": PROMPT_VERSION,
        "system_prompt": SYSTEM,
        "references": PROMPT_REFS,
        "max_tokens": args.max_tokens,
        "reasoning_effort": REASONING_EFFORT,
        "truncation_chars": 1800,
        "labels": ["comply", "refuse", "garble"],
        "started_at": datetime.now(timezone.utc).isoformat(),
        "inputs": [str(p) for p in paths],
    }
    (RUN / "stage2_judge_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    with out.open("a", encoding="utf-8") as fout:
        for i, row in enumerate(pending, 1):
            user = user_prompt(row)
            prompt = tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                tokenize=False, add_generation_prompt=True, reasoning_effort=REASONING_EFFORT,
            )
            t0 = time.time()
            raw = generate(model, tokenizer, prompt=prompt, max_tokens=args.max_tokens, verbose=False)
            rec = {
                "row_id": row["row_id"], "model_id": row["model_id"], "content_id": row["content_id"],
                "sample": row["sample"], "tag": parse_tag(raw), "raw": raw,
                "prompt_version": PROMPT_VERSION,
                "latency_s": round(time.time() - t0, 3), "ts": datetime.now(timezone.utc).isoformat(),
            }
            fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
            fout.flush()
            if i % 25 == 0 or i == len(pending):
                print(f"{i}/{len(pending)} {row['row_id']} {rec['tag']} {rec['latency_s']}s", flush=True)
    print(f"-> {out}", flush=True)


if __name__ == "__main__":
    main()
