#!/usr/bin/env python3
"""batched (x16, mlx batch_generate) vs unbatched generation, by label.

  .venv/bin/python experiments/activation_pilot/batching_gives_same_output.py   # -> results/activation_pilot/batching_check.{md,jsonl}, batching_run.json

Matched rows: results/e1b/gen_{qwen3.5-9b,aya-expanse-8b}.jsonl (batched) vs *.unbatched.bak (unbatched).
Step 1: exact text equality. Step 2: Qwen3Guard-Gen-4B (official bf16 weights via mlx_lm, greedy, 64 tokens,
batch 16, same as E1b) labels the UNBATCHED text; compared to the stored qg4 label of the batched text.
Rows with identical text are not re-judged (same input, so same label up to judge-side batching).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
R = ROOT / "results" / "e1b"
OUT = ROOT / "results" / "activation_pilot"
OUT.mkdir(exist_ok=True)


def jl(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def main() -> None:
    from judge_qwen3guard import MAX_TOKENS, parse
    from mlx_lm import load
    from multilingual_judge import BATCH, batched
    prompts = {r["row_id"]: r["prompt"] for r in jl(R / "sample.jsonl")}
    qg4 = {(r["model"], r["row_id"]): r["verdict"] for r in jl(R / "judge_qg4.jsonl")}
    pairs = []
    for m in ("qwen3.5-9b", "aya-expanse-8b"):
        b = {r["row_id"]: r["response"] for r in jl(R / f"gen_{m}.jsonl")}
        for r in jl(R / f"gen_{m}.unbatched.bak"):
            if r["row_id"] in b:
                pairs.append(dict(model=m, row_id=r["row_id"], batched=b[r["row_id"]], unbatched=r["response"]))
    for p in pairs:
        p["same_text"] = p["batched"] == p["unbatched"]
        p["label_batched"] = qg4[(p["model"], p["row_id"])]
    todo = [p for p in pairs if not p["same_text"]]
    model, tok = load("Qwen/Qwen3Guard-Gen-4B")
    texts = [tok.apply_chat_template([{"role": "user", "content": prompts[p["row_id"]]}, {"role": "assistant", "content": p["unbatched"]}],
                                     tokenize=False, add_generation_prompt=True) for p in todo]
    for i, raw in batched(model, tok, texts, MAX_TOKENS):
        todo[i]["label_unbatched"], todo[i]["raw_unbatched"] = parse(raw)[0], raw
    for p in pairs:
        p.setdefault("label_unbatched", p["label_batched"])
    with open(OUT / "batching_check.jsonl", "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    md = ["# batched vs unbatched (pilot data, not a finding)\n"]
    for m in ("qwen3.5-9b", "aya-expanse-8b"):
        s = [p for p in pairs if p["model"] == m]
        d = [p for p in s if not p["same_text"]]
        flip = [p for p in d if p["label_unbatched"] != p["label_batched"]]
        md.append(f"- {m}: matched {len(s)}; identical text {len(s) - len(d)}; text differs {len(d)}; label flips among differing {len(flip)}"
                  f" (batched->unbatched: {sorted({(p['label_batched'], p['label_unbatched']) for p in flip})}); overall label agree {len(s) - len(flip)}/{len(s)}")
    md.append("\nFlipped rows (row_id): " + ", ".join(f"{p['model']}:{p['row_id']}" for p in pairs if p["label_unbatched"] != p["label_batched"]))
    (OUT / "batching_check.md").write_text("\n".join(md))
    json.dump(dict(script="experiments/activation_pilot/batching_gives_same_output.py", judge="Qwen/Qwen3Guard-Gen-4B", judge_batch=BATCH,
                   git_hash=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
                   cmd=" ".join(sys.argv)), open(OUT / "batching_run.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
