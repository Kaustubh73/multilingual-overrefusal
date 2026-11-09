#!/usr/bin/env python3
"""response length (chars, Qwen tokens) and token-limit hits by task and language, vs judge labels.

  python experiments/activation_pilot/response_length_and_truncation.py   # -> results/activation_pilot/response_length.{csv,md}, run.json

Sources: results/e2/gen.jsonl (Gemma3-12B, English, done_reason), results/e1b/gen_*.jsonl (4 models,
lo/mt/ur/yo, no done_reason: a response of >=500 Qwen tokens is used as a proxy for the 512 cap).
Descriptive only. Re-run on the pilot output once it exists.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
OUT = R / "activation_pilot"
OUT.mkdir(exist_ok=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-9B")


def table(t: pd.DataFrame) -> str:
    t = t.reset_index()
    h = "| " + " | ".join(map(str, t.columns)) + " |\n|" + "---|" * len(t.columns) + "\n"
    return h + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in t.values)


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


rows = []
e2j = {j["row_id"]: j["tag"] for j in jl(R / "e2" / "judge.jsonl")}
for g in jl(R / "e2" / "gen.jsonl"):
    rows.append(dict(src="e2", model="gemma3-12b-q4", lang="en", task=g["task"], row_id=g["row_id"],
                     resp=g["response"], hit=g.get("done_reason") == "length", judge=e2j.get(g["row_id"])))
for f in sorted((R / "e1b").glob("gen_*.jsonl")):
    m = f.stem[4:]
    j = {x["row_id"]: x["verdict"] for x in jl(R / "e1b" / "judge_qg4.jsonl") if x["model"] == m}
    for g in jl(f):
        lang, task = g["row_id"].split("-", 1)[0], g["row_id"].split("-")[1]
        rows.append(dict(src="e1b", model=m, lang=lang, task=task, row_id=g["row_id"],
                         resp=g["response"], hit=None, judge=j.get(g["row_id"])))
df = pd.DataFrame(rows)
df["chars"] = df.resp.str.len()
df["tokens"] = [len(tok(t, add_special_tokens=False)["input_ids"]) for t in df.resp]
df["hit"] = df.apply(lambda r: r.hit if r.hit is not None else r.tokens >= 500, axis=1)
df["hit_is_proxy"] = df.src == "e1b"
df.drop(columns="resp").to_csv(OUT / "response_length.csv", index=False)

md = ["# response length (descriptive, pilot data, not a finding)\n",
      "Tokens = Qwen3.5 tokenizer on the response text. `hit` = done_reason=='length' (e2) or >=500 tokens (e1b proxy).\n"]
for src, sub in df.groupby("src"):
    md.append(f"\n## {src} (n={len(sub)})\n")
    for by in ["task", "lang", "model"]:
        t = sub.groupby(by).agg(n=("tokens", "size"), med_chars=("chars", "median"), med_tok=("tokens", "median"),
                                p90_tok=("tokens", lambda s: s.quantile(.9)), hit_rate=("hit", "mean"),
                                refuse_rate=("judge", lambda s: (s == "refuse").mean())).round(2)
        md.append(f"\n### by {by}\n\n" + table(t) + "\n")
    t = sub.groupby("judge").agg(n=("tokens", "size"), med_tok=("tokens", "median"), hit_rate=("hit", "mean")).round(2)
    md.append("\n### length by judge label (qg4 for e1b, qwen3guard for e2)\n\n" + table(t) + "\n")
(OUT / "response_length.md").write_text("\n".join(md))
json.dump(dict(script="experiments/activation_pilot/response_length_and_truncation.py", tokenizer="Qwen/Qwen3.5-9B",
               git_hash=subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip(),
               cmd=" ".join(sys.argv), n=len(df)), open(OUT / "response_length_run.json", "w"), indent=1)
print(len(df), "rows ->", OUT / "response_length.md")
