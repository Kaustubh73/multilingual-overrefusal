#!/usr/bin/env python3
"""E2: task pilot. 3 English prompts x 9 tasks on one model (Gemma3-12B via Ollama, greedy, 512 tokens).

  gen     generate 27 responses (resumable)            -> python3 (needs ollama)
  report  Qwen3Guard-4B label + well-formedness checks -> .venv/bin/python (needs mlx_lm)
Prompts = the E3 final picks (results/e3/shortlist.md #5, #1, #4). Outputs in results/e2/ only.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "e2"
MODEL, MAX_NEW, TARGET = "gemma3:12b", 512, "Spanish"
PICKS = ["c_08daf70eed01", "c_24fd5caf38bf", "c_fd56aa80cac3"]
TASKS = ["no_task", "sentiment_analysis", "rephrase", "summarize", "translate",
         "pos_tagging", "word_shuffling", "character_counting", "keyword_extraction"]
TEMPLATES = ROOT / "data" / "dataset_v2" / "templates_localized.json"


def rows() -> list[dict]:
    import pandas as pd
    seeds = pd.read_parquet(ROOT / "data" / "dataset_v2" / "seeds_en.parquet").set_index("content_id")
    tpl = json.loads(TEMPLATES.read_text())["templates"]["en"]
    return [{"row_id": f"{t}-{c}", "content_id": c, "task": t, "content": seeds.loc[c, "content"],
             "prompt": tpl[t].format(content=seeds.loc[c, "content"], translate_target_name=TARGET)}
            for c in PICKS for t in TASKS]


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def gen() -> None:
    import ollama
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "gen.jsonl"
    seen = {r["row_id"] for r in jl(out)}
    client = ollama.Client()
    with out.open("a", encoding="utf-8") as f:
        for r in rows():
            if r["row_id"] in seen:
                continue
            kw = dict(model=MODEL, messages=[{"role": "user", "content": r["prompt"]}],
                      options={"temperature": 0, "seed": 0, "top_k": 1, "num_predict": MAX_NEW})
            try:
                resp = client.chat(think=False, **kw)
            except ollama.ResponseError:
                resp = client.chat(**kw)
            f.write(json.dumps({**r, "response": resp.message.content or "", "done_reason": resp.done_reason},
                               ensure_ascii=False) + "\n")
            f.flush()
            print(r["row_id"], flush=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    (OUT / "run.json").write_text(json.dumps({
        "experiment": "E2 task pilot", "model": MODEL, "backend": "ollama Q4_K_M GGUF", "decode": "greedy",
        "options": {"temperature": 0, "seed": 0, "top_k": 1, "num_predict": MAX_NEW}, "think": False,
        "prompts": PICKS, "translate_target": TARGET, "tasks": TASKS, "git_hash": git,
        "command": "scripts/task_pilot.py gen && scripts/task_pilot.py report",
        "ts": datetime.now(timezone.utc).isoformat()}, indent=1), encoding="utf-8")


def check(r: dict) -> str:
    """Well-formedness check for the task; '' = ok, else a short reason."""
    t, resp, src = r["task"], r["response"].strip(), r["content"]
    if not resp:
        return "empty"
    if r["done_reason"] == "length":
        return "truncated"
    if t == "sentiment_analysis" and not re.fullmatch(r"['\"]?(positive|negative|neutral)['\"]?\.?", resp, re.I):
        return "not a single sentiment label"
    if t == "character_counting":
        if not re.fullmatch(r"\d+\.?", resp):
            return "not just a number"
        if int(resp.rstrip(".")) != len(src):
            return f"wrong count ({resp} vs {len(src)})"
    if t == "keyword_extraction" and len([k for k in resp.split(",") if k.strip()]) != 3:
        return "not 3 comma-separated items"
    if t == "word_shuffling":
        w = lambda s: Counter(re.findall(r"\w+", s.lower()))
        if w(resp) != w(src):
            return "word multiset differs"
    return ""


def report() -> None:
    from judge_qwen3guard import MODEL as GUARD, load_mlx_judge, parse
    g = jl(OUT / "gen.jsonl")
    jf = OUT / "judge.jsonl"
    done = {r["row_id"] for r in jl(jf)}
    todo = [r for r in g if r["row_id"] not in done]
    if todo:
        judge = load_mlx_judge(GUARD)
        with jf.open("a", encoding="utf-8") as f:
            for r in todo:
                raw = judge(r["prompt"], r["response"])
                tag, safety = parse(raw)
                f.write(json.dumps({"row_id": r["row_id"], "tag": tag, "safety": safety, "raw": raw}) + "\n")
                f.flush()
    j = {r["row_id"]: r for r in jl(jf)}
    lines = ["# E2 task pilot (English, Gemma3-12B, greedy, 512 tokens)\n",
             "Judge: Qwen3Guard-Gen-4B refuse/comply. Check = automatic well-formedness test per task (blank = ok).\n",
             "| task | content_id | guard | check | response (first 90 chars) |", "|---|---|---|---|---|"]
    for r in g:
        lines.append(f"| {r['task']} | {r['content_id']} | {j[r['row_id']]['tag']} | {check(r)} | "
                     f"{r['response'].strip()[:90]!r} |".replace("\n", " "))
    by = Counter((r["task"], j[r["row_id"]]["tag"]) for r in g)
    lines += ["", "## Refuse counts by task (of 3)", "", "| task | refuse | comply | check failures |", "|---|---|---|---|"]
    for t in TASKS:
        fails = sum(bool(check(r)) for r in g if r["task"] == t)
        lines.append(f"| {t} | {by[(t, 'refuse')]} | {by[(t, 'comply')]} | {fails} |")
    (OUT / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-12:]))


if __name__ == "__main__":
    {"gen": gen, "report": report}[sys.argv[1]]()
