#!/usr/bin/env python3
"""E3 stage 1 ranking: merge results/e3/stage1.jsonl (Gemma3-12B, Gemma4-E4B via Ollama) with the old
screen (runs/harmful-screen-2026-09, read-only) and rank seeds by k*(n-k) disagreement.

Scoring mirrors scripts/screen_report.py: ties -> fewer hedges -> seed-0 shuffle of the sorted seed list.
The refusal regex is a screen for picking seeds, not a reported rate. Writes to results/e3/ only.
"""
from __future__ import annotations

import csv
import json
import random
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from screen_report import HEDGE_RE, refuse_fixed

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "runs" / "harmful-screen-2026-09"
OUT = ROOT / "results" / "e3"
NEW_FILE = OUT / "stage1.jsonl"

LLAMA, QWEN, AYA, G2 = ("meta-llama/Llama-3.1-8B-Instruct", "Qwen/Qwen3-8B", "CohereLabs/aya-23-8B", "google/gemma-2-9b-it")
G3, G4 = "gemma3:12b", "gemma4:e4b"
SHORT = {LLAMA: "Llama-3.1-8B", QWEN: "Qwen3-8B", AYA: "Aya-23-8B", G2: "Gemma-2-9B (MLX)", G3: "Gemma3-12B", G4: "Gemma4-E4B"}
# 5-model = Llama, Qwen3, Aya, Gemma-2 + one new Gemma; one variant per new model, plus all six.
VARIANTS = {
    "A_gemma3": [LLAMA, QWEN, AYA, G3],
    "B_gemma4": [LLAMA, QWEN, AYA, G4],
    "C_5model_g3": [LLAMA, QWEN, AYA, G2, G3],
    "D_5model_g4": [LLAMA, QWEN, AYA, G2, G4],
    "E_6model": [LLAMA, QWEN, AYA, G2, G3, G4],
}


def read(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]


def main():
    sys.path.insert(0, str(ROOT / "scripts"))
    from screen_harmful import load_seeds
    seeds = [s["content_id"] for s in load_seeds()]
    prompts = {s["content_id"]: s["prompt"] for s in load_seeds()}

    # ---- verification of new rows
    new = read(NEW_FILE)
    problems = []
    for m in (G3, G4):
        rows = [o for o in new if o["model_id"] == m]
        keys = [(o["model_id"], o["content_id"]) for o in rows]
        empty = sum(not (o["response"] or "").strip() for o in rows)
        print(f"verify {m}: rows={len(rows)} dup={len(keys) - len(set(keys))} empty={empty} "
              f"missing={len(set(seeds) - {o['content_id'] for o in rows})}")
        if len(rows) != 861 or len(keys) != len(set(keys)) or empty:
            problems.append(m)
    dups_all = len(new) - len({(o["model_id"], o["content_id"]) for o in new})
    print("total rows", len(new), "duplicates", dups_all)
    if problems:
        sys.exit(f"verification failed: {problems}")

    # ---- unified table: (model, cid) -> (refuse bool, hedge bool, response)
    tab: dict[tuple[str, str], dict] = {}
    for o in read(OLD / "minty_llamacpp" / "stage1.jsonl"):
        if o["model_id"] in (LLAMA, QWEN, AYA):
            tab[(o["model_id"], o["content_id"])] = o
    for o in read(OLD / "stage1.jsonl"):
        if o["model_id"] == G2 and o["backend"] == "mlx":
            tab[(o["model_id"], o["content_id"])] = o
    for o in new:
        tab[(o["model_id"], o["content_id"])] = o

    def refuse(o):  # new models: recorded "refusal" (asterisks stripped); old: same recompute as screen_report
        return o["refusal"] if o["backend"] == "ollama" else refuse_fixed(o["response"])

    def hedge(o):
        return (not refuse(o)) and bool(HEDGE_RE.search(o["response"].replace("*", "")))

    assert all((m, c) in tab for m in SHORT for c in seeds), "missing (model, seed) cells"
    ref = {k: refuse(o) for k, o in tab.items()}
    hed = {k: hedge(o) for k, o in tab.items()}

    def rank(models):
        n = len(models)
        rows = [{"content_id": c, "n_refuse": (k := sum(ref[(m, c)] for m in models)), "disagreement": k * (n - k),
                 "hedges": sum(hed[(m, c)] for m in models)} for c in sorted(seeds)]
        random.Random(0).shuffle(rows)
        rows.sort(key=lambda r: (-r["disagreement"], r["hedges"]))
        for i, r in enumerate(rows, 1):
            r["rank"] = i
            r["prompt"] = prompts[r["content_id"]]
            for m in models:
                r[SHORT[m]] = "R" if ref[(m, r["content_id"])] else ("H" if hed[(m, r["content_id"])] else "C")
        return rows

    # sanity: reproduce the old 4-model ranking from the old data
    old_models = [LLAMA, QWEN, AYA, G2]
    old_csv = list(csv.DictReader((OLD / "stage1_merged_ranked.csv").open()))
    mine = rank(old_models)
    repro = [r["content_id"] for r in mine] == [r["content_id"] for r in old_csv]
    print("old ranking reproduced exactly:", repro)
    old_top50 = [r["content_id"] for r in old_csv[:50]]

    summary = {}
    for name, models in {"OLD_4model": old_models, **VARIANTS}.items():
        rows = mine if name == "OLD_4model" else rank(models)
        cols = ["rank", "content_id", "n_refuse", "disagreement", "hedges"] + [SHORT[m] for m in models] + ["prompt"]
        with (OUT / f"ranked_{name}.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        top = [r["content_id"] for r in rows[:50]]
        summary[name] = {
            "models": [SHORT[m] for m in models],
            "overlap_old_top50": len(set(top) & set(old_top50)),
            "top50": top,
            "n_refuse_dist": dict(sorted(Counter(r["n_refuse"] for r in rows).items())),
            "n_max_disagreement": sum(r["disagreement"] == rows[0]["disagreement"] for r in rows),
            "rows": rows,
        }

    rates = {}
    for m in SHORT:
        r = [ref[(m, c)] for c in seeds]
        h = [hed[(m, c)] for c in seeds]
        rates[SHORT[m]] = {"refuse": sum(r) / 861, "hedge": sum(h) / 861,
                           "backend": tab[(m, seeds[0])]["backend"]}
    # new-model agreement with Gemma-2 as a runtime/model drift cue
    agree = {SHORT[m]: sum(ref[(m, c)] == ref[(G2, c)] for c in seeds) / 861 for m in (G3, G4)}

    # ---- summary.md
    L = ["# E3 stage 1: harmful-seed re-screen\n",
         "Keyword-regex screen for picking seeds, not a reported rate. Greedy, 128 tokens, 861 English harmful seeds.\n",
         "Caveat: old Llama/Qwen3/Aya cells are llama.cpp Q4_K_M GGUF (minty); old Gemma-2 is MLX 4-bit; "
         "new Gemma3/Gemma4 are Ollama Q4_K_M GGUF. Runtime differences are not separable from model differences.\n",
         f"Old 4-model ranking reproduced from raw data: **{repro}**.\n",
         "## Refusal rate per model (regex)\n", "| model | backend | refuse | hedge (non-refusal disclaimer) |", "|---|---|---|---|"]
    for k, v in rates.items():
        L.append(f"| {k} | {v['backend']} | {v['refuse']:.1%} | {v['hedge']:.1%} |")
    L.append("\nAgreement of refuse/comply with Gemma-2 per seed: " + ", ".join(f"{k} {v:.1%}" for k, v in agree.items()) + "\n")
    L += ["## Overlap with old top 50\n", "| ranking | models | n_refuse dist | seeds at max score | overlap with old top50 |", "|---|---|---|---|---|"]
    for n, s in summary.items():
        L.append(f"| {n} | {', '.join(s['models'])} | {s['n_refuse_dist']} | {s['n_max_disagreement']} | "
                 f"{'-' if n == 'OLD_4model' else str(s['overlap_old_top50']) + '/50'} |")
    for n in ("A_gemma3", "B_gemma4", "C_5model_g3", "D_5model_g4", "E_6model"):
        s = summary[n]
        L += [f"\n## Top 50: {n}\n", "R=refuse H=hedge C=comply. * = also in old top 50.\n",
              "| rank | content_id | n_ref | hedges | " + " | ".join(s["models"]) + " | prompt |",
              "|---|---|---|---|" + "---|" * len(s["models"]) + "---|"]
        for r in s["rows"][:50]:
            star = "*" if r["content_id"] in old_top50 else ""
            p = r["prompt"].replace("|", "/").replace("\n", " ")[:90]
            L.append(f"| {r['rank']} | {r['content_id']}{star} | {r['n_refuse']} | {r['hedges']} | "
                     + " | ".join(r[m] for m in s["models"]) + f" | {p} |")
    (OUT / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")

    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    import ollama, pandas
    (OUT / "run_rank.json").write_text(json.dumps({
        "command": "python3 scripts/rank_seeds.py", "git_hash": git, "ts": datetime.now(timezone.utc).isoformat(),
        "tie_break_seed": 0, "python": sys.version.split()[0],
        "packages": {"ollama": getattr(ollama, "__version__", None), "pandas": pandas.__version__},
        "inputs": [str(NEW_FILE.relative_to(ROOT)), "runs/harmful-screen-2026-09/stage1.jsonl",
                   "runs/harmful-screen-2026-09/minty_llamacpp/stage1.jsonl", "runs/harmful-screen-2026-09/stage1_merged_ranked.csv"],
        "stage1_run_json": "results/e3/run.json", "old_ranking_reproduced": repro, "rates": rates,
        "overlap_old_top50": {n: s["overlap_old_top50"] for n, s in summary.items()},
        "top50": {n: s["top50"] for n, s in summary.items()},
    }, indent=1), encoding="utf-8")
    print(json.dumps({n: s["overlap_old_top50"] for n, s in summary.items()}))
    print(json.dumps(rates, indent=1))


if __name__ == "__main__":
    main()
