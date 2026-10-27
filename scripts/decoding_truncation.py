#!/usr/bin/env python3
"""E5: decoding parameters on the E2 grid (3 prompts x 9 tasks, Gemma3-12B via Ollama).

Arms: greedy@256 (greedy@512 = E2), T=1.0 x k=10 @512 (seed = sample index, top_p 1, top_k 0).
  gen     -> python3;  report -> .venv/bin/python (Qwen3Guard-4B). Outputs in results/e5/.
Questions: refusal variance across samples, truncation at 256 vs 512, how many samples k are enough.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from task_pilot import MODEL, ROOT, TASKS, jl, rows  # noqa: E402

OUT = ROOT / "results" / "e5"
K = 10


def gen() -> None:
    import ollama
    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "gen.jsonl"
    seen = {(r["arm"], r["row_id"]) for r in jl(out)}
    arms = [("greedy256", 256, {"temperature": 0, "seed": 0, "top_k": 1}, 0)] + \
           [(f"T1_s{k}", 512, {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "seed": k}, k) for k in range(K)]
    client = ollama.Client()
    with out.open("a", encoding="utf-8") as f:
        for arm, n, opts, _ in arms:
            for r in rows():
                if (arm, r["row_id"]) in seen:
                    continue
                kw = dict(model=MODEL, messages=[{"role": "user", "content": r["prompt"]}],
                          options={**opts, "num_predict": n})
                try:
                    resp = client.chat(think=False, **kw)
                except ollama.ResponseError:
                    resp = client.chat(**kw)
                if not (resp.message.content or "").strip():
                    print("EMPTY response, not written (rerun to retry):", arm, r["row_id"], flush=True)
                    continue
                f.write(json.dumps({"arm": arm, "max_new": n, "row_id": r["row_id"], "task": r["task"],
                                    "content_id": r["content_id"], "prompt": r["prompt"],
                                    "response": resp.message.content or "", "done_reason": resp.done_reason},
                                   ensure_ascii=False) + "\n")
                f.flush()
            print(arm, "done", flush=True)
    (OUT / "run.json").write_text(json.dumps({
        "experiment": "E5 decoding", "model": MODEL, "arms": "greedy@256; T=1.0,top_p=1,top_k=0,seed=k (k<10) @512",
        "greedy512_from": "results/e2/gen.jsonl", "command": "scripts/decoding_truncation.py gen && report"}, indent=1))


def report() -> None:
    from judge_qwen3guard import MODEL as GUARD, load_mlx_judge, parse
    g = jl(OUT / "gen.jsonl") + [{**r, "arm": "greedy512", "max_new": 512} for r in jl(ROOT / "results" / "e2" / "gen.jsonl")]
    jf = OUT / "judge.jsonl"
    done = {(r["arm"], r["row_id"]) for r in jl(jf)}
    todo = [r for r in g if (r["arm"], r["row_id"]) not in done]
    if todo:
        judge = load_mlx_judge(GUARD)
        with jf.open("a", encoding="utf-8") as f:
            for r in todo:
                tag, _ = parse(judge(r["prompt"], r["response"]))
                f.write(json.dumps({"arm": r["arm"], "row_id": r["row_id"], "tag": tag}) + "\n")
                f.flush()
    tag = {(r["arm"], r["row_id"]): r["tag"] for r in jl(jf)}
    L = ["# E5 decoding (English, Gemma3-12B, 27 cells = 3 prompts x 9 tasks)\n"]
    # truncation
    L += ["## Truncation (done_reason=length)\n", "| arm | truncated / n |", "|---|---|"]
    for arm in ("greedy256", "greedy512"):
        s = [r for r in g if r["arm"] == arm]
        L.append(f"| {arm} | {sum(r['done_reason'] == 'length' for r in s)} / {len(s)} |")
    s = [r for r in g if r["arm"].startswith("T1_")]
    L.append(f"| T=1.0 @512 | {sum(r['done_reason'] == 'length' for r in s)} / {len(s)} |")
    by_task = defaultdict(lambda: [0, 0])
    for r in s:
        by_task[r["task"]][0] += r["done_reason"] == "length"; by_task[r["task"]][1] += 1
    L += ["", "T=1.0 @512 truncation by task: " + ", ".join(f"{t} {a}/{n}" for t, (a, n) in by_task.items()), ""]
    # variance
    cells = defaultdict(list)
    for r in g:
        if r["arm"].startswith("T1_"):
            cells[r["row_id"]].append(tag[(r["arm"], r["row_id"])] == "refuse")
    mixed = {c: sum(v) for c, v in cells.items() if 0 < sum(v) < len(v)}
    L += ["## Refusal variance (guard), 10 samples/cell\n",
          f"cells with mixed outcomes: {len(mixed)}/{len(cells)}", f"mixed cells (refusals of 10): {mixed}", ""]
    gr = {r["row_id"]: tag[("greedy512", r["row_id"])] == "refuse" for r in g if r["arm"] == "greedy512"}
    g256 = {r["row_id"]: tag[("greedy256", r["row_id"])] == "refuse" for r in g if r["arm"] == "greedy256"}
    agree = sum(gr[c] == (sum(v) >= 5) for c, v in cells.items())
    L += [f"greedy512 vs majority of 10 samples: agree {agree}/{len(cells)}",
          f"greedy256 vs greedy512 label: agree {sum(g256[c] == gr[c] for c in gr)}/{len(gr)}",
          f"overall refusal rate: greedy512 {sum(gr.values())/len(gr):.3f}, greedy256 {sum(g256.values())/len(g256):.3f}, "
          f"T1 mean {sum(sum(v) for v in cells.values())/(10*len(cells)):.3f}", ""]
    # k sufficiency: overall refusal-rate estimate from k samples/cell vs k=10, over all k-subsets of sample indices
    L += ["## How many samples k? overall refusal-rate estimate vs k=10 (mean |error| over all k-subsets)\n",
          "| k | mean abs error | max abs error |", "|---|---|---|"]
    full = sum(sum(v) for v in cells.values()) / (10 * len(cells))
    for k in (1, 2, 3, 5, 7):
        errs = [abs(sum(sum(v[i] for i in sub) for v in cells.values()) / (k * len(cells)) - full)
                for sub in combinations(range(10), k)]
        L.append(f"| {k} | {st.mean(errs):.4f} | {max(errs):.4f} |")
    (OUT / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    {"gen": gen, "report": report}[sys.argv[1]]()
