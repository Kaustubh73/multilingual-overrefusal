#!/usr/bin/env python3
"""E1b: multilingual judge check, Qwen3Guard-Gen 4B vs 8B on ur / lo / mt / yo.

  sample    build the stratified prompt sample (180 rows: 4 langs x 9 tasks x 5)
  generate  3 recent models answer every sampled prompt (greedy, 512 tokens, resumable)
  judge     Qwen3Guard-Gen-4B and -8B label every response refuse/comply (resumable)
  handtag   build the blind hand-tag sheet + key (one response per sampled row)
  report    4B-vs-8B agreement overall / by language / by task / by model, plus
            hand-tag scoring if results/e1b/handtag/handtag_labels.jsonl exists

Run with .venv/bin/python from the repo root. Inputs are read-only; outputs go to
results/e1b/. Seed 0 everywhere; decoding is greedy (no sampling), so reruns match.

Sample design: the same 45 seeds (27 harmful + 18 harmless where Sakonii labels allow)
are used in all four languages; the task a seed gets rotates by language so each
lang x task cell has 5 rows and every cell has ~3 harmful rows. Seeds with an empty or
unresolved translation in any of the four languages, or longer than 300 characters,
are excluded. Content is the Full-GT translation (Google Translate): v2 sheet for ur/mt/lo,
v1 Full-GT for yo. The wrapper is the localized template, so the prompt is fully in the
target language.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
OUT = ROOT / "results" / "e1b"
SEED = 0
LANGS = ["ur", "lo", "mt", "yo"]
TEMPLATES = ROOT / "data" / "dataset_v2" / "templates_localized.json"
V1_FULL = ROOT / "data" / "dataset_v1" / "sheets" / "translations_gt_full.parquet"
NEW4 = ROOT / "data" / "dataset_v2" / "sheets" / "new4_forward.csv"  # forward translations only (E4 back-translations may still be resolving)
SEEDS = ROOT / "data" / "dataset_v1" / "seeds_en.parquet"
N_SEEDS = 45
TASKS = ["no_task", "sentiment_analysis", "rephrase", "summarize", "translate",
         "pos_tagging", "word_shuffling", "character_counting", "keyword_extraction"]
# translate target per source language, from assign_translate_targets (seed 42, v2 langs)
TRANSLATE_TARGET = {"ur": "zh", "lo": "tr", "mt": "id", "yo": "ar"}
GEN_MODELS = {
    "qwen3.5-9b": "lmstudio-community/Qwen3.5-9B-MLX-8bit",
    "gemma-4-e4b": "ollama:gemma4:e4b",  # mlx_lm 0.31.3 cannot load gemma-4-12B ("gemma4_unified"), so Gemma 4 runs via Ollama
    "llama-3.1-8b": "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit",
    "aya-expanse-8b": "mlx-community/aya-expanse-8b-4bit",  # Aya was in the stage-2 lineup (aya-23-8B); expanse is the newer 8B
}
JUDGES = {"qg4": "Qwen/Qwen3Guard-Gen-4B", "qg8": "Qwen/Qwen3Guard-Gen-8B"}
MAX_NEW = 512
RUBRIC_VERSION = "handtag-v1.1-2026-10-22"


def jl(p: Path) -> list[dict]:
    return [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()] if p.exists() else []


def load_content() -> pd.DataFrame:
    """content_id, lang, content for the four sampled languages (Full GT)."""
    new = pd.read_csv(NEW4, keep_default_na=False)
    rows = [{"content_id": r.content_id, "lang": L, "content": getattr(r, L)}
            for r in new.itertuples() for L in ("ur", "mt", "lo")]
    v1 = pd.read_parquet(V1_FULL)
    v1 = v1[v1.lang == "yo"][["content_id", "lang", "content"]]
    return pd.concat([pd.DataFrame(rows), v1], ignore_index=True)


def cmd_sample(_a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = pd.read_parquet(SEEDS)
    content = load_content()
    bad = re.compile(r"^(Loading\.\.\.|#ERROR!|#N/A)")
    ok = content[content.content.map(lambda s: bool(str(s).strip()) and not bad.match(str(s)))]
    have = ok.groupby("content_id").lang.nunique()
    good = set(have[have == len(LANGS)].index)
    pool = seeds[seeds.content_id.isin(good) & (seeds.content.str.len() <= 300)].copy()
    rng = random.Random(SEED)
    harm = sorted(pool[pool.label == "harmful"].content_id)
    ben = sorted(pool[pool.label == "harmless"].content_id)
    rng.shuffle(harm); rng.shuffle(ben)
    chosen = harm[:27] + ben[:18]          # harmful first, then harmless: rank order for task rotation
    templ = json.loads(TEMPLATES.read_text())
    names = templ["lang_names"]
    meta = seeds.set_index("content_id")
    cmap = ok.set_index(["content_id", "lang"]).content
    rows = []
    for li, L in enumerate(LANGS):
        for rank, cid in enumerate(chosen):
            task = TASKS[(rank + li * 2) % len(TASKS)]
            content_L = cmap[(cid, L)]
            tgt = TRANSLATE_TARGET[L] if task == "translate" else ""
            tname = names.get(L, {}).get(tgt, tgt) if tgt else ""
            prompt = templ["templates"][L][task].format(content=content_L, translate_target_name=tname)
            rows.append({"row_id": f"{L}-{task}-{cid}", "content_id": cid, "lang": L, "task": task,
                         "label": meta.loc[cid, "label"], "category": meta.loc[cid, "category"] or "",
                         "en_content": meta.loc[cid, "content"], "content": content_L, "prompt": prompt,
                         "translate_target": tgt})
    df = pd.DataFrame(rows)
    df.to_json(OUT / "sample.jsonl", orient="records", lines=True, force_ascii=False)
    cell = df.groupby(["lang", "task"]).size()
    assert cell.min() == cell.max() == 5 and len(df) == 180, cell.describe()
    print(len(df), "rows;", df.groupby("lang").label.value_counts().unstack().to_dict())


# ------------------------------------------------------------------ generation


BATCH = 16  # MLX batch size; ~3x faster than one prompt at a time. Greedy output can differ slightly
# from unbatched runs (padding changes float rounding), so every MLX row in a file uses batches of BATCH
# in sample order, and run.json records it.


def batched(model, tok, texts: list[str], max_tokens: int):
    """Yield (index, text) for greedy generation over prompts rendered to strings, BATCH at a time."""
    import importlib
    gm = importlib.import_module("mlx_lm.generate")
    for i in range(0, len(texts), BATCH):
        chunk = texts[i : i + BATCH]
        out = gm.batch_generate(model, tok, [tok.encode(t) for t in chunk], max_tokens=max_tokens).texts
        for j, o in enumerate(out):
            yield i + j, o


def ollama_chat(model: str, prompt: str) -> str:
    import urllib.request
    body = {"model": model, "stream": False, "think": False,
            "messages": [{"role": "user", "content": prompt}],
            "options": {"temperature": 0, "seed": SEED, "num_predict": MAX_NEW, "num_ctx": 4096}}
    req = urllib.request.Request("http://localhost:11434/api/chat", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=900))["message"].get("content", "")


def cmd_generate(a) -> None:
    from mlx_lm import generate, load
    sample = jl(OUT / "sample.jsonl")
    for key in (a.models or list(GEN_MODELS)):
        out = OUT / f"gen_{key}.jsonl"
        done = {r["row_id"] for r in jl(out)}
        todo = [r for r in sample if r["row_id"] not in done]
        if not todo:
            continue
        if GEN_MODELS[key].startswith("ollama:"):
            with out.open("a", encoding="utf-8") as f:
                for r in todo:
                    t0 = time.time()
                    resp = ollama_chat(GEN_MODELS[key][len("ollama:"):], r["prompt"])
                    f.write(json.dumps({"row_id": r["row_id"], "model": key, "response": resp,
                                        "latency_s": round(time.time() - t0, 1)}, ensure_ascii=False) + "\n")
                    f.flush()
            continue
        print(f"[{key}] loading {GEN_MODELS[key]}; {len(todo)} prompts (batch {BATCH})", flush=True)
        model, tok = load(GEN_MODELS[key])
        texts = []
        for r in todo:
            msg = [{"role": "user", "content": r["prompt"]}]
            try:
                texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True, enable_thinking=False))
            except TypeError:
                texts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        t0 = time.time()
        with out.open("a", encoding="utf-8") as f:
            for i, resp in batched(model, tok, texts, MAX_NEW):
                f.write(json.dumps({"row_id": todo[i]["row_id"], "model": key, "response": resp,
                                    "batch": BATCH}, ensure_ascii=False) + "\n")
                f.flush()
                if i % BATCH == BATCH - 1:
                    print(f"[{key}] {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        del model
        import gc, mlx.core as mx
        gc.collect(); mx.clear_cache()


# ------------------------------------------------------------------ judging


def cmd_judge(a) -> None:
    from mlx_lm import generate, load
    from judge_qwen3guard import MAX_TOKENS, parse
    sample = {r["row_id"]: r for r in jl(OUT / "sample.jsonl")}
    items = []
    for key in GEN_MODELS:
        items += [{**g, "prompt": sample[g["row_id"]]["prompt"]} for g in jl(OUT / f"gen_{key}.jsonl")]
    for tag in (a.judges or list(JUDGES)):
        out = OUT / f"judge_{tag}.jsonl"
        done = {(r["row_id"], r["model"]) for r in jl(out)}
        todo = [it for it in items if (it["row_id"], it["model"]) not in done]
        if not todo:
            continue
        print(f"[{tag}] loading {JUDGES[tag]}; {len(todo)} responses (batch {BATCH})", flush=True)
        model, tok = load(JUDGES[tag])  # official bf16 weights
        texts = [tok.apply_chat_template([{"role": "user", "content": it["prompt"]},
                                          {"role": "assistant", "content": it["response"]}],
                                         tokenize=False, add_generation_prompt=True) for it in todo]
        t0 = time.time()
        with out.open("a", encoding="utf-8") as f:
            for i, raw in batched(model, tok, texts, MAX_TOKENS):
                verdict, safety = parse(raw)
                f.write(json.dumps({"row_id": todo[i]["row_id"], "model": todo[i]["model"], "verdict": verdict,
                                    "safety": safety, "raw": raw, "batch": BATCH}, ensure_ascii=False) + "\n")
                f.flush()
                if i % BATCH == BATCH - 1:
                    print(f"[{tag}] {i + 1}/{len(todo)} {time.time() - t0:.0f}s", flush=True)
        del model
        import gc, mlx.core as mx
        gc.collect(); mx.clear_cache()


# ------------------------------------------------------------------ hand-tag sheet


def cmd_handtag(_a) -> None:
    """One response per sampled row (models rotated), blind sheet + separate key."""
    sample = {r["row_id"]: r for r in jl(OUT / "sample.jsonl")}
    gens = {k: {g["row_id"]: g for g in jl(OUT / f"gen_{k}.jsonl")} for k in GEN_MODELS}
    keys = [k for k in GEN_MODELS if len(gens[k]) == len(sample)]  # only fully generated models
    rng = random.Random(SEED)
    order = list(sample)
    cell_count = defaultdict(int)
    picked = []
    for i, rid in enumerate(order):
        r = sample[rid]
        c = cell_count[(r["lang"], r["task"])]; cell_count[(r["lang"], r["task"])] += 1
        # rotate the model within the cell and shift by task so totals stay balanced
        k = keys[(c + TASKS.index(r["task"]) + LANGS.index(r["lang"])) % len(keys)]
        picked.append((rid, k))
    rng.shuffle(picked)
    hd = OUT / "handtag"
    hd.mkdir(exist_ok=True)
    sheet, key = [], []
    for n, (rid, k) in enumerate(picked, start=1):
        r, g = sample[rid], gens[k][rid]
        item = f"m{n:03d}"
        sheet.append({"n": n, "item": item, "prompt": r["prompt"], "response": g["response"]})
        key.append({"item": item, "row_id": rid, "model": k, "lang": r["lang"], "task": r["task"], "label": r["label"]})
    (hd / "handtag_sheet.jsonl").write_text("\n".join(json.dumps(s, ensure_ascii=False) for s in sheet) + "\n")
    (hd / "handtag_key.json").write_text(json.dumps(key, indent=1, ensure_ascii=False))
    (hd / "rubric_version.txt").write_text(RUBRIC_VERSION + "\n")
    print("handtag sheet", len(sheet), Counter((k["lang"], k["model"]) for k in key))


# ------------------------------------------------------------------ report


def kappa(a: list, b: list) -> float:
    n = len(a)
    po = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[k] * cb[k] for k in set(a) | set(b)) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def cmd_report(_a) -> None:
    sample = {r["row_id"]: r for r in jl(OUT / "sample.jsonl")}
    J = {t: {(r["row_id"], r["model"]): r for r in jl(OUT / f"judge_{t}.jsonl")} for t in JUDGES}
    keys = sorted(set(J["qg4"]) & set(J["qg8"]))
    df = pd.DataFrame([{"row_id": k[0], "model": k[1], "lang": sample[k[0]]["lang"], "task": sample[k[0]]["task"],
                        "label": sample[k[0]]["label"], "qg4": J["qg4"][k]["verdict"], "qg8": J["qg8"][k]["verdict"]}
                       for k in keys])
    gens = {(g["row_id"], k): g["response"] for k in GEN_MODELS for g in jl(OUT / f"gen_{k}.jsonl")}
    df["resp_len"] = [len(gens[(r, m)]) for r, m in zip(df.row_id, df.model)]
    df["agree"] = df.qg4 == df.qg8
    df.to_csv(OUT / "judge_pairs.csv", index=False)

    def table(by):
        t = []
        for k, g in df.groupby(by):
            valid = g.dropna(subset=["qg4", "qg8"])
            kp = kappa(list(valid.qg4), list(valid.qg8)) if len(valid) > 1 else float("nan")
            t.append((k, len(g), g.agree.mean(), kp, (g.qg4 == "refuse").mean(), (g.qg8 == "refuse").mean(),
                      int(g.qg4.isna().sum() + g.qg8.isna().sum())))
        return t

    def md(by, t):
        s = [f"| {by} | n | 4B=8B | kappa(4B,8B) | refuse% 4B | refuse% 8B | unparsed |", "|---|---|---|---|---|---|---|"]
        s += [f"| {k} | {n} | {ag:.3f} | {kp:.3f} | {r4:.1%} | {r8:.1%} | {u} |" for k, n, ag, kp, r4, r8, u in t]
        return "\n".join(s)

    L = ["# E1b: Qwen3Guard-Gen 4B vs 8B on ur / lo / mt / yo", "",
         f"Pilot, not a finding. {len(df)} judged responses = {df.row_id.nunique()} prompts x {df.model.nunique()} generators "
         f"(greedy, {MAX_NEW} tokens): {', '.join(GEN_MODELS.values())}. Judges are bf16 official weights, greedy, "
         "`Refusal: Yes/No` mapped to refuse/comply. Rubric note added: `reject premise, then pivot` (docs/handtag-protocol.md, item 13).", "",
         f"Overall 4B=8B agreement: **{df.agree.mean():.3f}**, kappa {kappa(list(df.qg4.fillna('x')), list(df.qg8.fillna('x'))):.3f}.", "",
         "## By language", "", md("lang", table("lang")), "", "## By task", "", md("task", table("task")), "",
         "## By generator", "", md("model", table("model")), ""]
    dis = df[~df.agree]
    L += [f"Disagreements: {len(dis)} / {len(df)}. By language: {dict(dis.lang.value_counts())}. "
          f"Direction: 4B refuse / 8B comply = {int(((dis.qg4=='refuse')&(dis.qg8=='comply')).sum())}, "
          f"4B comply / 8B refuse = {int(((dis.qg4=='comply')&(dis.qg8=='refuse')).sum())}.", ""]
    lab_path = next((q for q in (OUT / "handtag" / "_shown" / "handtag_labels.jsonl", OUT / "handtag" / "handtag_labels.jsonl")
                     if q.exists() and q.stat().st_size), None)
    if lab_path:
        key = {k["item"]: k for k in json.load(open(OUT / "handtag" / "handtag_key.json"))}
        lab = {}
        for ev in jl(lab_path):
            lab[ev["item"]] = ev
        hr = []
        for it, ev in lab.items():
            k = key[it]
            hr.append({"lang": k["lang"], "task": k["task"], "model": k["model"], "hand": ev["label"],
                       "qg4": J["qg4"][(k["row_id"], k["model"])]["verdict"], "qg8": J["qg8"][(k["row_id"], k["model"])]["verdict"]})
        h = pd.DataFrame(hr)
        h = h[h.hand.isin(["comply", "refuse"])]
        L += ["## Hand-tag scoring (garble items excluded)", "", "| group | n | acc 4B | acc 8B | kappa 4B | kappa 8B |", "|---|---|---|---|---|---|"]
        for name, g in [("all", h)] + [(f"lang={k}", g) for k, g in h.groupby("lang")]:
            L.append(f"| {name} | {len(g)} | {(g.hand==g.qg4).mean():.3f} | {(g.hand==g.qg8).mean():.3f} | "
                     f"{kappa(list(g.hand), list(g.qg4.fillna('x'))):.3f} | {kappa(list(g.hand), list(g.qg8.fillna('x'))):.3f} |")
        L.append("")
    else:
        L += ["Hand tags: not yet done (`results/e1b/handtag/` holds the blind sheet). Re-run `report` after tagging.", ""]
    (OUT / "summary.md").write_text("\n".join(L))
    from importlib import metadata
    (OUT / "run.json").write_text(json.dumps({
        "script": "scripts/multilingual_judge.py", "seed": SEED, "decode": f"greedy; gen max_tokens {MAX_NEW}; judge max_tokens 64; MLX models batched x{BATCH} (ollama row-by-row); a first Qwen3.5 pass of 150 rows was unbatched and was discarded",
        "gen_models": GEN_MODELS, "judges": JUDGES, "mlx_lm": metadata.version("mlx-lm"),
        "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "rubric": RUBRIC_VERSION, "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("sample").set_defaults(fn=cmd_sample)
    p = sp.add_parser("generate"); p.add_argument("--models", nargs="*"); p.set_defaults(fn=cmd_generate)
    p = sp.add_parser("judge"); p.add_argument("--judges", nargs="*"); p.set_defaults(fn=cmd_judge)
    sp.add_parser("handtag").set_defaults(fn=cmd_handtag)
    sp.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)
