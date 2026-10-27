#!/usr/bin/env python3
"""E4: translation QA for the four new languages (pt, mt, ur, lo). No hand tagging.

  score    drift metrics (chrF++, BLEU, MiniLM cosine, length ratio) for en vs back-translation
  intent   Qwen3Guard-Gen prompt moderation on 400 random harmful seeds: original en, back-translation
           and the forward translation itself, for pt/mt/ur/lo and the v1 reference langs
           (resumable jsonl)
  report   per-language tables + flags -> results/e4/summary.md, flags.json, run.json

Inputs (read-only): data/dataset_v2/sheets/new4_filled.csv (Google Sheets export:
content_id, en, pt, mt, ur, lo, bt_pt, bt_mt, bt_ur, bt_lo) and, as reference,
data/dataset_v1/backtranslate_scored.parquet (v1 Google Translate round trips).
Outputs: results/e4/.  Run with .venv/bin/python. Seeds fixed; decoding is greedy.

Intent retention: a harmful seed (Sakonii label == harmful) whose English text
Qwen3Guard flags (Unsafe or Controversial) counts as "intent kept" in a language when the
back-translation (or forward text) is still flagged. Lost = the guard says Safe.
Caveat: round trip uses the same engine both ways and the guard is weakest on the
languages we are testing, so forward-text loss is confounded with guard coverage.
Flag rule (fixed before looking): a language is flagged when its back-translation
intent loss is >10 percentage points above the es reference, or its mean cosine is
>0.10 below the es reference.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from translation_qa.backtranslate_metrics import score_pairs  # noqa: E402

OUT = ROOT / "results" / "e4"
FILLED = ROOT / "data" / "dataset_v2" / "sheets" / "new4_filled.csv"
V1 = ROOT / "data" / "dataset_v1" / "backtranslate_scored.parquet"
SEEDS = ROOT / "data" / "dataset_v1" / "seeds_en.parquet"
NEW = ["pt", "mt", "ur", "lo"]
REF = ["es", "hi", "yo", "sw"]  # v1 Google Translate references: high, mid-low, two low
GUARD = "Qwen/Qwen3Guard-Gen-8B"
N_HARMFUL = 400  # random (seed 0) subset of the 861 harmful seeds; ~1 s per guard call
SAFETY_RE = re.compile(r"Safety:\s*(Safe|Unsafe|Controversial)", re.I)
BAD = re.compile(r"^(Loading\.\.\.|#ERROR!|#N/A|Error)", re.I)


def load_pairs() -> pd.DataFrame:
    """Long table: content_id, lang, en, fwd, back, label, new(bool)."""
    seeds = pd.read_parquet(SEEDS)[["content_id", "label", "category"]]
    raw = pd.read_csv(FILLED, keep_default_na=False)
    rows = []
    for L in NEW:
        sub = raw[["content_id", "en", L, f"bt_{L}"]].rename(columns={L: "fwd", f"bt_{L}": "back"})
        sub["lang"] = L
        rows.append(sub)
    new = pd.concat(rows, ignore_index=True)
    bad = new[new.fwd.map(lambda s: bool(BAD.match(str(s))) or not str(s).strip()) |
              new.back.map(lambda s: bool(BAD.match(str(s))) or not str(s).strip())]
    if len(bad):
        raise SystemExit(f"{len(bad)} unresolved/empty cells, e.g. {bad.head(3).to_dict('records')}")
    v1 = pd.read_parquet(V1)
    v1 = v1[v1.lang.isin(REF)][["content_id", "lang", "en_content", "lang_content", "back_en"]]
    v1.columns = ["content_id", "lang", "en", "fwd", "back"]
    df = pd.concat([new.assign(new=True), v1.assign(new=False)], ignore_index=True)
    return df.merge(seeds, on="content_id", how="left")


def cmd_score(_a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_pairs()
    new = df[df.new].reset_index(drop=True)
    m = score_pairs(new.en, new.back)
    for k, v in m.items():
        new[k] = v
    new["same_as_en"] = new.fwd.str.strip().str.lower() == new.en.str.strip().str.lower()
    new.to_parquet(OUT / "drift_scored.parquet", index=False)
    print(new.groupby("lang")[["chrf", "bleu", "embed_cosine", "len_ratio"]].mean().round(3))


def guard_loader():
    """Qwen3Guard-Gen prompt moderation as one forward pass: append 'Safety:' to the
    assistant turn and take argmax over the first token of Safe / Unsafe / Controversial
    (checked equal to greedy generation on spot checks; ~1 s per prompt on the Mac)."""
    import mlx.core as mx
    from mlx_lm import load
    mx.set_cache_limit(2 * 1024**3)  # variable-length prompts otherwise grow the MLX buffer cache to 25 GB
    model, tok = load(GUARD)
    ids = {k: tok.encode(" " + k)[0] for k in ("Safe", "Unsafe", "Controversial")}

    def judge(text: str) -> tuple[str | None, str]:
        p = tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                    add_generation_prompt=True) + "Safety:"
        lg = model(mx.array(tok.encode(p))[None])[0, -1]
        sc = {k: round(lg[v].item(), 3) for k, v in ids.items()}
        return max(sc, key=sc.get), json.dumps(sc)

    return judge


def cmd_intent(a) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_pairs()
    df = df[df.label == "harmful"]
    ids = sorted(df.content_id.unique())
    random.Random(0).shuffle(ids)
    df = df[df.content_id.isin(set(ids[: a.limit or N_HARMFUL]))]
    path = OUT / "intent_guard.jsonl"
    done = set()
    if path.exists():
        done = {(r["content_id"], r["lang"], r["kind"]) for r in map(json.loads, path.open()) if r}
    # one job per distinct text: en once, then back and fwd per language
    jobs = [(cid, "en", "en", t) for cid, t in df.drop_duplicates("content_id")[["content_id", "en"]].itertuples(index=False)]
    for r in df.itertuples(index=False):
        jobs.append((r.content_id, r.lang, "back", r.back))
        if r.lang in NEW or r.lang == "yo":  # forward text: new langs + yo as the guard-coverage baseline
            jobs.append((r.content_id, r.lang, "fwd", r.fwd))
    jobs = [j for j in jobs if (j[0], j[1], j[2]) not in done]
    print(f"{len(jobs)} guard calls to run ({len(done)} done)", flush=True)
    judge = guard_loader()
    t0 = time.time()
    with path.open("a", encoding="utf-8") as f:
        for i, (cid, lang, kind, text) in enumerate(jobs):
            s, raw = judge(text)
            f.write(json.dumps({"content_id": cid, "lang": lang, "kind": kind, "safety": s, "raw": raw}, ensure_ascii=False) + "\n")
            if i % 200 == 0:
                f.flush()
                print(f"{i}/{len(jobs)} {time.time()-t0:.0f}s", flush=True)


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return (float("nan"), float("nan"))
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def cmd_report(_a) -> None:
    drift = pd.read_parquet(OUT / "drift_scored.parquet")
    v1 = pd.read_parquet(V1)
    ref_drift = v1[v1.lang.isin(REF)].groupby("lang")[["chrf", "bleu", "embed_cosine", "len_ratio"]].mean()
    new_drift = drift.groupby("lang")[["chrf", "bleu", "embed_cosine", "len_ratio"]].mean()
    g = pd.DataFrame([json.loads(l) for l in (OUT / "intent_guard.jsonl").open()])
    flagged = lambda s: s in ("Unsafe", "Controversial")
    en = {r.content_id: r.safety for r in g[g.kind == "en"].itertuples()}
    en_flag = {c for c, s in en.items() if flagged(s)}
    rows = []
    for L in NEW + REF:
        r = {"lang": L, "set": "new" if L in NEW else "v1 reference"}
        for kind in ("back", "fwd"):
            sub = g[(g.lang == L) & (g.kind == kind) & g.content_id.isin(en_flag)]
            lost = int(sum(not flagged(s) for s in sub.safety))
            lo, hi = wilson(lost, len(sub))
            r[f"{kind}_n"] = len(sub); r[f"{kind}_lost"] = lost
            r[f"{kind}_loss"] = lost / len(sub) if len(sub) else float("nan")
            r[f"{kind}_ci"] = (round(lo, 3), round(hi, 3))
        rows.append(r)
    loss = pd.DataFrame(rows).set_index("lang")
    cos_ref = float(ref_drift.loc["es", "embed_cosine"]) if "es" in ref_drift.index else float("nan")
    loss_ref = float(loss.loc["es", "back_loss"])
    flags = {}
    for L in NEW:
        why = []
        if loss.loc[L, "back_loss"] - loss_ref > 0.10:
            why.append(f"back-translation intent loss {loss.loc[L,'back_loss']:.1%} vs es {loss_ref:.1%}")
        if cos_ref - float(new_drift.loc[L, "embed_cosine"]) > 0.10:
            why.append(f"mean cosine {new_drift.loc[L,'embed_cosine']:.3f} vs es {cos_ref:.3f}")
        flags[L] = why
    # worst rows for manual eyeballing: harmful, flagged in en, back says Safe
    bk = g[(g.kind == "back") & g.lang.isin(NEW)].merge(drift[["content_id", "lang", "en", "fwd", "back", "chrf", "embed_cosine"]], on=["content_id", "lang"])
    bk = bk[bk.content_id.isin(en_flag) & ~bk.safety.map(flagged)]
    bk.sort_values("embed_cosine").to_csv(OUT / "intent_lost_rows.csv", index=False)
    drift.sort_values("embed_cosine").groupby("lang").head(25)[["lang", "content_id", "en", "fwd", "back", "chrf", "embed_cosine"]].to_csv(OUT / "lowest_cosine_rows.csv", index=False)

    L = ["# E4: translation QA, pt / mt / ur / lo", "",
         f"Google Translate (Sheets GOOGLETRANSLATE), en -> L -> en. Pilot QA, no hand tags: indicative, not a finding. "
         f"{len(drift)//len(NEW)} seeds per language; intent check on {len(en_flag)} harmful seeds that Qwen3Guard-Gen-8B flags in English "
         f"(of {int((g.kind=='en').sum())} harmful seeds checked).", "",
         "## Drift (mean over seeds)", "", "| lang | set | chrF++ | BLEU | cosine | len ratio |", "|---|---|---|---|---|---|"]
    for lg, row in list(new_drift.iterrows()) + list(ref_drift.iterrows()):
        L.append(f"| {lg} | {'new' if lg in NEW else 'v1 ref'} | {row.chrf:.1f} | {row.bleu:.1f} | {row.embed_cosine:.3f} | {row.len_ratio:.2f} |")
    L += ["", "## Harmful intent retained (Qwen3Guard-Gen-8B prompt moderation; flagged = Unsafe or Controversial)", "",
          "Loss = English flagged, other text judged Safe. back = back-translation to English (clean read of meaning drift); "
          "fwd = the in-language prompt itself (also depends on guard coverage of that language).", "",
          "| lang | set | back lost / n | back loss (95% CI) | fwd lost / n | fwd loss (95% CI) |", "|---|---|---|---|---|---|"]
    for lg, r in loss.iterrows():
        L.append(f"| {lg} | {r['set']} | {r.back_lost}/{r.back_n} | {r.back_loss:.1%} {r.back_ci} | {r.fwd_lost}/{r.fwd_n} | {r.fwd_loss:.1%} {r.fwd_ci} |")
    L += ["", "## Flags", "", "Rule (fixed in the script header): flag when back-translation intent loss is >10 pp above es, or mean cosine is >0.10 below es.", ""]
    for lg, why in flags.items():
        L.append(f"- **{lg}**: " + ("FLAGGED, " + "; ".join(why) if why else "not flagged"))
    L += ["", "Files: `drift_scored.parquet` (row level), `intent_guard.jsonl` (raw guard output), `intent_lost_rows.csv` (harmful seeds whose back-translation the guard calls Safe, lowest cosine first), `lowest_cosine_rows.csv` (25 worst per language).", ""]
    (OUT / "summary.md").write_text("\n".join(L))
    (OUT / "flags.json").write_text(json.dumps({"flags": flags, "loss": json.loads(loss.to_json(orient="index")),
                                                "drift_new": json.loads(new_drift.to_json(orient="index")),
                                                "drift_ref": json.loads(ref_drift.to_json(orient="index"))}, indent=1))
    def pkg(n):
        from importlib import metadata
        return metadata.version(n)
    (OUT / "run.json").write_text(json.dumps({
        "script": "translation_qa/intent_qa.py", "decode": "single forward pass, argmax over Safe/Unsafe/Controversial after 'Safety:'",
        "guard": GUARD, "mt_engine": "googletranslate_sheets (GOOGLETRANSLATE in Google Sheets)",
        "embed_model": "sentence-transformers/all-MiniLM-L6-v2",
        "versions": {n: pkg(n) for n in ("sacrebleu", "sentence-transformers", "mlx-lm", "torch", "pandas")},
        "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip(),
        "inputs": [str(FILLED.relative_to(ROOT)), str(V1.relative_to(ROOT)), str(SEEDS.relative_to(ROOT))],
        "finished_utc": datetime.now(timezone.utc).isoformat()}, indent=1))
    print("\n".join(L))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("score").set_defaults(fn=cmd_score)
    p = sp.add_parser("intent"); p.add_argument("--limit", type=int, default=0); p.set_defaults(fn=cmd_intent)
    sp.add_parser("report").set_defaults(fn=cmd_report)
    a = ap.parse_args()
    a.fn(a)
