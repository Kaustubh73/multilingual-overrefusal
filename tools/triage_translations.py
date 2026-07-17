#!/usr/bin/env python3
"""Triage the Track B translation review CSV into a prioritized worksheet.

Read-only on the source CSV. Runs NO model inference and touches no GPU.
It flags rows that need human attention and ranks them so the weekly
20-row review session attacks the highest-leverage rows first.

Flags (see TRANSLATION_REVIEW_GUIDE.md / track-b-translation-review.md):
  DRAFT      empty Hindi or Hinglish -> needs a human draft
  COPY       target identical to English (copy-through, e.g. test_0013)
  DISCLAIMER model disclaimer / refusal text in a translation field (test_0091)
  LEN        suspicious length ratio vs English
  SCRIPT     wrong script (Latin in Hindi, or Devanagari in Hinglish)
  ENC        encoding / mojibake artifact

Usage:
  python tools/triage_translations.py
  python tools/triage_translations.py --csv data/track_b/reviewed_translations.csv
  python tools/triage_translations.py --out-dir results --top 20

Outputs (written to --out-dir, default: alongside the input CSV):
  translation_triage.csv  one row per logical CSV row, with flags + priority
  translation_triage.md   human-readable worksheet, highest priority first
"""
from __future__ import annotations
import argparse
import csv
import os
import re
import sys

COLS = ["sample_id", "english", "hindi", "hinglish", "translation_source",
        "reviewed", "meaning_score", "naturalness_score",
        "surface_sensitivity_score", "notes"]

DISC_RE = re.compile(
    r"(disclaimer|i (cannot|can't|am unable|won'?t)|as an ai|i'?m sorry|"
    r"important note|i must (decline|refuse)|seek professional|not appropriate)",
    re.IGNORECASE)
ENC_RE = re.compile(r"[\uFFFD]|Ã.|Â.|à¤|â€")
DEV_RE = re.compile(r"[\u0900-\u097F]")

# severity: higher = review sooner
SEV = {"disc": 6, "enc": 6, "copy": 5, "script": 4, "len": 3, "draft": 2}


def dev_ratio(s: str) -> float:
    if not s:
        return 0.0
    letters = len(re.sub(r"[^A-Za-z\u0900-\u097F]", "", s))
    if not letters:
        return 0.0
    return len(DEV_RE.findall(s)) / letters


def is_true(v: str) -> bool:
    return str(v).strip().lower() == "true"


def is_smoke(o: dict) -> bool:
    return "smoke" in (o.get("notes") or "").lower() \
        or "smoke" in (o.get("translation_source") or "").lower()


def flags_for(o: dict) -> list[str]:
    f: list[str] = []
    en = (o.get("english") or "").strip()
    hi = (o.get("hindi") or "").strip()
    hl = (o.get("hinglish") or "").strip()
    if not hi or not hl:
        f.append("draft")
    if en:
        if hi and hi.lower() == en.lower():
            f.append("copy")
        elif hl and hl.lower() == en.lower():
            f.append("copy")
    if DISC_RE.search(hi) or DISC_RE.search(hl):
        f.append("disc")
    if en:
        for t in (hi, hl):
            if t:
                r = len(t) / len(en)
                if r < 0.35 or r > 3.2:
                    f.append("len")
                    break
    if hi and dev_ratio(hi) < 0.5:
        f.append("script")
    if hl and dev_ratio(hl) > 0.3:
        f.append("script")
    if ENC_RE.search(hi or "") or ENC_RE.search(hl or ""):
        f.append("enc")
    # de-dup, keep order
    seen, out = set(), []
    for x in f:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def priority(o: dict, fl: list[str]) -> int:
    if is_true(o.get("reviewed", "")):
        return -1
    if not fl:
        return 0
    s = max(SEV.get(x, 0) for x in fl)
    # a row that is ONLY empty (pure draft) is slower than a quick fix;
    # rank quick content fixes above blank-slate drafts
    if fl == ["draft"]:
        s = 1
    return s


def load(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(here)
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", default=os.path.join(
        repo, "data", "track_b", "reviewed_translations.csv"),
        help="path to reviewed_translations.csv (read only)")
    ap.add_argument("--out-dir", default=None,
                    help="where to write worksheet (default: next to input CSV)")
    ap.add_argument("--top", type=int, default=20,
                    help="rows to highlight in the 'this session' block")
    args = ap.parse_args()

    if not os.path.exists(args.csv):
        print(f"ERROR: CSV not found: {args.csv}", file=sys.stderr)
        return 2
    rows = load(args.csv)
    out_dir = args.out_dir or os.path.dirname(os.path.abspath(args.csv))
    os.makedirs(out_dir, exist_ok=True)

    enriched = []
    for o in rows:
        fl = flags_for(o)
        enriched.append({
            "sample_id": o.get("sample_id", ""),
            "reviewed": "true" if is_true(o.get("reviewed", "")) else "false",
            "smoke": "true" if is_smoke(o) else "false",
            "priority": priority(o, fl),
            "flags": "|".join(fl),
            "n_flags": len(fl),
            "english_gist": (o.get("english") or "").replace("\n", " ")[:80],
            "len_en": len((o.get("english") or "").strip()),
            "len_hi": len((o.get("hindi") or "").strip()),
            "len_hl": len((o.get("hinglish") or "").strip()),
        })

    pending = [e for e in enriched if e["reviewed"] == "false"]
    pending.sort(key=lambda e: (-e["priority"], e["sample_id"]))

    # ---- CSV worksheet ----
    csv_path = os.path.join(out_dir, "translation_triage.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(enriched[0].keys()))
        w.writeheader()
        # pending first (by priority), then reviewed
        done = [e for e in enriched if e["reviewed"] == "true"]
        for e in pending + done:
            w.writerow(e)

    # ---- tallies ----
    tally = {k: 0 for k in SEV}
    for e in pending:
        for fcode in e["flags"].split("|"):
            if fcode in tally:
                tally[fcode] += 1
    n_total = len(enriched)
    n_rev = sum(1 for e in enriched if e["reviewed"] == "true")
    n_pend = len(pending)
    n_draft = sum(1 for e in pending if "draft" in e["flags"].split("|"))

    NAME = {"draft": "DRAFT (empty)", "copy": "COPY (==English)",
            "disc": "DISCLAIMER", "len": "LEN ratio", "script": "SCRIPT",
            "enc": "ENC mojibake"}

    # ---- Markdown worksheet ----
    md_path = os.path.join(out_dir, "translation_triage.md")
    L = []
    L.append("# Track B translation triage worksheet")
    L.append("")
    L.append(f"_Generated from `{os.path.relpath(args.csv, repo)}` — read-only, no inference._")
    L.append("")
    L.append(f"- Total rows: **{n_total}**  ·  reviewed: **{n_rev}**  ·  "
             f"pending: **{n_pend}**  ·  need draft: **{n_draft}**")
    L.append(f"- Gate G-translate-60: **{n_rev}/60** toward production unblock "
             f"(then 115 for G-full)")
    L.append("")
    L.append("## Pending flag tally")
    L.append("")
    L.append("| Flag | Count | Meaning |")
    L.append("| --- | ---: | --- |")
    for k in sorted(tally, key=lambda x: -SEV[x]):
        meanings = {
            "disc": "model disclaimer/refusal in a field — strip & re-translate",
            "enc": "encoding artifact / mojibake — fix source text",
            "copy": "target identical to English — localize or exclude",
            "script": "wrong script (Latin in Hindi / Devanagari in Hinglish)",
            "len": "length ratio off vs English — verify completeness",
            "draft": "empty Hindi/Hinglish — needs a human draft",
        }
        L.append(f"| {NAME[k]} | {tally[k]} | {meanings[k]} |")
    L.append("")
    L.append(f"## This session — top {args.top} by priority")
    L.append("")
    L.append("Work top-to-bottom. Quick content fixes (disclaimer/copy/script) "
             "rank above blank-slate drafts.")
    L.append("")
    L.append("| # | sample_id | flags | smoke | English gist |")
    L.append("| ---: | --- | --- | :---: | --- |")
    for i, e in enumerate(pending[:args.top], 1):
        fl = " ".join(e["flags"].split("|")) if e["flags"] else "clean"
        smk = "•" if e["smoke"] == "true" else ""
        L.append(f"| {i} | `{e['sample_id']}` | {fl} | {smk} | {e['english_gist']} |")
    L.append("")
    L.append("## How to clear these")
    L.append("")
    L.append("Open the interactive tool for fast row-by-row editing + scoring:")
    L.append("")
    L.append("```bash")
    L.append("open tools/translation_review.html   # then Load CSV -> "
             "data/track_b/reviewed_translations.csv")
    L.append("```")
    L.append("")
    L.append("Scores: meaning & naturalness both **≥4** to pass. Set "
             "`reviewed=true` and `translation_source=human`/`human-edited` "
             "only after a human confirms (agents must not).")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    print(f"Read   {args.csv}  ({n_total} rows, {n_rev} reviewed, {n_pend} pending)")
    print(f"Wrote  {csv_path}")
    print(f"Wrote  {md_path}")
    print("\nTop pending flags:",
          ", ".join(f"{NAME[k]}={tally[k]}" for k in sorted(tally, key=lambda x: -SEV[x])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
