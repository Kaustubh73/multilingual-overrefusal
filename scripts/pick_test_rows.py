#!/usr/bin/env python3
"""Pick one test row per (lang, intended_task, label) triple from HF dataset."""
from __future__ import annotations

import re
from collections import defaultdict

from datasets import load_dataset

TRIPLES = [
    ("en", "no_task", "harmful"),
    ("es", "sentiment_analysis", "harmful"),
    ("fr", "rephrase", "harmless"),
    ("de", "summarize", "harmful"),
    ("ja", "translate", "harmless"),
    ("zh", "pos_tagging", "harmful"),
    ("ar", "word_shuffling", "harmless"),
    ("ru", "character_counting", "harmful"),
    ("th", "no_task", "harmless"),
    ("hi", "sentiment_analysis", "harmless"),
    ("id", "rephrase", "harmful"),
    ("tr", "summarize", "harmless"),
    ("yo", "translate", "harmful"),
    ("sw", "pos_tagging", "harmless"),
    ("ne", "character_counting", "harmful"),
]

AVOID = re.compile(
    r"\b(child|children|suicide|self-harm|self harm|rape|porn|sexual)\b", re.I
)
PREFER = re.compile(
    r"\b(cybercrime|fraud|malware|hacking|hack|weapon|discrimination|discriminat)\b",
    re.I,
)


def en_seed_text(row, en_lookup):
    cid = row.get("content_id")
    if cid is None:
        return ""
    key = (cid, "en", "no_task")
    if key in en_lookup:
        return en_lookup[key]
    return en_lookup.get((cid, "en"), "")


def score_candidate(row, en_lookup, used_cids, categories_used):
    en = en_seed_text(row, en_lookup)
    length = len(en)
    score = 0.0
    notes = []

    if not en.strip():
        score -= 5000
        notes.append("no_en_seed")
    elif not (40 <= length <= 140):
        dist = min(abs(length - 40), abs(length - 140))
        score -= dist * 5
        notes.append(f"en_len={length}")
    else:
        score += 100
        # sweet spot near middle of band
        score += 20 - abs(length - 90) * 0.2

    label = row.get("label", "")
    if label == "harmful" and AVOID.search(en):
        score -= 1000
        notes.append("avoid_term")
    if label == "harmful" and PREFER.search(en):
        score += 30

    cid = row.get("content_id")
    if cid in used_cids:
        score -= 200
        notes.append("dup_cid")

    cat = row.get("category") or row.get("harm_category")
    if cat and cat not in categories_used:
        score += 15

    return score, en, notes


def main():
    ds = load_dataset("kaustubhg73/multilingual-multitask-refusal", split="test")
    cols = set(ds.column_names)
    print("COLUMNS:", sorted(cols), flush=True)
    print("N:", len(ds), flush=True)

    # normalize field names
    def get(row, *keys, default=""):
        for k in keys:
            if k in row and row[k] is not None:
                return row[k]
        return default

    rows = [dict(r) for r in ds]

    en_lookup = {}
    for r in rows:
        cid = get(r, "content_id")
        lang = get(r, "lang", "language")
        task = get(r, "intended_task", "task")
        text = get(r, "content", "prompt", "text")
        if cid is not None:
            en_lookup[(cid, lang, task)] = text
            en_lookup.setdefault((cid, lang), text)

    # index by triple
    by_triple = defaultdict(list)
    for r in rows:
        lang = get(r, "lang", "language")
        task = get(r, "intended_task", "task")
        label = get(r, "label")
        by_triple[(lang, task, label)].append(r)

    used_cids = set()
    categories_used = set()
    picks = []
    fallbacks = []

    for i, triple in enumerate(TRIPLES, 1):
        lang, task, label = triple
        cands = by_triple.get(triple, [])
        alt_note = None

        if not cands:
            # closest: same lang+label, then lang+task, then lang only
            for key_fn in [
                lambda r: get(r, "lang", "language") == lang
                and get(r, "label") == label,
                lambda r: get(r, "lang", "language") == lang
                and get(r, "intended_task", "task") == task,
                lambda r: get(r, "lang", "language") == lang,
            ]:
                cands = [r for r in rows if key_fn(r)]
                if cands:
                    alt_note = "fallback: relaxed triple match"
                    break
            if not cands:
                alt_note = "MISSING: no row for lang"
                picks.append((i, triple, None, alt_note))
                continue

        scored = []
        for r in cands:
            s, en, notes = score_candidate(r, en_lookup, used_cids, categories_used)
            scored.append((s, r, en, notes))
        scored.sort(key=lambda x: x[0], reverse=True)
        best = scored[0]
        r = best[1]
        en = best[2]
        cid = get(r, "content_id", default="?")
        if cid != "?":
            used_cids.add(cid)
        cat = get(r, "category", "harm_category", default=None)
        if cat:
            categories_used.add(cat)

        prompt = get(r, "prompt", "content", "text")
        if len(prompt) > 180:
            prompt = prompt[:177] + "..."

        bracket = get(r, "resource_bracket", "bracket", "language_bracket", default=lang)
        picks.append((i, triple, r, alt_note, en, prompt, bracket, cid))

    harmful = sum(1 for p in picks if p[1][2] == "harmful" and p[2] is not None)
    harmless = sum(1 for p in picks if p[1][2] == "harmless" and p[2] is not None)

    print("\n| # | lang | bracket | task | label | content_id | EN content | localized prompt |")
    print("|---|------|---------|------|-------|------------|------------|------------------|")
    for p in picks:
        i, (lang, task, label), r, *rest = p
        if r is None:
            note = rest[0] if rest else "missing"
            print(
                f"| {i} | {lang} | — | {task} | {label} | — | — | **{note}** |"
            )
            continue
        alt_note, en, prompt, bracket, cid = rest
        en_cell = en.replace("|", "\\|").replace("\n", " ")
        prompt_cell = prompt.replace("|", "\\|").replace("\n", " ")
        note_suffix = f" ({alt_note})" if alt_note else ""
        print(
            f"| {i} | {lang} | {bracket} | {task} | {label} | {cid} | {en_cell} | {prompt_cell}{note_suffix} |"
        )

    print(f"\nHarmful count: {harmful}")
    print(f"Harmless count: {harmless}")
    print(f"Distinct content_ids: {len(used_cids)}")


if __name__ == "__main__":
    main()
