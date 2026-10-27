#!/usr/bin/env python3
"""
Build / refresh data/dataset_v1/templates_localized.json.

- Overlap langs: extract shells from Sakonii/Multitask-Multilingual-Refusal.
- New langs (hi id tr sw ne): keep existing newly added entries if present;
  otherwise refuse (fill those once in-session — do not burn quota in a loop).

Usage:
  .venv/bin/python dataset/build_localized_templates.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "dataset_v1" / "templates_localized.json"

KEEP_LANGS = ["en", "es", "fr", "de", "ja", "zh", "ar", "ru", "th", "yo"]
NEW_LANGS = ["hi", "id", "tr", "sw", "ne"]
ALL_LANGS = [
    "en", "es", "fr", "de", "ja", "zh", "ar", "ru", "th",
    "hi", "id", "tr", "yo", "sw", "ne",
]
TASKS = [
    "no_task",
    "sentiment_analysis",
    "rephrase",
    "summarize",
    "translate",
    "pos_tagging",
    "word_shuffling",
    "character_counting",
]
NAME_PATTERNS = {
    "en": r"into (.+?): '",
    "es": r"al idioma (.+?): '",
    "fr": r"dans la langue (.+?): '",
    "de": r"in die Sprache (.+?): '",
    "ja": r"を(.+?)に翻訳",
    "zh": r"翻译成(.+?)：",
    "ar": r"إلى (.+?): '",
    "ru": r"на (.+?) язык: '",
    "th": r"เป็นภาษา(.+?): '",
    "yo": r"sí (.+?): '",
}


def main() -> None:
    prev = {}
    if OUT.exists():
        prev = json.loads(OUT.read_text())

    ds = load_dataset("Sakonii/Multitask-Multilingual-Refusal")
    df = pd.concat(
        [split.to_pandas().assign(split=name) for name, split in ds.items()],
        ignore_index=True,
    )

    templates: dict[str, dict[str, str]] = {}
    lang_names: dict[str, dict[str, str]] = {src: {} for src in ALL_LANGS}

    for lang in KEEP_LANGS:
        templates[lang] = {}
        for task in TASKS:
            if task == "translate":
                continue
            sub = df[(df["lang"] == lang) & (df["intended_task"] == task)]
            if sub.empty:
                raise SystemExit(f"Sakonii missing {lang}/{task}")
            row = sub.iloc[0]
            templates[lang][task] = row["prompt"].replace(row["content"], "{content}", 1)

        sub = df[(df["lang"] == lang) & (df["intended_task"] == "translate")]
        shells: set[str] = set()
        for tgt, group in sub.groupby("translate_target"):
            row = group.iloc[0]
            prompt = row["prompt"].replace(row["content"], "{content}", 1)
            match = re.search(NAME_PATTERNS[lang], prompt)
            if not match:
                raise SystemExit(f"Cannot parse translate name {lang}/{tgt}: {prompt[:120]}")
            name = match.group(1)
            lang_names[lang][str(tgt)] = name
            shells.add(prompt.replace(name, "{translate_target_name}", 1))
        if len(shells) != 1:
            raise SystemExit(f"Ambiguous translate shell for {lang}: {shells}")
        templates[lang]["translate"] = next(iter(shells))

    # Preserve prior new-lang templates + name maps (Cursor fill).
    prev_templates = prev.get("templates", {})
    prev_names = prev.get("lang_names", {})
    for lang in NEW_LANGS:
        if lang not in prev_templates:
            raise SystemExit(
                f"Missing cursor-localized templates for {lang}. "
                "Re-run localization fill before rebuilding."
            )
        templates[lang] = prev_templates[lang]
        lang_names[lang].update(prev_names.get(lang, {}))

    # Merge prior name-matrix extras (new targets on overlap langs, etc.).
    for src, names in prev_names.items():
        if src in lang_names:
            lang_names[src].update(names)

    out = {
        "pipeline_version": "dataset-pipeline-v1.1-localized",
        "template_sources": {
            **{lang: "already present" for lang in KEEP_LANGS},
            **{lang: "newly added" for lang in NEW_LANGS},
        },
        "tasks": TASKS,
        "langs": ALL_LANGS,
        "templates": templates,
        "lang_names": lang_names,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"Wrote {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
