#!/usr/bin/env python3
"""Report Track B translation review progress against G-translate-60 and G-n30.

Read-only — no GPU, no inference. Reads reviewed_translations CSV (prefers
reviewed_translations.human.csv when it contains SOP-valid rows).

Usage:
  python tools/translation_progress.py
  python tools/translation_progress.py --json
  python tools/translation_progress.py --csv data/track_b/reviewed_translations.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow `python tools/translation_progress.py` from repo root
_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.track_b_translate import (  # noqa: E402
    DEFAULT_REVIEWED_PATH,
    filter_sop_valid,
    resolve_reviewed_path,
)


def _task_of(sample_id: str) -> str:
    sid = (sample_id or "").lower()
    if "translate" in sid or sid.startswith("test_0") and int(
        "".join(c for c in sid if c.isdigit()) or "0"
    ) < 91:
        return "translate"
    return "sentiment"


def _task_of_row(row) -> str:
    notes = str(row.get("notes", "")).lower()
    if "sentiment" in notes:
        return "sentiment"
    if "translate" in notes:
        return "translate"
    sid = str(row.get("sample_id", ""))
    if sid.startswith("test_"):
        try:
            num = int(sid.split("_")[1])
            return "translate" if num < 91 else "sentiment"
        except (IndexError, ValueError):
            pass
    return "unknown"


def compute_progress(csv_path: Path | None = None) -> dict:
    path = resolve_reviewed_path(csv_path or DEFAULT_REVIEWED_PATH)
    import pandas as pd

    raw = pd.read_csv(path, dtype=str, keep_default_na=False)
    valid = filter_sop_valid(raw)

    smoke = raw[
        raw["notes"].astype(str).str.lower().str.contains("smoke", na=False)
    ]

    translate_valid = sum(
        1 for _, r in valid.iterrows() if _task_of_row(r) == "translate"
    )
    sentiment_valid = sum(
        1 for _, r in valid.iterrows() if _task_of_row(r) == "sentiment"
    )
    sample_ids = set(valid["sample_id"].astype(str).tolist())
    langs_per_id = 3  # en shell + hi + hing reviewed together per row

    return {
        "csv_path": str(path),
        "total_rows": len(raw),
        "smoke_scaffold_rows": len(smoke),
        "sop_valid_rows": len(valid),
        "sop_valid_translate": translate_valid,
        "sop_valid_sentiment": sentiment_valid,
        "unique_sample_ids": len(sample_ids),
        "g_translate_60": {
            "target": 60,
            "current": translate_valid,
            "met": translate_valid >= 60,
        },
        "g_n30": {
            "target_sample_ids": 30,
            "current_sample_ids": len(sample_ids),
            "met": len(sample_ids) >= 30,
        },
        "g_full_115": {
            "target": 115,
            "current": len(valid),
            "met": len(valid) >= 115,
        },
        "langs_per_sample_id": langs_per_id,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=None, help="reviewed_translations CSV")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    args = parser.parse_args()

    stats = compute_progress(args.csv)

    if args.json:
        print(json.dumps(stats, indent=2))
        return 0

    g60 = stats["g_translate_60"]
    gn30 = stats["g_n30"]
    print(f"CSV: {stats['csv_path']}")
    print(f"SOP-valid: {stats['sop_valid_rows']}/115 "
          f"(translate {stats['sop_valid_translate']}, "
          f"sentiment {stats['sop_valid_sentiment']})")
    print(f"Smoke scaffold (not counted): {stats['smoke_scaffold_rows']}")
    print(f"G-translate-60: {g60['current']}/{g60['target']} "
          f"{'✓' if g60['met'] else '—'}")
    print(f"G-n30: {gn30['current_sample_ids']}/{gn30['target_sample_ids']} sample_ids "
          f"{'✓' if gn30['met'] else '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
