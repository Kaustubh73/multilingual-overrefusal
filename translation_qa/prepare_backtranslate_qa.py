#!/usr/bin/env python3
"""
Back-translation QA for forward Google Translate quality.

Step A — emit one workbook with a sheet per language:
  .venv/bin/python translation_qa/prepare_backtranslate_qa.py --prepare \\
      --src data/dataset_v1/final_gt_full.parquet \\
      --outdir data/dataset_v1/sheets/backtranslate

  → data/dataset_v1/sheets/backtranslate/backtranslate.xlsx

Step B — upload to Google Sheets, wait for back_en, download as xlsx:
  .venv/bin/python translation_qa/prepare_backtranslate_qa.py --score \\
      --filled data/dataset_v1/sheets/backtranslate/backtranslate_filled.xlsx \\
      --out data/dataset_v1/backtranslate_scored.parquet
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from translation_qa.backtranslate_metrics import DEFAULT_EMBED_MODEL, score_pairs  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("backtranslate_qa")

DEFAULT_SRC = ROOT / "data" / "dataset_v1" / "final_gt_full.parquet"
DEFAULT_OUTDIR = ROOT / "data" / "dataset_v1" / "sheets" / "backtranslate"
DEFAULT_XLSX = DEFAULT_OUTDIR / "backtranslate.xlsx"
DEFAULT_FILLED = DEFAULT_OUTDIR / "backtranslate_filled.xlsx"
DEFAULT_SCORED = ROOT / "data" / "dataset_v1" / "backtranslate_scored.parquet"

NON_EN_LANGS = [
    "es", "fr", "de", "ja", "zh", "ar", "ru", "th", "hi", "id", "tr", "yo", "sw", "ne",
]

BRACKET = {
    "en": "high",
    "es": "high",
    "fr": "high",
    "de": "mid_high",
    "ja": "mid_high",
    "zh": "mid_high",
    "ar": "mid",
    "ru": "mid",
    "th": "mid",
    "hi": "mid_low",
    "id": "mid_low",
    "tr": "mid_low",
    "yo": "low",
    "sw": "low",
    "ne": "low",
}

SHEET_COLS = [
    "content_id",
    "lang",
    "label",
    "split",
    "category",
    "en_content",
    "lang_content",
    "back_en",
    "mt_engine",
]


def _bracket(lang: str) -> str:
    return BRACKET.get(lang, "unknown")


def _load_content_pairs(
    src: Path, langs: list[str], limit: int | None
) -> pd.DataFrame:
    """One row per (content_id, lang): EN seed + forward translation (no_task)."""
    df = pd.read_parquet(src)
    bare = df[df["intended_task"] == "no_task"].copy()

    en = (
        bare[bare["lang"] == "en"][
            ["content_id", "content", "label", "split", "category"]
        ]
        .rename(columns={"content": "en_content"})
        .drop_duplicates("content_id")
    )

    rows = []
    for lang in langs:
        sub = bare[bare["lang"] == lang][["content_id", "content", "mt_engine"]].rename(
            columns={"content": "lang_content"}
        )
        if sub.empty:
            log.warning("No rows for lang=%s — skip", lang)
            continue
        merged = en.merge(sub, on="content_id", how="inner")
        merged["lang"] = lang
        rows.append(merged)

    if not rows:
        raise SystemExit("No language pairs loaded")

    out = pd.concat(rows, ignore_index=True)
    out = out.sort_values(["lang", "content_id"]).reset_index(drop=True)

    if limit is not None:
        out = (
            out.groupby("lang", group_keys=False)
            .head(limit)
            .reset_index(drop=True)
        )
    return out


def _formula_rows(pairs: pd.DataFrame) -> pd.DataFrame:
    """Build sheet rows; back_en formula references lang_content column G."""
    records = []
    for i, r in enumerate(pairs.itertuples(index=False), start=2):
        cat = r.category if pd.notna(r.category) else ""
        records.append(
            {
                "content_id": r.content_id,
                "lang": r.lang,
                "label": r.label,
                "split": r.split,
                "category": cat,
                "en_content": r.en_content,
                "lang_content": r.lang_content,
                "back_en": f'=GOOGLETRANSLATE(G{i},"{r.lang}","en")',
                "mt_engine": r.mt_engine,
            }
        )
    return pd.DataFrame(records)[SHEET_COLS]


def prepare(
    src: Path,
    outdir: Path,
    langs: list[str],
    limit: int | None,
    *,
    xlsx_name: str = "backtranslate.xlsx",
) -> Path:
    """Write one .xlsx with a worksheet per language."""
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / xlsx_name
    pairs = _load_content_pairs(src, langs, limit)

    # Drop stale per-lang CSVs from the earlier layout (if any).
    for stale in outdir.glob("backtranslate_*.csv"):
        stale.unlink()
        log.info("Removed stale %s", stale.name)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for lang, group in pairs.groupby("lang", sort=True):
            frame = _formula_rows(group.reset_index(drop=True))
            # Sheet name = ISO code (Excel/Sheets limit 31 chars).
            frame.to_excel(writer, sheet_name=str(lang), index=False)
            log.info("Sheet %s (%d rows)", lang, len(frame))

    readme = outdir / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Back-translation QA (Google Sheets)",
                "",
                "Workbook: `backtranslate.xlsx` — one worksheet per language.",
                "",
                "Forward `lang_content` → `GOOGLETRANSLATE` back to English (`back_en`),",
                "then score against `en_content` (chrF++, BLEU, embedding cosine).",
                "",
                "## Steps",
                "",
                "1. Upload `backtranslate.xlsx` to Drive → open with **Google Sheets**",
                "   (formulas use `GOOGLETRANSLATE`, which is Sheets-specific).",
                "2. For each language tab, wait until `back_en` shows English text",
                "   (no leading `=`).",
                "3. File → Download → Microsoft Excel (`.xlsx`) → save as",
                "   `backtranslate_filled.xlsx` in this folder.",
                "4. Score:",
                "",
                "```bash",
                ".venv/bin/python translation_qa/prepare_backtranslate_qa.py --score \\",
                "  --filled data/dataset_v1/sheets/backtranslate/backtranslate_filled.xlsx \\",
                "  --out data/dataset_v1/backtranslate_scored.parquet",
                "```",
                "",
                f"Languages (sheets): {', '.join(langs)}",
                "Column G = `lang_content`; column H = `back_en` formula.",
                "",
                "Round-trip is a proxy (same engine forward/back can cancel errors).",
                "Compare languages relatively; do not treat cosine alone as absolute quality.",
                "",
            ]
        )
    )
    log.info("Wrote %s (%d language sheets)", out_path, pairs["lang"].nunique())
    log.info("README → %s", readme)
    return out_path


def _validate_filled_frame(raw: pd.DataFrame, label: str) -> pd.DataFrame:
    need = {"en_content", "lang_content", "back_en", "lang", "content_id"}
    missing = need - set(raw.columns)
    if missing:
        raise SystemExit(f"{label}: missing columns {sorted(missing)}")
    bad = raw["back_en"].astype(str).str.startswith("=").sum()
    if bad:
        raise SystemExit(
            f"{label}: {bad} unresolved GOOGLETRANSLATE formulas in back_en. "
            "Wait for Sheets, re-download, then --score again."
        )
    return raw


def _load_filled(path: Path) -> pd.DataFrame:
    """Load a filled multi-sheet xlsx (or a single CSV for smoke tests)."""
    if not path.exists():
        raise SystemExit(f"Missing filled file: {path}")

    if path.suffix.lower() in {".xlsx", ".xlsm", ".xls"}:
        sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
        frames = []
        for name, raw in sheets.items():
            if raw is None or raw.empty:
                log.warning("Skipping empty sheet %s", name)
                continue
            raw = _validate_filled_frame(raw, f"{path.name}:{name}")
            frames.append(raw)
            log.info("Loaded sheet %s (%d rows)", name, len(raw))
        if not frames:
            raise SystemExit(f"No data sheets in {path}")
        return pd.concat(frames, ignore_index=True)

    if path.suffix.lower() == ".csv":
        raw = pd.read_csv(path)
        raw = _validate_filled_frame(raw, path.name)
        log.info("Loaded CSV %s (%d rows)", path, len(raw))
        return raw

    raise SystemExit(f"Unsupported filled file type: {path.suffix}")


def score(
    filled: Path,
    out: Path,
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
) -> pd.DataFrame:
    raw = _load_filled(filled)
    metrics = score_pairs(
        raw["en_content"].tolist(),
        raw["back_en"].tolist(),
        model_name=model_name,
        batch_size=batch_size,
    )
    scored = raw.copy()
    for key, values in metrics.items():
        scored[key] = values
    scored["resource_bracket"] = scored["lang"].map(_bracket)
    scored["en_len"] = scored["en_content"].astype(str).str.len()
    scored["back_en_len"] = scored["back_en"].astype(str).str.len()

    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(out, index=False)
    scored.to_csv(out.with_suffix(".csv"), index=False)

    summary = (
        scored.groupby(["lang", "resource_bracket"], as_index=False)
        .agg(
            n=("content_id", "count"),
            embed_cosine_mean=("embed_cosine", "mean"),
            embed_cosine_median=("embed_cosine", "median"),
            chrf_mean=("chrf", "mean"),
            chrf_median=("chrf", "median"),
        )
        .sort_values(
            ["embed_cosine_mean", "chrf_mean"], ascending=[False, False]
        )
        .reset_index(drop=True)
    )
    for col in summary.columns:
        if col not in ("lang", "resource_bracket", "n"):
            summary[col] = summary[col].round(4)

    summary_path = out.with_name(out.stem + "_lang_summary.csv")
    summary.to_csv(summary_path, index=False)
    log.info("Scored rows → %s (%d)", out, len(scored))
    log.info("Summary → %s", summary_path)
    log.info(
        "Per-lang (cosine then chrF):\n%s",
        summary[
            ["lang", "n", "embed_cosine_mean", "chrf_mean"]
        ].to_string(index=False),
    )
    return scored


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--prepare",
        action="store_true",
        help="Emit backtranslate.xlsx with one sheet per language",
    )
    mode.add_argument(
        "--score",
        action="store_true",
        help="Score filled workbook (or CSV) — en_content vs back_en",
    )
    p.add_argument("--src", type=Path, default=DEFAULT_SRC)
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument(
        "--xlsx-name",
        default="backtranslate.xlsx",
        help="Output workbook filename under --outdir",
    )
    p.add_argument(
        "--langs",
        default=",".join(NON_EN_LANGS),
        help="Comma-separated ISO codes (default: all 14 non-en)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max rows per language (smoke)",
    )
    p.add_argument(
        "--filled",
        type=Path,
        default=DEFAULT_FILLED,
        help="Filled multi-sheet xlsx (or a single CSV)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_SCORED)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    if args.prepare:
        if not args.src.exists():
            raise SystemExit(f"Missing source parquet: {args.src}")
        prepare(
            args.src,
            args.outdir,
            langs,
            args.limit,
            xlsx_name=args.xlsx_name,
        )
        return

    score(
        args.filled,
        args.out,
        model_name=args.embed_model,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
