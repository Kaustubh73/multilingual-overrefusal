"""Translation quality assurance checks before evaluation."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.languages import ALL_LANGUAGES, SCRIPT_FAMILIES, translation_languages

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
ARABIC_RE = re.compile(r"[\u0600-\u06FF]")
JAPANESE_RE = re.compile(r"[\u3040-\u30FF\u4E00-\u9FFF]")
LATIN_RE = re.compile(r"[A-Za-z]")

SOP_MIN_SCORE = 4
MIN_LENGTH_RATIO = 0.15
MAX_LENGTH_RATIO = 6.0

QA_ISSUE_CODES = (
    "empty",
    "english_copy",
    "script_mismatch",
    "length_ratio",
    "low_meaning_score",
    "low_naturalness_score",
    "not_reviewed",
    "smoke_scaffold",
)


def _script_family(text: str) -> str:
    counts = {
        "devanagari": len(DEVANAGARI_RE.findall(text)),
        "arabic": len(ARABIC_RE.findall(text)),
        "japanese": len(JAPANESE_RE.findall(text)),
        "latin": len(LATIN_RE.findall(text)),
    }
    if max(counts.values()) == 0:
        return "unknown"
    return max(counts, key=counts.get)


def _is_smoke_scaffold(notes: str) -> bool:
    n = str(notes or "").lower()
    return "smoke-only" in n or "smoke scaffold" in n


def _parse_reviewed(value) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in ("true", "1", "yes")


def check_translation_row(
    *,
    english: str,
    translation: str,
    language: str,
    reviewed: bool = False,
    meaning_score: float | None = None,
    naturalness_score: float | None = None,
    notes: str = "",
    require_reviewed: bool = True,
) -> dict:
    """
    Validate a single (english, translation) pair for one language.

    Returns dict with ``passed``, ``issues`` (list of QA_ISSUE_CODES), and ``warnings``.
    """
    issues: list[str] = []
    warnings: list[str] = []
    english = str(english or "").strip()
    translation = str(translation or "").strip()

    if _is_smoke_scaffold(notes):
        issues.append("smoke_scaffold")
    if require_reviewed and not reviewed:
        issues.append("not_reviewed")
    if not translation:
        issues.append("empty")
        return {"language": language, "passed": False, "issues": issues, "warnings": warnings}

    if english and translation.lower() == english.lower():
        issues.append("english_copy")

    expected_script = SCRIPT_FAMILIES.get(language)
    if expected_script:
        actual = _script_family(translation)
        if actual != "unknown" and actual != expected_script:
            issues.append("script_mismatch")

    if english:
        ratio = len(translation) / max(len(english), 1)
        if ratio < MIN_LENGTH_RATIO or ratio > MAX_LENGTH_RATIO:
            issues.append("length_ratio")

    if meaning_score is not None and meaning_score < SOP_MIN_SCORE:
        issues.append("low_meaning_score")
    if naturalness_score is not None and naturalness_score < SOP_MIN_SCORE:
        issues.append("low_naturalness_score")

    passed = len(issues) == 0
    return {"language": language, "passed": passed, "issues": issues, "warnings": warnings}


def qa_reviewed_translations(
    df: pd.DataFrame,
    *,
    id_col: str = "prompt_id",
    english_col: str = "english",
    languages: tuple[str, ...] | None = None,
    require_reviewed: bool = True,
) -> pd.DataFrame:
    """
    Run QA on a reviewed_translations-style dataframe.

    Returns one row per (id_col, language) with pass/fail and issue codes.
    """
    languages = languages or translation_languages()
    rows: list[dict] = []

    for _, row in df.iterrows():
        english = str(row.get(english_col, ""))
        reviewed = _parse_reviewed(row.get("reviewed", False))
        notes = str(row.get("notes", ""))
        meaning = row.get("meaning_score")
        naturalness = row.get("naturalness_score")
        try:
            meaning_f = float(meaning) if str(meaning).strip() not in ("", "nan") else None
        except (TypeError, ValueError):
            meaning_f = None
        try:
            naturalness_f = (
                float(naturalness) if str(naturalness).strip() not in ("", "nan") else None
            )
        except (TypeError, ValueError):
            naturalness_f = None

        for lang in languages:
            if lang not in df.columns:
                continue
            result = check_translation_row(
                english=english,
                translation=str(row.get(lang, "")),
                language=lang,
                reviewed=reviewed,
                meaning_score=meaning_f,
                naturalness_score=naturalness_f,
                notes=notes,
                require_reviewed=require_reviewed,
            )
            rows.append(
                {
                    id_col: str(row.get(id_col, row.get("sample_id", ""))),
                    "language": lang,
                    "passed": result["passed"],
                    "issues": ";".join(result["issues"]),
                    "warnings": ";".join(result["warnings"]),
                }
            )

    return pd.DataFrame(rows)


def filter_qa_passing(
    df: pd.DataFrame,
    qa_report: pd.DataFrame,
    *,
    id_col: str = "prompt_id",
    languages: tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Keep only rows where all requested language translations passed QA."""
    languages = languages or translation_languages()
    failing_ids: set[str] = set()
    for lang in languages:
        lang_fail = qa_report[(qa_report["language"] == lang) & (~qa_report["passed"])]
        failing_ids.update(lang_fail[id_col].astype(str).tolist())
    out = df[~df[id_col].astype(str).isin(failing_ids)].copy()
    logger.info(
        "QA filter: %d -> %d rows (%d ids failed QA)",
        len(df),
        len(out),
        len(failing_ids),
    )
    return out


def summarize_qa(qa_report: pd.DataFrame) -> dict:
    """Aggregate QA pass rates per language."""
    if qa_report.empty:
        return {"n_rows": 0, "by_language": {}}
    by_lang: dict[str, dict] = {}
    for lang, group in qa_report.groupby("language"):
        n = len(group)
        passed = int(group["passed"].sum())
        by_lang[str(lang)] = {
            "n": n,
            "passed": passed,
            "pass_rate": passed / n if n else 0.0,
            "top_issues": group["issues"].str.split(";").explode().value_counts().head(5).to_dict(),
        }
    return {"n_rows": len(qa_report), "by_language": by_lang}


def write_qa_report(
    df: pd.DataFrame,
    out_path: Path | None = None,
    *,
    id_col: str = "prompt_id",
    require_reviewed: bool = True,
) -> tuple[pd.DataFrame, Path]:
    """Run QA and write CSV report; return (report_df, path)."""
    out_path = out_path or (PROJECT_ROOT / "results" / "translation_qa_report.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    report = qa_reviewed_translations(df, id_col=id_col, require_reviewed=require_reviewed)
    report.to_csv(out_path, index=False)
    summary = summarize_qa(report)
    logger.info("Translation QA: %s", summary)
    return report, out_path


def assert_catalog_languages_supported(languages: list[str] | tuple[str, ...]) -> None:
    """Raise ValueError if any language is outside ALL_LANGUAGES."""
    unknown = [lang for lang in languages if lang not in ALL_LANGUAGES]
    if unknown:
        raise ValueError(f"Unsupported languages: {unknown}. Supported: {list(ALL_LANGUAGES)}")
