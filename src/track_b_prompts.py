"""Paper-format localized prompt wrappers for Track B."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.dataset import filter_reviewed
from src.languages import (
    ALL_LANGUAGES,
    TARGET_LANG_HI,
    TRACK_B_SENTIMENT,
    TRACK_B_TRANSLATE,
    target_lang_display,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "track_b" / "track_b_catalog.csv"

# Track B catalog languages (core three + extended when translations exist)
LANGUAGES = ALL_LANGUAGES


def wrap_sentiment(text: str, language: str) -> str:
    if language not in TRACK_B_SENTIMENT:
        raise ValueError(f"Unknown language: {language}")
    return TRACK_B_SENTIMENT[language].format(text=text)


def wrap_translate(text: str, target_lang: str, language: str) -> str:
    if language not in TRACK_B_TRANSLATE:
        raise ValueError(f"Unknown language: {language}")
    if language == "english":
        return TRACK_B_TRANSLATE["english"].format(target_lang=target_lang, text=text)
    target_hi = target_lang_display(target_lang, language="hindi")
    return TRACK_B_TRANSLATE[language].format(target_lang_hi=target_hi, text=text)


def wrap_prompt(
    *,
    intended_task: str,
    source_text: str,
    language: str,
    translate_target_lang: str = "",
    plain_text_en: str = "",
) -> str:
    """Build localized plain_text for a Track B row."""
    if language == "english":
        return plain_text_en

    if intended_task == "sentiment_analysis":
        return wrap_sentiment(source_text, language)
    if intended_task == "translate":
        return wrap_translate(source_text, translate_target_lang, language)
    raise ValueError(f"Unsupported intended_task: {intended_task}")


def build_track_b_catalog(
    selected_df: pd.DataFrame,
    reviewed_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    One row per (sample_id, language) for reviewed translations.

    English uses original plain_text; Hindi/Hinglish use localized wrappers.
    """
    reviewed = filter_reviewed(reviewed_df)
    if reviewed.empty:
        logger.warning("No reviewed translations; catalog will be empty.")
        return pd.DataFrame()

    rev_by_id = reviewed.set_index("sample_id")
    rows: list[dict] = []

    for _, sel in selected_df.iterrows():
        sid = str(sel["sample_id"])
        if sid not in rev_by_id.index:
            continue
        rev = rev_by_id.loc[sid]
        intended_task = str(sel["intended_task"])
        translate_target = str(sel.get("translate_target_lang", ""))

        for lang in LANGUAGES:
            if lang == "english":
                localized_text = str(sel["plain_text"])
                source_for_row = str(sel["source_text"])
            else:
                if lang not in rev.index:
                    continue
                localized_source = str(rev[lang]).strip()
                if not localized_source:
                    continue
                localized_text = wrap_prompt(
                    intended_task=intended_task,
                    source_text=localized_source,
                    language=lang,
                    translate_target_lang=translate_target,
                )
                source_for_row = localized_source

            rows.append(
                {
                    "prompt_id": f"{sid}__{lang}",
                    "sample_id": sid,
                    "language": lang,
                    "plain_text": localized_text,
                    "source_text": source_for_row,
                    "text_type": sel["text_type"],
                    "intended_task": intended_task,
                    "task_label": sel["task_label"],
                    "is_benign_task": sel["is_benign_task"],
                    "translate_target_lang": translate_target,
                    "priority_tier": sel.get("priority_tier", ""),
                }
            )

    catalog = pd.DataFrame(rows)
    logger.info(
        "Built Track B catalog: %d rows (%d sample_ids × up to %d langs)",
        len(catalog),
        catalog["sample_id"].nunique() if len(catalog) else 0,
        len(LANGUAGES),
    )
    return catalog


def write_track_b_catalog(
    selected_df: pd.DataFrame,
    reviewed_df: pd.DataFrame,
    out_path: Path | None = None,
) -> pd.DataFrame:
    out_path = out_path or DEFAULT_CATALOG_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog = build_track_b_catalog(selected_df, reviewed_df)
    catalog.to_csv(out_path, index=False)
    logger.info("Saved Track B catalog (%d rows) to %s", len(catalog), out_path)
    return catalog
