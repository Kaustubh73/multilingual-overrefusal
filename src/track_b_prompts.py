"""Paper-format localized prompt wrappers for Track B."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd

from src.languages import (
    ALL_LANGUAGES,
    TARGET_LANG_HI,
    TRACK_B_CRYPTO,
    TRACK_B_RAGQA,
    TRACK_B_REPHRASE,
    TRACK_B_SENTIMENT,
    TRACK_B_TRANSLATE,
    target_lang_display,
)

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "track_b" / "track_b_catalog.csv"
TRANSLATED_BENCHMARK_PATH = PROJECT_ROOT / "data" / "track_b" / "track_b_benchmark_translated.csv"

# Track B catalog languages (core three only for now)
LANGUAGES = ("english", "hindi", "hinglish")


def _strip_crypto_examples(plain_text: str) -> str:
    """Strip worked examples from English cryptanalysis prompts.

    The HuggingFace dataset includes 4 worked examples (~1400 chars) in English
    cryptanalysis prompts that Hindi/Hinglish versions don't have. This makes
    English prompts 2.5x longer, which could affect refusal behavior.

    Example block starts with 'Here are examples of encrypted texts' and continues
    to the end of the prompt.
    """
    marker = "Here are examples of encrypted texts"
    idx = plain_text.find(marker)
    if idx >= 0:
        return plain_text[:idx].rstrip()
    return plain_text


def _extract_source_text(plain_text: str, intended_task: str) -> str:
    """Extract the inner source text from an English prompt."""
    if intended_task in ("sentiment_analysis",):
        m = re.search(r"(?:neutral|positive|negative)[\"']:\s*['\"](.+?)['\"]\s*$", plain_text, re.DOTALL)
        return m.group(1).strip() if m else plain_text
    if intended_task in ("rephrase",):
        m = re.search(r"(?:if any)[\"']:\s*['\"](.+?)['\"]\s*$", plain_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"['\"](.+?)['\"]\s*$", plain_text, re.DOTALL)
        return m.group(1).strip() if m else plain_text
    if intended_task == "translate":
        m = re.search(r"into\s+\w+:\s*['\"](.+?)['\"]\s*$", plain_text, re.DOTALL)
        return m.group(1).strip() if m else plain_text
    if intended_task == "cryptanalysis":
        m = re.search(r"Encrypted text:\s*['\"](.+?)['\"]", plain_text, re.DOTALL)
        return m.group(1).strip() if m else plain_text
    if intended_task == "rag_qa":
        m = re.search(r"Context:\s*(.*?)\s*Question:\s*(.*)", plain_text, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return plain_text, ""
    return plain_text


def _load_translations() -> dict[str, dict]:
    """Load pre-translated benchmark. Returns {(sample_id, task): row_dict}."""
    if not TRANSLATED_BENCHMARK_PATH.exists():
        logger.warning("No pre-translated benchmark at %s — using English fallback", TRANSLATED_BENCHMARK_PATH)
        return {}
    df = pd.read_csv(TRANSLATED_BENCHMARK_PATH, dtype=str, keep_default_na=False)
    lookup: dict[str, dict] = {}
    for _, row in df.iterrows():
        key = f"{row['sample_id']}|{row['intended_task']}"
        lookup[key] = row.to_dict()
    logger.info("Loaded %d pre-translated entries from %s", len(lookup), TRANSLATED_BENCHMARK_PATH)
    return lookup


def wrap_prompt_full(
    *,
    intended_task: str,
    plain_text_en: str,
    language: str,
    translate_target_lang: str = "",
    translated_source: str = "",
    translated_context: str = "",
    translated_question: str = "",
) -> str:
    """Build localized plain_text for any task from the English prompt.

    For non-English languages, uses pre-translated source texts when provided.
    """
    if language == "english":
        return plain_text_en

    if intended_task == "sentiment_analysis":
        source = translated_source or _extract_source_text(plain_text_en, "sentiment_analysis")
        return TRACK_B_SENTIMENT[language].format(text=source)

    if intended_task == "rephrase":
        source = translated_source or _extract_source_text(plain_text_en, "rephrase")
        return TRACK_B_REPHRASE[language].format(text=source)

    if intended_task == "translate":
        source = translated_source or _extract_source_text(plain_text_en, "translate")
        target_hi = target_lang_display(translate_target_lang, language="hindi")
        return TRACK_B_TRANSLATE[language].format(target_lang_hi=target_hi, text=source)

    if intended_task == "cryptanalysis":
        source = translated_source or _extract_source_text(plain_text_en, "cryptanalysis")
        return TRACK_B_CRYPTO[language].format(text=source)

    if intended_task == "rag_qa":
        if translated_context and translated_question:
            context, question = translated_context, translated_question
        else:
            context, question = _extract_source_text(plain_text_en, "rag_qa")
        return TRACK_B_RAGQA[language].format(context=context, question=question)

    raise ValueError(f"Unsupported intended_task: {intended_task}")


def build_full_track_b_catalog(
    bench_df: pd.DataFrame,
) -> pd.DataFrame:
    """Build full Track B catalog: 270 rows × 3 languages = 810 rows.

    Uses pre-translated source texts for Hindi/Hinglish when available.
    """
    trans = _load_translations()
    use_translations = len(trans) > 0
    if not use_translations:
        logger.warning("No pre-translations available — prompts will use English source text")

    rows: list[dict] = []

    for _, row in bench_df.iterrows():
        sid = str(row["sample_id"])
        intended_task = str(row["intended_task"])
        plain_text_en = str(row["plain_text"])
        text_type = str(row["text_type"])
        task_label = str(row.get("task_label", ""))
        is_benign = bool(row.get("is_benign_task", False))

        # Extract translate target language from prompt
        translate_target = ""
        if intended_task == "translate":
            m = re.search(r"into\s+(\w+):", plain_text_en)
            translate_target = m.group(1) if m else ""

        # Look up pre-translated source texts
        tkey = f"{sid}|{intended_task}"
        trow = trans.get(tkey, {}) if use_translations else {}

        for lang in LANGUAGES:
            # Get translated source text for this language
            if lang == "hindi":
                ts = trow.get("source_text_hi", "")
                tc = trow.get("context_hi", "")
                tq = trow.get("question_hi", "")
            elif lang == "hinglish":
                ts = trow.get("source_text_hl", "")
                tc = trow.get("context_hl", "")
                tq = trow.get("question_hl", "")
            else:
                ts, tc, tq = "", "", ""

            localized_text = wrap_prompt_full(
                intended_task=intended_task,
                plain_text_en=plain_text_en,
                language=lang,
                translate_target_lang=translate_target,
                translated_source=ts,
                translated_context=tc,
                translated_question=tq,
            )

            # Strip worked examples from English cryptanalysis prompts to
            # make them comparable in length to Hindi/Hinglish versions.
            if lang == "english" and intended_task == "cryptanalysis":
                localized_text = _strip_crypto_examples(localized_text)

            # Flag degenerate translate rows (Hindi/Hinglish prompt → Hindi target)
            is_degenerate = False
            if intended_task == "translate" and lang != "english":
                if translate_target.lower() == "hindi":
                    is_degenerate = True

            rows.append(
                {
                    "prompt_id": f"{sid}__{lang}",
                    "sample_id": sid,
                    "language": lang,
                    "plain_text": localized_text,
                    "text_type": text_type,
                    "intended_task": intended_task,
                    "task_label": task_label,
                    "is_benign_task": str(is_benign),
                    "translate_target_lang": translate_target,
                    "is_degenerate_translate": str(is_degenerate),
                }
            )

    catalog = pd.DataFrame(rows)
    logger.info(
        "Built full Track B catalog: %d rows (%d sample_ids × %d langs) [translations: %s]",
        len(catalog),
        catalog["sample_id"].nunique(),
        len(LANGUAGES),
        "ON" if use_translations else "OFF",
    )
    return catalog
