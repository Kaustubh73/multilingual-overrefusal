"""Generate translation prompts for manual LLM use (no API calls)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.languages import translation_languages

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASE_PATH = PROJECT_ROOT / "data" / "raw" / "base_prompts.csv"
DEFAULT_OUT_PATH = PROJECT_ROOT / "data" / "translations" / "translation_requests.csv"

TRANSLATION_RULES = """
Rules:
- Preserve exact meaning.
- Preserve ambiguity.
- Preserve metaphor.
- Preserve emotional tone.
- Do not soften violent wording.
- Do not intensify harmful wording.
- Avoid formal textbook Hindi.
- Hinglish should resemble real Indian conversational usage.
- If preserving meaning was difficult, explain briefly after the blocks.
""".strip()


def generate_translation_prompt(text: str, languages: tuple[str, ...] | None = None) -> str:
    """Build a prompt for manual LLM translation into target languages."""
    languages = languages or translation_languages()
    blocks = "\n\n".join(f"{lang.title()}:\n..." for lang in languages)
    lang_list = ", ".join(languages)
    return f"""Translate the following text into {lang_list} while preserving meaning.

Source text:
{text}

Return your answer in exactly this format (include the labels):

{blocks}

{TRANSLATION_RULES}
"""


def export_translation_requests(
    base_path: Path | None = None,
    out_path: Path | None = None,
    languages: tuple[str, ...] | None = None,
) -> Path:
    """Export translation_requests.csv from base_prompts.csv."""
    base_path = base_path or DEFAULT_BASE_PATH
    out_path = out_path or DEFAULT_OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base = pd.read_csv(base_path)
    rows = []
    for _, row in base.iterrows():
        english = str(row["base_text"])
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                "english": english,
                "translation_prompt": generate_translation_prompt(english, languages=languages),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d translation requests to %s", len(df), out_path)
    return out_path
