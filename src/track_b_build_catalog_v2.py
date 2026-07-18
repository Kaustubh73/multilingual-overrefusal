"""Build Track B catalog from reviewed translations (v2)."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.languages import ALL_LANGUAGES
from src.track_b_prompts import wrap_prompt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REVIEWED_PATH = PROJECT_ROOT / "data" / "track_b" / "review_translations_v2.csv"
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "data" / "track_b" / "selected_rows_v2.csv"
DEFAULT_CATALOG_PATH = PROJECT_ROOT / "data" / "track_b" / "track_b_catalog_v2.csv"


def build_catalog(
    reviewed_path: Path | None = None,
    selected_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    """Build catalog from reviewed translations."""
    reviewed_path = reviewed_path or DEFAULT_REVIEWED_PATH
    selected_path = selected_path or DEFAULT_SELECTED_PATH
    out_path = out_path or DEFAULT_CATALOG_PATH

    reviewed = pd.read_csv(reviewed_path, dtype=str, keep_default_na=False)
    selected = pd.read_csv(selected_path, dtype=str, keep_default_na=False)

    # Filter to reviewed rows only
    reviewed_mask = reviewed["reviewed"].str.lower().isin(["true", "1", "yes"])
    reviewed_rows = reviewed[reviewed_mask].copy()
    logger.info("Reviewed rows: %d / %d", len(reviewed_rows), len(reviewed))

    if reviewed_rows.empty:
        logger.warning("No reviewed translations found.")
        return pd.DataFrame()

    # Build catalog
    catalog_rows: list[dict] = []
    for _, rev in reviewed_rows.iterrows():
        sid = str(rev["sample_id"])
        sel_row = selected[selected["sample_id"] == sid]
        if sel_row.empty:
            logger.warning("sample_id %s not found in selected rows", sid)
            continue
        sel = sel_row.iloc[0]

        intended_task = str(sel["intended_task"])
        translate_target = str(sel.get("translate_target_lang", ""))

        for lang in ALL_LANGUAGES:
            # For Hindi/Hinglish translate prompts, avoid targeting Hindi (text is already Hindi)
            localized_translate_target = translate_target
            if lang in ("hindi", "hinglish") and intended_task == "translate" and translate_target == "hindi":
                localized_translate_target = "english"
            if lang == "english":
                localized_text = str(sel["plain_text"])
                source_for_row = str(sel["source_text"])
            elif lang == "hindi":
                localized_text = str(rev.get("hindi", "")).strip()
                source_for_row = localized_text
                if not localized_text:
                    continue
                # Wrap with prompt template
                localized_text = wrap_prompt(
                    intended_task=intended_task,
                    source_text=localized_text,
                    language=lang,
                    translate_target_lang=localized_translate_target,
                )
            elif lang == "hinglish":
                localized_text = str(rev.get("hinglish", "")).strip()
                source_for_row = localized_text
                if not localized_text:
                    continue
                # Wrap with prompt template
                localized_text = wrap_prompt(
                    intended_task=intended_task,
                    source_text=localized_text,
                    language=lang,
                    translate_target_lang=localized_translate_target,
                )
            else:
                # Other languages - skip for now
                continue

            catalog_rows.append({
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
                "priority_tier": sel.get("priority_tier", "standard"),
                "meaning_score": str(rev.get("meaning_score", "")),
                "naturalness_score": str(rev.get("naturalness_score", "")),
            })

    catalog = pd.DataFrame(catalog_rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    catalog.to_csv(out_path, index=False)
    logger.info("Built catalog: %d rows (%d sample_ids)", len(catalog), catalog["sample_id"].nunique())
    return catalog


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--reviewed", type=str, default=None)
    parser.add_argument("--selected", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    reviewed = Path(args.reviewed) if args.reviewed else None
    selected = Path(args.selected) if args.selected else None
    out = Path(args.out) if args.out else None

    build_catalog(reviewed_path=reviewed, selected_path=selected, out_path=out)
