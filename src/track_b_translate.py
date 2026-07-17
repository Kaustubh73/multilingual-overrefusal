"""Track B translation workflow: requests + reviewed_translations CSV."""

from __future__ import annotations

import logging
import os
import re
import time
import json
from collections.abc import Callable
from pathlib import Path
from threading import Event, Thread

import pandas as pd
import requests

from src.languages import translation_column_names
from src.translate import TRANSLATION_RULES, generate_translation_prompt

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SELECTED_PATH = PROJECT_ROOT / "data" / "track_b" / "selected_rows.csv"
DEFAULT_REQUESTS_PATH = PROJECT_ROOT / "data" / "track_b" / "translation_requests.csv"
DEFAULT_REVIEWED_PATH = PROJECT_ROOT / "data" / "track_b" / "reviewed_translations.csv"
HUMAN_REVIEWED_PATH = PROJECT_ROOT / "data" / "track_b" / "reviewed_translations.human.csv"
SOP_MIN_SCORE = 4


def _is_smoke_scaffold(notes: str) -> bool:
    n = str(notes or "").lower()
    return "smoke-only" in n or "smoke scaffold" in n


def filter_sop_valid(df: pd.DataFrame) -> pd.DataFrame:
    """Rows that pass human SOP (reviewed, not smoke scaffold, scores ≥ 4)."""
    if df.empty:
        return df.copy()

    def _valid(row: pd.Series) -> bool:
        reviewed = str(row.get("reviewed", "")).strip().lower() in ("true", "1", "yes")
        if not reviewed or _is_smoke_scaffold(str(row.get("notes", ""))):
            return False
        try:
            meaning = float(row.get("meaning_score") or 0)
            naturalness = float(row.get("naturalness_score") or 0)
        except (TypeError, ValueError):
            return False
        return meaning >= SOP_MIN_SCORE and naturalness >= SOP_MIN_SCORE

    return df[df.apply(_valid, axis=1)].copy()


def resolve_reviewed_path(reviewed_path: Path | None = None) -> Path:
    """Prefer human-reviewed CSV from command-center when it has SOP-valid rows."""
    reviewed_path = reviewed_path or DEFAULT_REVIEWED_PATH
    if HUMAN_REVIEWED_PATH.exists() and HUMAN_REVIEWED_PATH != reviewed_path:
        human_df = pd.read_csv(HUMAN_REVIEWED_PATH, dtype=str, keep_default_na=False)
        if not filter_sop_valid(human_df).empty:
            logger.info("Using human-reviewed CSV: %s", HUMAN_REVIEWED_PATH)
            return HUMAN_REVIEWED_PATH
    return reviewed_path

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_OLLAMA_MODEL = "gemma3:12b"

REVIEWED_COLUMNS = [
    "sample_id",
    "english",
    *translation_column_names(),
    "translation_source",
    "reviewed",
    "meaning_score",
    "naturalness_score",
    "surface_sensitivity_score",
    "notes",
]

HINDI_RE = re.compile(r"Hindi:\s*(.*?)\s*Hinglish:", re.DOTALL | re.IGNORECASE)
HINGLISH_RE = re.compile(r"Hinglish:\s*(.*)", re.DOTALL | re.IGNORECASE)


def _ollama_available() -> bool:
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        return resp.status_code == 200
    except requests.RequestException:
        return False


def _query_ollama(prompt: str, model: str = DEFAULT_OLLAMA_MODEL) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["response"]


def _build_ollama_prompt(text: str) -> str:
    return f"""You are performing research-grade translation for an LLM safety study.

Translate the English text into:
1. Natural Hindi (Devanagari)
2. Natural Hinglish (Roman script used by Indian speakers)

Preserve exact meaning, emotional tone, ambiguity, metaphor, and surface-sensitive wording.
Do not soften or intensify harmful wording unnecessarily.

Text:
{text}

Return EXACTLY:

Hindi:
...

Hinglish:
...
"""


def _build_hf_messages(text: str) -> list[dict[str, str]]:
    """Build a strict machine-readable translation request for a hosted model."""
    return [
        {
            "role": "system",
            "content": (
                "You are a specialized linguistic expert and professional English-to-Hindi and "
                "English-to-Hinglish translator. Your sole function is to translate the supplied "
                "source text. Do not answer it, assess it, reason about it, add a disclaimer, or "
                "omit or soften its content. Preserve exact meaning, intent, tone, ambiguity, "
                "formatting, punctuation, capitalization, line breaks, code, variable names, "
                "function names, dictionaries, URLs, and text in angle brackets. Keep proper names, "
                "technical terms, and culturally specific references recognisable; transliterate in "
                "Hindi where natural. Hindi must be Devanagari. Hinglish must be natural Roman-script "
                "Hindi code-mix and must contain no Devanagari. Return no analysis or <think> text."
            ),
        },
        {
            "role": "user",
            "content": (
                "Translate the source text below into both target languages. Return exactly one valid "
                "minified JSON object and nothing else, with this schema: "
                '{"hindi":"...","hinglish":"..."}. Values must be the translations only; do not "'
                "include labels, angle-bracket wrappers, commentary, or safety text.\n\nSOURCE TEXT:\n" + text
            ),
        },
    ]


def _query_hf(
    text: str,
    *,
    model: str,
    provider: str = "auto",
    timeout: float = 90,
    on_wait: Callable[[str], None] | None = None,
) -> str:
    """Call Hugging Face routed inference without persisting the user token."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set. Export a Hugging Face user access token first.")
    try:
        from huggingface_hub import InferenceClient
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install requirements.txt") from exc
    client = InferenceClient(model=model, provider=provider, api_key=token, timeout=timeout)
    stop_heartbeat = Event()

    def _heartbeat() -> None:
        elapsed = 0
        while not stop_heartbeat.wait(15):
            elapsed += 15
            if on_wait:
                on_wait(f"still waiting for provider response ({elapsed}s of {int(timeout)}s)")

    heartbeat = Thread(target=_heartbeat, daemon=True)
    if on_wait:
        heartbeat.start()
    try:
        completion = client.chat_completion(
            messages=_build_hf_messages(text),
            max_tokens=4096,
            temperature=0,
            response_format={"type": "json_object"},
        )
    finally:
        stop_heartbeat.set()
        if on_wait:
            heartbeat.join(timeout=1)
    content = completion.choices[0].message.content
    if not content:
        raise RuntimeError("Hugging Face returned an empty translation response")
    return str(content)


def hf_authenticated_username() -> str:
    """Return the HF account for ``HF_TOKEN`` without exposing token contents."""
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is not set.")
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise RuntimeError("huggingface_hub is required; install requirements.txt") from exc
    identity = HfApi(token=token).whoami()
    username = identity.get("name") or identity.get("fullname")
    if not username:
        raise RuntimeError("Hugging Face did not return an account name for HF_TOKEN.")
    return str(username)


def _clean_translation_text(text: str) -> str:
    """Strip markdown artifacts from Ollama translation output."""
    cleaned = text.strip()
    cleaned = re.sub(r"^[*_\s`]+|[*_\s`]+$", "", cleaned)
    cleaned = re.sub(r"\*\*", "", cleaned)
    cleaned = re.sub(r"^---+.*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    # Drop trailing parenthetical roman transliteration lines in Hindi field
    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    kept: list[str] = []
    for ln in lines:
        if re.match(r"^\(.+\)$", ln):
            continue
        if ln.startswith("(") and ln.endswith(")"):
            continue
        kept.append(ln)
    if kept:
        cleaned = kept[0] if len(kept) == 1 else " ".join(kept)
    return cleaned.strip()


def _parse_ollama_response(response: str) -> tuple[str, str]:
    """Extract clean Hindi/Hinglish fields from JSON or legacy labelled output."""
    response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL | re.IGNORECASE).strip()
    json_match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if json_match:
        try:
            payload = json.loads(json_match.group(0))
            hindi = str(payload.get("hindi", "")).strip()
            hinglish = str(payload.get("hinglish", "")).strip()
            if hindi and hinglish:
                return hindi, hinglish
        except (json.JSONDecodeError, AttributeError):
            pass
    hindi = ""
    hinglish = ""
    hindi_matches = list(HINDI_RE.finditer(response))
    hinglish_matches = list(HINGLISH_RE.finditer(response))
    hindi_match = hindi_matches[-1] if hindi_matches else None
    hinglish_match = hinglish_matches[-1] if hinglish_matches else None
    if hindi_match:
        hindi = _clean_translation_text(hindi_match.group(1))
    if hinglish_match:
        hinglish = _clean_translation_text(hinglish_match.group(1))
    return hindi, hinglish


def export_translation_requests(
    selected_path: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    """Write translation_requests.csv from selected_rows.csv."""
    selected_path = selected_path or DEFAULT_SELECTED_PATH
    out_path = out_path or DEFAULT_REQUESTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    selected = pd.read_csv(selected_path, dtype=str, keep_default_na=False)
    rows = []
    for _, row in selected.iterrows():
        english = str(row["source_text"])
        rows.append(
            {
                "sample_id": row["sample_id"],
                "english": english,
                "translation_prompt": generate_translation_prompt(english),
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d translation requests to %s", len(df), out_path)
    return out_path


def init_reviewed_template(
    selected_path: Path | None = None,
    out_path: Path | None = None,
    *,
    overwrite: bool = False,
) -> Path:
    """Initialize reviewed_translations.csv from selected_rows."""
    selected_path = selected_path or DEFAULT_SELECTED_PATH
    out_path = out_path or DEFAULT_REVIEWED_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists() and not overwrite:
        logger.info("Review template already exists at %s; skipping init", out_path)
        return out_path

    selected = pd.read_csv(selected_path, dtype=str, keep_default_na=False)
    n = len(selected)
    trans_cols = translation_column_names()
    template_data: dict = {
        "sample_id": selected["sample_id"].astype(str),
        "english": selected["source_text"].astype(str),
        "translation_source": [""] * n,
        "reviewed": ["False"] * n,
        "meaning_score": [""] * n,
        "naturalness_score": [""] * n,
        "surface_sensitivity_score": [""] * n,
        "notes": [""] * n,
    }
    for col in trans_cols:
        template_data[col] = [""] * n
    template = pd.DataFrame(template_data)
    template.to_csv(out_path, index=False)
    logger.info("Saved review template (%d rows) to %s", len(template), out_path)
    return out_path


def _rows_for_smoke(
    df: pd.DataFrame,
    *,
    n_rows: int = 5,
    sample_ids: list[str] | None = None,
) -> list[int]:
    if sample_ids is not None:
        id_set = {str(sid) for sid in sample_ids}
        return [idx for idx, sid in enumerate(df["sample_id"]) if str(sid) in id_set]
    return list(range(min(n_rows, len(df))))


def assist_translate_with_ollama(
    reviewed_path: Path | None = None,
    *,
    model: str = DEFAULT_OLLAMA_MODEL,
    limit: int | None = None,
    sample_ids: list[str] | None = None,
    overwrite_existing: bool = False,
) -> int:
    """Fill Hindi/Hinglish via Ollama, optionally replacing existing drafts."""
    reviewed_path = reviewed_path or DEFAULT_REVIEWED_PATH
    df = pd.read_csv(reviewed_path, dtype=str, keep_default_na=False)
    updated = 0
    smoke_id_set = {str(sid) for sid in sample_ids} if sample_ids is not None else None

    for idx, row in df.iterrows():
        if smoke_id_set is not None and str(row["sample_id"]) not in smoke_id_set:
            continue
        if limit is not None and updated >= limit:
            break
        english = str(row["english"]).strip()
        if not english:
            continue
        if (
            not overwrite_existing
            and str(row["hindi"]).strip()
            and str(row["hinglish"]).strip()
        ):
            continue

        logger.info("Ollama translate [%s]: %s", row["sample_id"], english[:60])
        try:
            raw = _query_ollama(_build_ollama_prompt(english), model=model)
            hindi, hinglish = _parse_ollama_response(raw)
            if not hindi or not hinglish:
                raise ValueError(f"Failed to parse Ollama output: {raw[:200]}")
            df.at[idx, "hindi"] = hindi
            df.at[idx, "hinglish"] = hinglish
            df.at[idx, "translation_source"] = f"candidate:{model}"
            df.at[idx, "reviewed"] = "False"
            df.at[idx, "meaning_score"] = ""
            df.at[idx, "naturalness_score"] = ""
            df.at[idx, "surface_sensitivity_score"] = ""
            df.at[idx, "notes"] = "Generated by Ollama — draft only; human review required"
            updated += 1
            df.to_csv(reviewed_path, index=False)
            time.sleep(0.5)
        except Exception as exc:
            logger.error("Ollama failed for %s: %s", row["sample_id"], exc)
            df.at[idx, "notes"] = f"ERROR: {exc}"
            df.to_csv(reviewed_path, index=False)

    logger.info("Ollama assist updated %d rows in %s", updated, reviewed_path)
    return updated


def assist_translate_with_hf(
    reviewed_path: Path,
    *,
    model: str,
    provider: str = "auto",
    sample_ids: list[str] | None = None,
    overwrite_existing: bool = False,
    progress: Callable[[str], None] | None = None,
    request_timeout: float = 90,
    retries: int = 1,
) -> int:
    """Draft Hindi/Hinglish via Hugging Face routed inference; never approve rows."""
    df = pd.read_csv(reviewed_path, dtype=str, keep_default_na=False)
    target_ids = {str(sid) for sid in sample_ids} if sample_ids is not None else None
    target_rows = [
        row for _, row in df.iterrows()
        if target_ids is None or str(row["sample_id"]) in target_ids
    ]
    total = len(target_rows)
    updated = 0
    processed = 0
    started = time.monotonic()
    for idx, row in df.iterrows():
        if target_ids is not None and str(row["sample_id"]) not in target_ids:
            continue
        if not overwrite_existing and str(row["hindi"]).strip() and str(row["hinglish"]).strip():
            continue
        english = str(row["english"]).strip()
        if not english:
            continue
        processed += 1
        attempt = processed
        row_started = time.monotonic()
        if progress:
            progress(f"[{attempt}/{total}] Sending {row['sample_id']} to Hugging Face …")
        logger.info("HF translate [%s] model=%s", row["sample_id"], model)
        try:
            raw = ""
            for request_attempt in range(1, retries + 1):
                try:
                    raw = _query_hf(
                        english,
                        model=model,
                        provider=provider,
                        timeout=request_timeout,
                        on_wait=(
                            lambda message, a=attempt, t=total: progress(f"[{a}/{t}] {message}")
                            if progress else None
                        ),
                    )
                    break
                except Exception as exc:
                    if request_attempt == retries:
                        raise
                    if progress:
                        progress(
                            f"[{attempt}/{total}] request failed ({type(exc).__name__}); "
                            f"retrying once in 3s"
                        )
                    time.sleep(3)
            hindi, hinglish = _parse_ollama_response(raw)
            if not hindi or not hinglish:
                raise ValueError(f"Failed to parse HF output: {raw[:200]}")
            df.at[idx, "hindi"] = hindi
            df.at[idx, "hinglish"] = hinglish
            df.at[idx, "translation_source"] = f"candidate:hf:{model}"
            df.at[idx, "reviewed"] = "False"
            df.at[idx, "meaning_score"] = ""
            df.at[idx, "naturalness_score"] = ""
            df.at[idx, "surface_sensitivity_score"] = ""
            df.at[idx, "notes"] = "Generated through Hugging Face — draft only; human review required"
            updated += 1
            if progress:
                elapsed = time.monotonic() - row_started
                overall = time.monotonic() - started
                progress(
                    f"[{updated}/{total}] ✓ {row['sample_id']} saved "
                    f"({elapsed:.1f}s; elapsed {overall / 60:.1f}m)"
                )
        except Exception as exc:
            logger.error("Hugging Face failed for %s: %s", row["sample_id"], exc)
            df.at[idx, "notes"] = f"ERROR: {exc}"
            if progress:
                progress(f"[{attempt}/{total}] ✗ {row['sample_id']} failed; details saved in notes")
        df.to_csv(reviewed_path, index=False)
    if progress:
        progress(f"Finished: {updated}/{total} rows drafted in {(time.monotonic() - started) / 60:.1f}m.")
    logger.info("HF assist updated %d rows in %s", updated, reviewed_path)
    return updated


def seed_smoke_placeholders(
    reviewed_path: Path | None = None,
    *,
    n_rows: int = 5,
    sample_ids: list[str] | None = None,
) -> int:
    """Seed smoke rows with placeholder translations marked reviewed=true."""
    reviewed_path = reviewed_path or DEFAULT_REVIEWED_PATH
    df = pd.read_csv(reviewed_path, dtype=str, keep_default_na=False)
    seeded = 0

    def _is_messy(val: str) -> bool:
        return bool(val) and ("**" in val or len(val) > 200 or "\n---" in val)

    for idx in _rows_for_smoke(df, n_rows=n_rows, sample_ids=sample_ids):
        row = df.iloc[idx]
        english = str(row["english"]).strip()
        if not english:
            continue
        hindi = str(row["hindi"]).strip()
        hinglish = str(row["hinglish"]).strip()
        if _is_messy(hindi):
            hindi = english
        elif hindi:
            hindi = _clean_translation_text(hindi)
        else:
            hindi = english
        if _is_messy(hinglish):
            hinglish = english
        elif hinglish:
            hinglish = _clean_translation_text(hinglish)
        else:
            hinglish = english
        df.at[idx, "hindi"] = hindi
        df.at[idx, "hinglish"] = hinglish
        if str(row.get("translation_source", "")).strip() in ("", "smoke-placeholder"):
            df.at[idx, "translation_source"] = "smoke-placeholder"
        df.at[idx, "reviewed"] = "True"
        df.at[idx, "notes"] = "smoke-only scaffold validation — not for claims"
        seeded += 1

    df.to_csv(reviewed_path, index=False)
    logger.info("Seeded %d smoke rows (reviewed=true) in %s", seeded, reviewed_path)
    return seeded


def seed_smoke_reviewed(
    reviewed_path: Path | None = None,
    *,
    n_rows: int = 5,
    sample_ids: list[str] | None = None,
) -> int:
    """Mark smoke rows with translations as reviewed=true."""
    reviewed_path = reviewed_path or DEFAULT_REVIEWED_PATH
    df = pd.read_csv(reviewed_path, dtype=str, keep_default_na=False)
    marked = 0

    for idx in _rows_for_smoke(df, n_rows=n_rows, sample_ids=sample_ids):
        if str(df.at[idx, "hindi"]).strip() and str(df.at[idx, "hinglish"]).strip():
            df.at[idx, "reviewed"] = "True"
            existing_notes = str(df.at[idx, "notes"]).strip()
            smoke_note = "smoke-only auto-reviewed — full run requires human review"
            if smoke_note not in existing_notes:
                df.at[idx, "notes"] = (
                    f"{existing_notes}; {smoke_note}" if existing_notes else smoke_note
                )
            marked += 1

    df.to_csv(reviewed_path, index=False)
    logger.info("Marked %d rows reviewed=true (smoke) in %s", marked, reviewed_path)
    return marked


def prepare_track_b_translations(
    *,
    selected_path: Path | None = None,
    requests_path: Path | None = None,
    reviewed_path: Path | None = None,
    assist_translate: bool = False,
    seed_smoke: bool = False,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    overwrite_review: bool = False,
) -> None:
    """Full translation prep: requests + review template + optional Ollama + smoke seed."""
    selected_path = selected_path or DEFAULT_SELECTED_PATH
    if not selected_path.exists():
        raise FileNotFoundError(f"Missing {selected_path}; run row selection first.")

    selected = pd.read_csv(selected_path, dtype=str, keep_default_na=False)
    smoke_sample_ids: list[str] | None = None
    if seed_smoke:
        from src.track_b_select import select_smoke_sample_ids

        smoke_sample_ids = select_smoke_sample_ids(selected)
        translate_ids = set(
            selected.loc[selected["intended_task"] == "translate", "sample_id"].astype(str)
        )
        sentiment_ids = set(
            selected.loc[selected["intended_task"] == "sentiment_analysis", "sample_id"].astype(str)
        )
        logger.info(
            "Smoke catalog: %d sample_ids (%d translate, %d sentiment)",
            len(smoke_sample_ids),
            sum(sid in translate_ids for sid in smoke_sample_ids),
            sum(sid in sentiment_ids for sid in smoke_sample_ids),
        )

    export_translation_requests(selected_path, requests_path)
    init_reviewed_template(
        selected_path,
        reviewed_path,
        overwrite=overwrite_review,
    )

    if assist_translate:
        if _ollama_available():
            assist_translate_with_ollama(
                reviewed_path,
                model=ollama_model,
                limit=len(smoke_sample_ids) if seed_smoke else None,
                sample_ids=smoke_sample_ids,
            )
        else:
            logger.warning("Ollama unavailable; using placeholder smoke translations")
            if seed_smoke:
                seed_smoke_placeholders(reviewed_path, sample_ids=smoke_sample_ids)

    if seed_smoke:
        if assist_translate and _ollama_available():
            seed_smoke_reviewed(reviewed_path, sample_ids=smoke_sample_ids)
        seed_smoke_placeholders(reviewed_path, sample_ids=smoke_sample_ids)
