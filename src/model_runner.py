"""Run validation inference via Ollama (preferred) or HuggingFace fallback."""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_OUTPUTS_PATH = PROJECT_ROOT / "results" / "raw_outputs.csv"

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
OLLAMA_TAGS_URL = "http://localhost:11434/api/tags"

MAX_RETRIES = 3
RETRY_BACKOFF_SEC = 2.0
REQUEST_TIMEOUT_SEC = 300

# Canonical validation model ids -> Ollama tag (when available locally).
OLLAMA_MODEL_MAP: dict[str, str] = {
    "meta-llama/Llama-3.1-8B-Instruct": "llama3.1:8b-instruct",
    "qwen2.5:7b-instruct": "qwen2.5:7b-instruct",
    "gemma3:12b": "gemma3:12b",
}

HF_FALLBACK_MODELS = {"meta-llama/Llama-3.1-8B-Instruct"}

_session: requests.Session | None = None
_active_backend: str | None = None
_active_model: str | None = None
_ollama_tags_cache: set[str] | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
    return _session


def _reset_backend() -> None:
    global _active_backend, _active_model
    _active_backend = None
    _active_model = None


def list_ollama_models() -> set[str]:
    """Return locally available Ollama model names."""
    global _ollama_tags_cache
    if _ollama_tags_cache is not None:
        return _ollama_tags_cache
    try:
        resp = _get_session().get(OLLAMA_TAGS_URL, timeout=10)
        resp.raise_for_status()
        models = {m["name"] for m in resp.json().get("models", [])}
        # Ollama tags may include :latest aliases; also match base name.
        expanded = set(models)
        for name in list(models):
            if ":" in name:
                expanded.add(name.split(":")[0])
        _ollama_tags_cache = expanded
        return expanded
    except requests.RequestException as exc:
        logger.warning("Could not list Ollama models: %s", exc)
        _ollama_tags_cache = set()
        return _ollama_tags_cache


def _ollama_tag_available(tag: str) -> bool:
    available = list_ollama_models()
    if tag in available:
        return True
    base = tag.split(":")[0] if ":" in tag else tag
    return any(m == tag or m.startswith(f"{base}:") for m in available)


def resolve_backend(model_name: str) -> tuple[str, str]:
    """
    Resolve (backend, resolved_id) for a canonical validation model name.

    backend is 'ollama' or 'huggingface'.
  """
    ollama_tag = OLLAMA_MODEL_MAP.get(model_name, model_name)
    if _ollama_tag_available(ollama_tag):
        return "ollama", ollama_tag
    if model_name in HF_FALLBACK_MODELS:
        logger.warning(
            "Ollama model %s not found; falling back to HuggingFace for %s",
            ollama_tag,
            model_name,
        )
        return "huggingface", model_name
    raise RuntimeError(
        f"Model {model_name} (Ollama tag: {ollama_tag}) is not available. "
        f"Run: ollama pull {ollama_tag}"
    )


def load_model(model_name: str) -> None:
    """Select backend and warm the model (Ollama: noop; HF: load weights once)."""
    global _active_backend, _active_model

    backend, resolved_id = resolve_backend(model_name)
    if _active_backend == backend and _active_model == resolved_id:
        return

    if backend == "huggingface":
        from src.inference import set_model_id

        set_model_id(resolved_id)
        from src.inference import get_model_and_tokenizer

        get_model_and_tokenizer(resolved_id)

    _active_backend = backend
    _active_model = resolved_id
    logger.info("Loaded model backend=%s id=%s", backend, resolved_id)


def _ollama_chat(prompt: str, model_tag: str) -> str:
    payload = {
        "model": model_tag,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
    }
    resp = _get_session().post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=REQUEST_TIMEOUT_SEC,
    )
    resp.raise_for_status()
    data = resp.json()
    message = data.get("message") or {}
    content = message.get("content", "")
    return str(content).strip()


def _hf_generate(prompt: str, model_id: str) -> str:
    from src.inference import generate_response

    return generate_response(prompt, model_id=model_id)


def run_model(model_name: str, prompt: str) -> str:
    """
    Generate one response for model_name and prompt.

    Loads the model/backend once per process when the model changes.
    Retries transient failures up to MAX_RETRIES times.
    """
    load_model(model_name)
    assert _active_backend is not None and _active_model is not None

    last_error: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            if _active_backend == "ollama":
                return _ollama_chat(prompt, _active_model)
            return _hf_generate(prompt, _active_model)
        except Exception as exc:
            last_error = exc
            logger.error(
                "Model error (attempt %d/%d) model=%s backend=%s: %s",
                attempt,
                MAX_RETRIES,
                model_name,
                _active_backend,
                exc,
            )
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_BACKOFF_SEC * attempt)

    raise RuntimeError(
        f"Failed after {MAX_RETRIES} attempts for model={model_name}"
    ) from last_error


def _load_raw_outputs(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    return pd.DataFrame(
        columns=[
            "prompt_id",
            "task",
            "language",
            "model",
            "prompt",
            "response",
        ]
    )


def _append_raw_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df = _load_raw_outputs(path)
    df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    df.to_csv(path, index=False)


def _completed_keys(path: Path) -> set[tuple[str, str, str, str]]:
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    keys = set()
    for _, row in df.iterrows():
        task = str(row.get("task", ""))
        keys.add(
            (str(row["prompt_id"]), task, str(row["language"]), str(row["model"]))
        )
    return keys


def run_validation_batch(
    jobs_df: pd.DataFrame,
    out_path: Path | None = None,
    *,
    resume: bool = True,
) -> pd.DataFrame:
    """
    Run all (prompt_id, task, language, model) jobs and append to raw_outputs.csv.

    jobs_df columns: prompt_id, task, language, model, prompt
    """
    out_path = out_path or DEFAULT_RAW_OUTPUTS_PATH
    done = _completed_keys(out_path) if resume else set()

    for i, row in jobs_df.iterrows():
        task = str(row.get("task", ""))
        key = (
            str(row["prompt_id"]),
            task,
            str(row["language"]),
            str(row["model"]),
        )
        if key in done:
            logger.info("Skipping completed %s", key)
            continue

        model_name = str(row["model"])
        prompt = str(row["prompt"])
        logger.info(
            "Running [%s] %s / %s / %s / %s",
            i + 1 if isinstance(i, int) else "?",
            row["prompt_id"],
            task,
            row["language"],
            model_name,
        )

        try:
            response = run_model(model_name, prompt)
        except Exception as exc:
            logger.error("Giving up on %s: %s", key, exc)
            response = f"[MODEL_ERROR] {exc}"

        record = {
            "prompt_id": row["prompt_id"],
            "task": task,
            "language": row["language"],
            "model": model_name,
            "prompt": prompt,
            "response": response,
        }
        _append_raw_row(out_path, record)
        done.add(key)

    return _load_raw_outputs(out_path)
