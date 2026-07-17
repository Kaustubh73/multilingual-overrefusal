"""Run instruction-tuned LM inference (default: Llama-3.1-8B-Instruct, 4-bit on CUDA)."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "data" / "results" / "responses.csv"

DEFAULT_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# Open, no gating — use while waiting for Llama access (see README).
PILOT_OPEN_MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

MAX_NEW_TOKENS = 256

_model = None
_tokenizer = None
_active_model_id: str | None = None


def get_model_id() -> str:
    """Resolve model id from env or default."""
    return os.environ.get("OVERREFUSAL_MODEL", DEFAULT_MODEL_ID)


def set_model_id(model_id: str) -> None:
    """Set the model to load; clears cached weights if the id changed."""
    global _active_model_id, _model, _tokenizer
    os.environ["OVERREFUSAL_MODEL"] = model_id
    if _active_model_id != model_id:
        _model = None
        _tokenizer = None
    _active_model_id = model_id


def _get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_model_weights(model_id: str, device: str):
    """Load model weights for the given device."""
    use_quant = device == "cuda"
    if use_quant:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
        )
        return AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )

    if device != "cuda":
        logger.warning(
            "CUDA not available; loading %s without 4-bit quantization (slower, more memory).",
            model_id,
        )
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)
    if device in ("mps", "cuda") and not use_quant:
        model = model.to(device)
    return model


def get_model_and_tokenizer(model_id: str | None = None):
    """Load model and tokenizer once per process (singleton)."""
    global _model, _tokenizer, _active_model_id

    model_id = model_id or get_model_id()
    if _model is not None and _tokenizer is not None and _active_model_id == model_id:
        return _model, _tokenizer

    device = _get_device()
    logger.info("Loading model %s on device=%s", model_id, device)

    try:
        _model = _load_model_weights(model_id, device)
        _tokenizer = AutoTokenizer.from_pretrained(model_id)
    except OSError as exc:
        err = str(exc).lower()
        if "gated" in err or "authorized list" in err:
            logger.error(
                "Cannot access gated model '%s'.\n"
                "  1. Request access: https://huggingface.co/%s\n"
                "  2. After approval: huggingface-cli login\n"
                "  3. Until then, run with an open model, e.g.:\n"
                "       python main.py run --dry-run --model %s",
                model_id,
                model_id,
                PILOT_OPEN_MODEL_ID,
            )
        raise

    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model.eval()
    _active_model_id = model_id
    return _model, _tokenizer


def generate_response(
    prompt: str,
    max_new_tokens: int = MAX_NEW_TOKENS,
    model_id: str | None = None,
) -> str:
    """Generate a single model response for a wrapped prompt."""
    model, tokenizer = get_model_and_tokenizer(model_id)

    messages = [{"role": "user", "content": prompt}]
    try:
        chat_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        chat_prompt = prompt

    inputs = tokenizer(chat_prompt, return_tensors="pt")
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    generated = output_ids[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated, skip_special_tokens=True).strip()
    return response


def run_inference(
    prompts_df: pd.DataFrame,
    out_path: Path | None = None,
    model_id: str | None = None,
) -> pd.DataFrame:
    """Run inference on all rows and save responses.csv."""
    if model_id:
        set_model_id(model_id)
    resolved_id = get_model_id()

    out_path = out_path or DEFAULT_RESULTS_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in prompts_df.iterrows():
        prompt = str(row["wrapped_prompt"])
        logger.info(
            "Generating [%d/%d] %s/%s (model=%s)",
            len(rows) + 1,
            len(prompts_df),
            row["prompt_id"],
            row["language"],
            resolved_id,
        )
        response = generate_response(prompt, model_id=resolved_id)
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                "language": row["language"],
                "wrapped_prompt": prompt,
                "response": response,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d responses to %s", len(df), out_path)
    return df
