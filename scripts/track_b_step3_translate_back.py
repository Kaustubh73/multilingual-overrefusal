#!/usr/bin/env python3
"""Step 3: Translate non-English responses back to English.

Uses sarvam-translate (Ollama) for Hindi/Urdu/Nepali responses.
Uses NLLB-200-distilled-600M (HF) for Spanish/Chinese responses.
English responses are kept as-is.

Usage:
  python scripts/track_b_step3_translate_back.py           # Full
  python scripts/track_b_step3_translate_back.py --limit 10  # Smoke test
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("track_b_translate_back")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
OLLAMA_URL = "http://localhost:11434/api/chat"
SARVAM_MODEL = "sarvam-translate:latest"

# NLLB target language codes
NLLB_LANG_MAP = {
    "spanish": "spa_Latn",
    "chinese": "zho_Hans",
    "urdu": "urd_Arab",
    "hindi": "hin_Deva",
    "nepali": "nep_Deva",
}

# Singleton NLLB model
_nllb_model = None
_nllb_tokenizer = None


def _load_nllb():
    global _nllb_model, _nllb_tokenizer
    if _nllb_model is not None:
        return _nllb_model, _nllb_tokenizer
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    name = "facebook/nllb-200-distilled-600M"
    log.info("Loading NLLB-200-distilled-600M...")
    _nllb_tokenizer = AutoTokenizer.from_pretrained(name)
    _nllb_model = AutoModelForSeq2SeqLM.from_pretrained(name)
    log.info("NLLB loaded (%.1f GB)", sum(p.numel() * 4 for p in _nllb_model.parameters()) / 1e9)
    return _nllb_model, _nllb_tokenizer


def sarvam_translate_to_english(text: str) -> str:
    payload = {
        "model": SARVAM_MODEL,
        "messages": [{"role": "user", "content": f"Translate to English: {text}"}],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def nllb_translate_to_english(text: str, source_lang: str) -> str:
    src_code = NLLB_LANG_MAP.get(source_lang)
    if not src_code:
        return text
    model, tokenizer = _load_nllb()
    tokenizer.src_lang = src_code
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=512,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(description="Translate-back non-English responses")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to translate (smoke test)")
    parser.add_argument(
        "--split", choices=["test", "train"], default="test",
        help="Dataset split (default: test)",
    )
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP 3: TRANSLATE-BACK NON-ENGLISH RESPONSES (split=%s)", args.split)
    log.info("=" * 60)

    raw_path = RESULTS_DIR / f"track_b_raw_{args.split}.csv"
    out_path = RESULTS_DIR / f"track_b_raw_translated_{args.split}.csv"

    if not raw_path.exists():
        log.error("Missing %s — run Step 2 first", raw_path)
        return 1

    df = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    log.info("Loaded %d rows from %s", len(df), raw_path)

    if args.limit:
        df = df.head(args.limit)
        log.info("Limited to first %d rows", args.limit)

    # Check Ollama
    log.info("Checking Ollama connection for sarvam-translate...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if SARVAM_MODEL not in models:
            log.error("Model %s not found. Available: %s", SARVAM_MODEL, models)
            return 1
        log.info("Ollama OK — %s available", SARVAM_MODEL)
    except Exception as e:
        log.error("Ollama not reachable: %s", e)
        return 1

    # Split
    non_en = df[df["language"] != "english"].copy()
    en_rows = df[df["language"] == "english"].copy()
    log.info("English rows (no translation needed): %d", len(en_rows))
    log.info("Non-English rows (translation needed): %d", len(non_en))
    log.info("")

    # Resume
    done_pids = set()
    translated_rows = []
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        done_pids = set(existing["prompt_id"].tolist())
        translated_rows = existing.to_dict("records")
        log.info("Resuming: %d rows already translated", len(done_pids))

    # Add English rows
    for _, erow in en_rows.iterrows():
        pid = erow["prompt_id"]
        if pid not in done_pids:
            row_dict = erow.to_dict()
            row_dict["response_english"] = erow["response"]
            row_dict["translate_method"] = "none"
            translated_rows.append(row_dict)
            done_pids.add(pid)
    pd.DataFrame(translated_rows).to_csv(out_path, index=False)

    # Translate non-English
    total = len(non_en)
    remaining = sum(1 for _, r in non_en.iterrows() if r["prompt_id"] not in done_pids)
    log.info("Translating %d rows (remaining: %d)...", total, remaining)
    log.info("-" * 60)

    sarvam_targets = {"hindi", "urdu", "nepali"}
    nllb_targets = {"spanish", "chinese"}
    errors = 0

    for i, (_, row) in enumerate(non_en.iterrows(), start=1):
        pid = row["prompt_id"]
        if pid in done_pids:
            continue

        response = str(row["response"])
        task = row["intended_task"]
        lang = row["language"]
        target_lang = str(row.get("translate_target_lang", "")).lower()

        if task == "translate":
            resp_lang = target_lang
        else:
            resp_lang = "hindi"

        t_start = time.time()
        try:
            if resp_lang in sarvam_targets or resp_lang == "unknown":
                translated = sarvam_translate_to_english(response)
                method = "sarvam"
            elif resp_lang in nllb_targets:
                translated = nllb_translate_to_english(response, resp_lang)
                method = "nllb"
            else:
                translated = response
                method = "passthrough"

            elapsed = time.time() - t_start
            preview = translated[:80].replace("\n", " ")
            log.info(
                "[%3d/%d] %s %-10s %-20s %s→en (%s) %.1fs | %s",
                i, total, pid, lang, task, resp_lang, method, elapsed, preview,
            )
        except Exception as e:
            elapsed = time.time() - t_start
            log.error("[%3d/%d] %s %-10s %-20s ERROR (%.1fs): %s", i, total, pid, lang, task, elapsed, e)
            translated = response
            method = f"error:{e}"
            errors += 1

        row_dict = row.to_dict()
        row_dict["response_english"] = translated
        row_dict["translate_method"] = method
        translated_rows.append(row_dict)
        done_pids.add(pid)

        if i % 10 == 0 or i == total:
            pd.DataFrame(translated_rows).to_csv(out_path, index=False)

    log.info("-" * 60)
    log.info("TRANSLATE-BACK COMPLETE")
    log.info("  Total rows:   %d", len(translated_rows))
    log.info("  Errors:       %d", errors)
    log.info("  Elapsed:      %.1f minutes", (time.time() - t0) / 60)
    log.info("  Saved to:     %s", out_path)
    log.info("=" * 60)
    return 1 if errors > max(total, 1) * 0.1 else 0


if __name__ == "__main__":
    sys.exit(main())
