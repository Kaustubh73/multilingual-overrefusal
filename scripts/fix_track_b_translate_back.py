#!/usr/bin/env python3
"""Fix: re-run translate-back with proper target language mapping.

The original Step 3 couldn't determine translate target languages because
track_b_raw.csv lacks the translate_target_lang column. This script:

1. Merges raw responses with catalog to get translate_target_lang
2. Re-runs translation for all non-English translate-task rows
3. Writes to a new file (does NOT overwrite original)
4. Runs Steps 4-5 on the fixed data
"""

import json
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
log = logging.getLogger("fix_translate_back")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = PROJECT_ROOT / "results" / "track_b_catalog.csv"
RAW_PATH = PROJECT_ROOT / "results" / "track_b_raw.csv"
ORIG_TRANSLATED_PATH = PROJECT_ROOT / "results" / "track_b_raw_translated.csv"
FIXED_TRANSLATED_PATH = PROJECT_ROOT / "results" / "track_b_raw_translated_fixed.csv"
FIXED_JUDGMENTS_PATH = PROJECT_ROOT / "results" / "track_b_judgments_fixed.csv"
OLLAMA_URL = "http://localhost:11434/api/chat"
SARVAM_MODEL = "sarvam-translate:latest"

# Target language codes for NLLB
NLLB_LANG_MAP = {
    "spanish": "spa_Latn",
    "chinese": "zho_Hans",
    "urdu": "urd_Arab",
    "hindi": "hin_Deva",
    "nepali": "nep_Deva",
}

# Map catalog target_lang → route
SARVAM_TARGETS = {"hindi", "urdu", "nepali"}
NLLB_TARGETS = {"spanish", "chinese"}

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
    src_code = NLLB_LANG_MAP.get(source_lang.lower())
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
    t0 = time.time()
    log.info("=" * 60)
    log.info("FIX: RE-RUN TRANSLATE-BACK WITH PROPER TARGET LANGUAGE MAPPING")
    log.info("=" * 60)

    # Load catalog for translate_target_lang
    catalog = pd.read_csv(CATALOG_PATH, dtype=str, keep_default_na=False)
    cat_meta = catalog[["prompt_id", "translate_target_lang"]].dropna()

    # Load raw responses
    raw = pd.read_csv(RAW_PATH, dtype=str, keep_default_na=False)
    raw = raw.merge(cat_meta, on="prompt_id", how="left")

    # Start with English rows (keep original translation)
    en_rows = raw[raw["language"] == "english"].copy()
    non_en = raw[raw["language"] != "english"].copy()

    log.info("English rows (keep original): %d", len(en_rows))
    log.info("Non-English rows: %d", len(non_en))

    # Check if we already did fixed ones
    done_pids = set()
    translated_rows = []
    if FIXED_TRANSLATED_PATH.exists():
        existing = pd.read_csv(FIXED_TRANSLATED_PATH, dtype=str, keep_default_na=False)
        done_pids = set(existing["prompt_id"].tolist())
        translated_rows = existing.to_dict("records")
        log.info("Resuming: %d rows already done", len(done_pids))

    # Add English rows
    for _, erow in en_rows.iterrows():
        pid = erow["prompt_id"]
        if pid not in done_pids:
            row_dict = erow.to_dict()
            row_dict["response_english"] = erow["response"]
            row_dict["translate_method"] = "none"
            translated_rows.append(row_dict)
            done_pids.add(pid)
    pd.DataFrame(translated_rows).to_csv(FIXED_TRANSLATED_PATH, index=False)

    # Check Ollama
    log.info("Checking Ollama...")
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

    total = len(non_en)
    remaining = sum(1 for _, r in non_en.iterrows() if r["prompt_id"] not in done_pids)
    log.info("Translating %d rows (remaining: %d)...", total, remaining)
    log.info("-" * 60)

    errors = 0
    for i, (_, row) in enumerate(non_en.iterrows(), start=1):
        pid = row["prompt_id"]
        if pid in done_pids:
            continue

        response = str(row["response"])
        task = row["intended_task"]
        lang = row["language"]
        target_lang = str(row.get("translate_target_lang", "")).strip().lower() if pd.notna(row.get("translate_target_lang")) else ""

        # Determine response language
        if task == "translate" and target_lang:
            resp_lang = target_lang.lower()
        else:
            resp_lang = "hindi"

        # Determine translate method
        resp_lang_clean = resp_lang.split("(")[0].strip()

        t_start = time.time()
        try:
            if resp_lang_clean in SARVAM_TARGETS or resp_lang_clean == "unknown":
                translated = sarvam_translate_to_english(response)
                method = "sarvam"
            elif resp_lang_clean in NLLB_TARGETS:
                translated = nllb_translate_to_english(response, resp_lang_clean)
                method = "nllb"
            else:
                translated = response
                method = "passthrough"

            elapsed = time.time() - t_start
            preview = translated[:80].replace("\n", " ")
            log.info(
                "[%3d/%d] %s %-10s %-20s %s→en (%s) %.1fs | %s",
                i, total, pid, lang, task, resp_lang_clean, method, elapsed, preview,
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
            pd.DataFrame(translated_rows).to_csv(FIXED_TRANSLATED_PATH, index=False)

    log.info("-" * 60)
    log.info("TRANSLATE-BACK FIX COMPLETE")
    log.info("  Total rows:   %d", len(translated_rows))
    log.info("  Errors:       %d", errors)
    log.info("  Elapsed:      %.1f minutes", (time.time() - t0) / 60)
    log.info("  Saved to:     %s", FIXED_TRANSLATED_PATH)
    log.info("")
    log.info("Next steps:")
    log.info("  python scripts/track_b_step4_judge.py --input results/track_b_raw_translated_fixed.csv --output results/track_b_judgments_fixed.csv")
    log.info("  python scripts/track_b_step5_report.py --judgments results/track_b_judgments_fixed.csv --report results/track_b_report_fixed.md")
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
