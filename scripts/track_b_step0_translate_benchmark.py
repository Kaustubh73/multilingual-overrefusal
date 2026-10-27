#!/usr/bin/env python3
"""Step 0: Pre-translate all 270 benchmark source texts to Hindi & Hinglish.

Hindi: sarvam-translate (fast, 4GB)
Hinglish: gemma3:12b with few-shot prompt (natural code-mixed output)

Resume-safe: skips rows that already have translations.
Verbose logging per row.

Usage:
  python scripts/track_b_step0_translate_benchmark.py
  python scripts/track_b_step0_translate_benchmark.py --limit 10   # smoke test
"""

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("track_b_translate_bench")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_PATH = PROJECT_ROOT / "data" / "track_b" / "track_b_benchmark_translated.csv"
OLLAMA_URL = "http://localhost:11434/api/chat"
SARVAM_MODEL = "sarvam-translate:latest"
GEMMA_MODEL = "gemma3:12b"

HINGLISH_PROMPT = """Convert English to Hinglish (natural Hindi-English code-mix). Output ONLY the Hinglish text, nothing else.

Examples:
English: I am going to the market to buy vegetables
Hinglish: Main market ja raha hoon sabziyan khareedne ke liye

English: She told me that the meeting has been postponed
Hinglish: Usne mujhe bola ki meeting postponed ho gayi hai

English: Analyze the sentiment of the following text
Hinglish: Neeche diye gaye text ka sentiment analyze karo

Now convert to Hinglish (output ONLY the Hinglish text):
"""


def extract_source_text(plain_text: str, task: str) -> str | tuple:
    """Extract the translatable source text from an English prompt."""
    if task == "sentiment_analysis":
        m = re.search(
            r"(?:neutral|positive|negative)[\"']:\s*['\"](.+?)['\"]\s*$",
            plain_text, re.DOTALL,
        )
        return m.group(1).strip() if m else plain_text

    if task == "rephrase":
        m = re.search(r"(?:if any)[\"']:\s*['\"](.+?)['\"]\s*$", plain_text, re.DOTALL)
        if m:
            return m.group(1).strip()
        m = re.search(r"['\"](.+?)['\"]\s*$", plain_text, re.DOTALL)
        return m.group(1).strip() if m else plain_text

    if task == "translate":
        m = re.search(r"into\s+\w+:\s*['\"](.+?)['\"]\s*$", plain_text, re.DOTALL)
        return m.group(1).strip() if m else plain_text

    if task == "cryptanalysis":
        m = re.search(r"Encrypted text:\s*['\"](.+?)['\"]", plain_text, re.DOTALL)
        return m.group(1).strip() if m else plain_text

    if task == "rag_qa":
        m = re.search(r"Context:\s*(.*?)\s*Question:\s*(.*)", plain_text, re.DOTALL)
        if m:
            return m.group(1).strip(), m.group(2).strip()
        return plain_text, ""

    return plain_text


def ollama_chat(model: str, prompt: str, timeout: float = 300, temperature: float = 0) -> str:
    """Single Ollama chat call."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": temperature},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json().get("message", {}).get("content", "").strip()


def translate_hindi(text: str, temperature: float = 0) -> str:
    """Translate English to Hindi using sarvam-translate."""
    return ollama_chat(SARVAM_MODEL, f"Translate to Hindi: {text}", temperature=temperature)


def translate_hinglish(text: str, temperature: float = 0) -> str:
    """Translate English to Hinglish using gemma3:12b."""
    return ollama_chat(GEMMA_MODEL, HINGLISH_PROMPT + text, temperature=temperature)


def devanagari_to_hinglish(text: str) -> str:
    """Transliterate Devanagari to Latin-script Hinglish."""
    from aksharamukha import transliterate
    import unicodedata
    iast = transliterate.process("devanagari", "IAST", text)
    nfd = unicodedata.normalize("NFD", iast)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    stripped = stripped.replace("ṃ", "n").replace("ḥ", "h").replace("ś", "sh").replace("ṣ", "sh")
    return stripped


# Patterns that indicate a safety refusal
REFUSAL_PATTERNS = [
    "🔒Normal Output",
    "🔒Normal",
    "As an AI assistant",
    "As an AI language model",
    "I cannot",
    "I can't",
    "I'm sorry, but I",
    "I am sorry, but I",
    "I apologize, but",
    "I'm not able to",
    "I'm unable to",
    "I am unable to",
    "I don't think I can",
    "It's not appropriate",
    "It is not appropriate",
    "I will not",
    "I won't",
    "This goes against",
    "I'm afraid I can't",
    "I'm afraid I cannot",
    "Sorry, I can't",
    "Sorry, I cannot",
    "I'm not comfortable",
    "I am not comfortable",
    "I don't feel comfortable",
    "It wouldn't be right",
    "It would not be right",
    "I must decline",
    "I have to decline",
    "I'm not allowed to",
    "I am not allowed to",
    "violates my",
    "goes against my",
    "I cannot assist with",
    "I can't assist with",
    "I cannot help with",
    "I can't help with",
    "I'm programmed to",
    "I am programmed to",
    "I'm designed to",
    "I am designed to",
    "I need to be careful",
    "Let me be clear",
    "I want to be transparent",
    "I must be honest",
]


def validate_translation(source: str, translated: str, lang: str, task: str) -> str | None:
    """Validate a translation. Returns error message or None if OK."""
    if not translated or not translated.strip():
        return "empty output"

    t = translated.strip()

    # Check for numeric garbage (repeated digits like 5000000...)
    if re.match(r"^[\d\s,\.]+$", t):
        return f"numeric garbage: {t[:60]}"

    # Check for extreme length mismatch (>4x or <0.15x)
    src_len = len(source)
    t_len = len(t)
    if src_len > 5 and t_len > src_len * 4:
        return f"too long ({t_len} chars vs {src_len} source)"
    if src_len > 10 and t_len < src_len * 0.15:
        return f"too short ({t_len} chars vs {src_len} source)"

    # Hindi: must be Devanagari-dominant
    if lang == "hindi":
        has_devanagari = any("\u0900" <= c <= "\u097F" for c in t)
        if not has_devanagari:
            return f"no Devanagari script: {t[:60]}"
        # Check it's not mostly English with a few Hindi words
        alpha_chars = [c for c in t if c.isalpha()]
        if alpha_chars:
            eng_count = sum(1 for c in alpha_chars if ord(c) < 128)
            if eng_count > len(alpha_chars) * 0.5:
                return f"mostly English ({eng_count}/{len(alpha_chars)} alpha chars): {t[:60]}"

    # Hinglish: must NOT be primarily Devanagari AND must not be a refusal
    if lang == "hinglish":
        deva_count = sum(1 for c in t if "\u0900" <= c <= "\u097F")
        if deva_count > len(t) * 0.3:
            return f"too much Devanagari ({deva_count}/{len(t)} chars): {t[:60]}"
        # Check for safety refusal patterns
        t_lower = t.lower()
        for pattern in REFUSAL_PATTERNS:
            if pattern.lower() in t_lower:
                return f"refusal detected ({pattern}): {t[:60]}"

    return None


def translate_with_retry(
    text: str, target: str, max_retries: int = 3,
) -> tuple[str, str | None]:
    """Translate with validation, retry with higher temp, fallback strategies."""
    for attempt in range(max_retries):
        temp = 0.3 * (attempt + 1)  # 0.3, 0.6, 0.9
        if target == "hindi":
            result = translate_hindi(text, temperature=temp)
        else:
            result = translate_hinglish(text, temperature=temp)

        error = validate_translation(text, result, target, "general")
        if error is None:
            return result, None

        if attempt < max_retries - 1:
            log.warning("    attempt %d failed (%s), retrying with temp=%.1f...", attempt + 1, error, temp)

    # Fallback strategies
    if target == "hindi":
        # Hindi: try gemma as final fallback
        log.warning("    sarvam failed, falling back to gemma for Hindi")
        try:
            result = ollama_chat(GEMMA_MODEL, f"Translate this English text to Hindi. Output ONLY the Hindi translation, nothing else:\n\n{text}")
            error = validate_translation(text, result, target, "general")
            if error is None:
                return result, None
        except Exception:
            pass

    if target == "hinglish":
        # Hinglish: try llama3.1:8b as second fallback
        log.warning("    gemma failed for Hinglish, trying llama3.1:8b")
        try:
            result = ollama_chat("llama3.1:8b-instruct-q8_0", HINGLISH_PROMPT + text)
            error = validate_translation(text, result, target, "general")
            if error is None:
                return result, None
        except Exception:
            pass
        # Final fallback: transliterate the Hindi translation
        log.warning("    llama3.1 also failed, falling back to transliteration")
        try:
            hi = translate_hindi(text, temperature=0.6)
            hi_err = validate_translation(text, hi, "hindi", "general")
            if hi_err is None:
                result = devanagari_to_hinglish(hi)
                return result, None
        except Exception:
            pass

    return result, error


def main():
    parser = argparse.ArgumentParser(description="Pre-translate benchmark source texts")
    parser.add_argument("--limit", type=int, default=None, help="Max rows to translate (smoke test)")
    args = parser.parse_args()

    t0 = time.time()
    log.info("=" * 60)
    log.info("STEP 0: PRE-TRANSLATE BENCHMARK SOURCE TEXTS")
    log.info("=" * 60)
    log.info("  Hindi:    %s (fast translation)", SARVAM_MODEL)
    log.info("  Hinglish: %s (few-shot code-mix)", GEMMA_MODEL)
    log.info("")

    # Load benchmark from HuggingFace
    from src.paper_benchmark import load_paper_benchmark
    log.info("Loading paper benchmark (test split) from HuggingFace...")
    bench = load_paper_benchmark(split="test")
    log.info("Loaded %d rows", len(bench))

    if args.limit:
        bench = bench.head(args.limit)
        log.info("Limited to %d rows", args.limit)

    # Check Ollama
    log.info("Checking Ollama connection...")
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        for needed in [SARVAM_MODEL, GEMMA_MODEL]:
            if needed not in models:
                log.error("Model %s not found. Available: %s", needed, models)
                return 1
        log.info("Ollama OK — %s and %s available", SARVAM_MODEL, GEMMA_MODEL)
    except Exception as e:
        log.error("Ollama not reachable: %s", e)
        return 1

    # Resume: load existing translations and validate them
    done_keys: set[str] = set()
    rows: list[dict] = []
    retranslate_count = 0
    if OUT_PATH.exists():
        existing = pd.read_csv(OUT_PATH, dtype=str, keep_default_na=False)
        for _, erow in existing.iterrows():
            key = f"{erow['sample_id']}|{erow['intended_task']}"
            task = str(erow["intended_task"])

            # Validate existing translations
            needs_retranslate = False
            if task == "rag_qa":
                for col, lang in [("context_hi", "hindi"), ("question_hi", "hindi"),
                                  ("context_hl", "hinglish"), ("question_hl", "hinglish")]:
                    val = str(erow.get(col, ""))
                    src = str(erow.get(col.replace("_hi", "_en").replace("_hl", "_en"), ""))
                    err = validate_translation(src, val, lang, task)
                    if err:
                        log.warning("  Re-translating %s %s: %s", key, col, err)
                        needs_retranslate = True
            else:
                for col, lang in [("source_text_hi", "hindi"), ("source_text_hl", "hinglish")]:
                    val = str(erow.get(col, ""))
                    err = validate_translation(str(erow.get("source_text_en", "")), val, lang, task)
                    if err:
                        log.warning("  Re-translating %s %s: %s", key, col, err)
                        needs_retranslate = True

            if needs_retranslate:
                retranslate_count += 1
                # Don't add to done_keys — will be re-translated
            else:
                done_keys.add(key)
                rows.append(erow.to_dict())

        log.info("Resuming: %d rows OK, %d need re-translation", len(done_keys), retranslate_count)

    # Process each benchmark row
    total = len(bench)
    errors = 0

    log.info("")
    log.info("Translating source texts for %d benchmark rows...", total)
    log.info("-" * 60)

    for i, (_, row) in enumerate(bench.iterrows(), start=1):
        sid = str(row["sample_id"])
        task = str(row["intended_task"])
        plain_text = str(row["plain_text"])
        text_type = str(row.get("text_type", ""))
        task_label = str(row.get("task_label", ""))
        is_benign = str(row.get("is_benign_task", ""))

        key = f"{sid}|{task}"
        if key in done_keys:
            continue

        # Extract source text(s)
        extracted = extract_source_text(plain_text, task)

        # Cryptanalysis: encrypted text is NOT translatable — keep as-is
        if task == "cryptanalysis":
            source_text = extracted if isinstance(extracted, str) else str(extracted)
            rows.append({
                "sample_id": sid,
                "intended_task": task,
                "text_type": text_type,
                "task_label": task_label,
                "is_benign_task": is_benign,
                "source_text_type": "single",
                "source_text_en": source_text,
                "source_text_hi": source_text,  # ciphertext stays as-is
                "source_text_hl": source_text,  # ciphertext stays as-is
            })
            done_keys.add(key)
            log.info("[%3d/%d] %s %-20s ciphertext (no translation needed)", i, total, sid, task)
            continue

        # Handle RAG_QA (returns tuple)
        if task == "rag_qa" and isinstance(extracted, tuple):
            context_en, question_en = extracted
            context_hi, context_hl = "", ""
            question_hi, question_hl = "", ""

            t_start = time.time()
            try:
                # Hindi via sarvam
                context_hi, err1 = translate_with_retry(context_en, "hindi")
                question_hi, err2 = translate_with_retry(question_en, "hindi")
                # Hinglish via gemma
                context_hl, err3 = translate_with_retry(context_en, "hinglish")
                question_hl, err4 = translate_with_retry(question_en, "hinglish")
                errs = [e for e in [err1, err2, err3, err4] if e]
                elapsed = time.time() - t_start
                if errs:
                    log.warning(
                        "[%3d/%d] %s %-20s 4 translations with %d warnings (%.1fs)",
                        i, total, sid, task, len(errs), elapsed,
                    )
                else:
                    log.info(
                        "[%3d/%d] %s %-20s 4 translations OK (%.1fs) | ctx_hi: %s... | q_hi: %s...",
                        i, total, sid, task, elapsed,
                        context_hi[:40].replace("\n", " "),
                        question_hi[:40].replace("\n", " "),
                    )
            except Exception as e:
                elapsed = time.time() - t_start
                log.error("[%3d/%d] %s %-20s ERROR (%.1fs): %s", i, total, sid, task, elapsed, e)
                errors += 1

            rows.append({
                "sample_id": sid,
                "intended_task": task,
                "text_type": text_type,
                "task_label": task_label,
                "is_benign_task": is_benign,
                "source_text_type": "rag_qa",
                "context_en": context_en,
                "question_en": question_en,
                "context_hi": context_hi,
                "question_hi": question_hi,
                "context_hl": context_hl,
                "question_hl": question_hl,
            })
            done_keys.add(key)

        else:
            # Single text tasks
            source_text = extracted if isinstance(extracted, str) else str(extracted)
            text_hi, text_hl = "", ""

            t_start = time.time()
            try:
                # Hindi via sarvam (fast)
                text_hi, err1 = translate_with_retry(source_text, "hindi")
                # Hinglish via gemma (quality)
                text_hl, err2 = translate_with_retry(source_text, "hinglish")
                elapsed = time.time() - t_start
                errs = [e for e in [err1, err2] if e]
                if errs:
                    log.warning(
                        "[%3d/%d] %s %-20s 2 translations with %d warnings (%.1fs) | hi: %s... | hl: %s...",
                        i, total, sid, task, len(errs), elapsed,
                        text_hi[:50].replace("\n", " "),
                        text_hl[:50].replace("\n", " "),
                    )
                else:
                    log.info(
                        "[%3d/%d] %s %-20s 2 translations OK (%.1fs) | hi: %s... | hl: %s...",
                        i, total, sid, task, elapsed,
                        text_hi[:50].replace("\n", " "),
                        text_hl[:50].replace("\n", " "),
                    )
            except Exception as e:
                elapsed = time.time() - t_start
                log.error("[%3d/%d] %s %-20s ERROR (%.1fs): %s", i, total, sid, task, elapsed, e)
                errors += 1

            rows.append({
                "sample_id": sid,
                "intended_task": task,
                "text_type": text_type,
                "task_label": task_label,
                "is_benign_task": is_benign,
                "source_text_type": "single",
                "source_text_en": source_text,
                "source_text_hi": text_hi,
                "source_text_hl": text_hl,
            })
            done_keys.add(key)

        # Save every 5 rows
        if i % 5 == 0 or i == total:
            pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
            log.info("  >> Saved checkpoint (%d rows)", len(rows))

    # Final save
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)

    log.info("")
    log.info("-" * 60)
    log.info("STEP 0 COMPLETE")
    log.info("  Total rows:   %d", len(rows))
    log.info("  Errors:       %d", errors)
    log.info("  Elapsed:      %.1f minutes", (time.time() - t0) / 60)
    log.info("  Saved to:     %s", OUT_PATH)
    log.info("=" * 60)
    return 1 if errors > max(total, 1) * 0.1 else 0


if __name__ == "__main__":
    sys.exit(main())
