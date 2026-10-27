#!/usr/bin/env python3
"""Rate Hindi/Hinglish translation quality using gpt-oss:20b.

For each non-English prompt, asks the judge to rate:
- Accuracy of translation (1-5)
- Fluency (1-5)
- Meaning preservation (1-5)

Outputs results/translation_quality_{split}.csv
"""

import argparse
import logging
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
log = logging.getLogger("translation_quality")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
LOG_DIR = PROJECT_ROOT / "logs"
OLLAMA_URL = "http://localhost:11434/api/chat"
JUDGE_MODEL = "gpt-oss:20b"

RATING_PROMPT = """You are an expert evaluator of Hindi/Hinglish translations from English.

Rate the following translation on three dimensions (1-5 scale, where 5 is perfect):

1. **Accuracy**: Does the translation convey the exact same meaning as the English original?
2. **Fluency**: Is the Hindi/Hinglish natural and grammatically correct?
3. **Meaning preservation**: Does the translation preserve the intent, tone, and nuance?

English original:
{english}

{lang} translation:
{translation}

Provide your ratings in this exact format:
Accuracy: [1-5]
Fluency: [1-5]
Meaning: [1-5]
Issues: [brief description of any problems, or "None"]"""


def rate_translation(english: str, translation: str, lang: str) -> dict:
    prompt = RATING_PROMPT.format(english=english, translation=translation, lang=lang)
    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json().get("message", {}).get("content", "").strip()

    import re
    acc_match = re.search(r"Accuracy:\s*(\d)", raw)
    flu_match = re.search(r"Fluency:\s*(\d)", raw)
    mean_match = re.search(r"Meaning:\s*(\d)", raw)
    issues_match = re.search(r"Issues?:\s*(.+?)(?:\n|$)", raw)

    return {
        "accuracy": int(acc_match.group(1)) if acc_match else 0,
        "fluency": int(flu_match.group(1)) if flu_match else 0,
        "meaning": int(mean_match.group(1)) if mean_match else 0,
        "issues": issues_match.group(1).strip() if issues_match else "",
        "raw": raw,
    }


def main():
    parser = argparse.ArgumentParser(description="Rate translation quality")
    parser.add_argument("--split", choices=["test", "train"], default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    t0 = time.time()

    # Add file handler
    LOG_DIR.mkdir(exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"translation_quality_{args.split}_{timestamp}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logging.getLogger().addHandler(file_handler)

    log.info("=" * 60)
    log.info("TRANSLATION QUALITY RATING (split=%s)", args.split)
    log.info("Log file: %s", log_file)
    log.info("=" * 60)

    catalog_path = RESULTS_DIR / f"track_b_catalog_{args.split}.csv"
    out_path = RESULTS_DIR / f"translation_quality_{args.split}.csv"

    cat = pd.read_csv(catalog_path, dtype=str, keep_default_na=False)

    # Get English prompts and their Hindi/Hinglish counterparts
    en_prompts = cat[cat["language"] == "english"].set_index("prompt_id")
    non_en = cat[cat["language"] != "english"].copy()

    if args.limit:
        sids = non_en["sample_id"].unique()[: args.limit]
        non_en = non_en[non_en["sample_id"].isin(sids)]

    log.info("Catalog: %d non-English rows to rate", len(non_en))
    log.info("Judge model: %s", JUDGE_MODEL)

    # Check Ollama
    try:
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if JUDGE_MODEL not in models:
            log.error("Model %s not found. Available: %s", JUDGE_MODEL, models)
            return 1
        log.info("Ollama OK — %s available", JUDGE_MODEL)
    except Exception as e:
        log.error("Ollama not reachable: %s", e)
        return 1

    # Resume
    done_pids = set()
    rows = []
    if out_path.exists():
        existing = pd.read_csv(out_path, dtype=str, keep_default_na=False)
        done_pids = set(existing["prompt_id"].tolist())
        rows = existing.to_dict("records")
        log.info("Resuming: %d rows already rated", len(done_pids))

    total = len(non_en)
    remaining = sum(1 for _, r in non_en.iterrows() if r["prompt_id"] not in done_pids)
    log.info("Total: %d | Done: %d | Remaining: %d", total, len(done_pids), remaining)

    errors = 0
    for i, (_, row) in enumerate(non_en.iterrows(), start=1):
        pid = row["prompt_id"]
        if pid in done_pids:
            continue

        en_pid = pid.rsplit("__", 1)[0] + "__english"
        if en_pid not in en_prompts.index:
            log.warning("[%4d/%d] %s — no English counterpart found, skipping", i, total, pid)
            continue

        english_text = en_prompts.loc[en_pid, "plain_text"]
        translation = row["plain_text"]
        lang = row["language"]

        t_start = time.time()
        try:
            result = rate_translation(english_text, translation, lang)
            elapsed = time.time() - t_start

            avg = (result["accuracy"] + result["fluency"] + result["meaning"]) / 3
            log.info(
                "[%4d/%d] %s %-10s avg=%.1f (acc=%d flu=%d mean=%d) %.1fs | %s",
                i, total, pid, lang, avg,
                result["accuracy"], result["fluency"], result["meaning"],
                elapsed, result["issues"][:60],
            )
        except Exception as e:
            elapsed = time.time() - t_start
            log.error("[%4d/%d] %s %-10s ERROR (%.1fs): %s", i, total, pid, lang, elapsed, e)
            result = {"accuracy": 0, "fluency": 0, "meaning": 0, "issues": str(e), "raw": ""}
            errors += 1

        rows.append({
            "prompt_id": pid,
            "sample_id": row["sample_id"],
            "language": lang,
            "intended_task": row["intended_task"],
            "accuracy": result["accuracy"],
            "fluency": result["fluency"],
            "meaning": result["meaning"],
            "avg_score": round((result["accuracy"] + result["fluency"] + result["meaning"]) / 3, 2),
            "issues": result["issues"],
        })
        done_pids.add(pid)

        if i % 10 == 0 or i == total:
            pd.DataFrame(rows).to_csv(out_path, index=False)

    # Summary
    rdf = pd.DataFrame(rows)
    log.info("")
    log.info("=" * 60)
    log.info("SUMMARY")
    log.info("=" * 60)

    for lang in ["hindi", "hinglish"]:
        subset = rdf[rdf["language"] == lang]
        if len(subset) == 0:
            continue
        log.info("%s (N=%d):", lang.upper(), len(subset))
        log.info("  Accuracy:  %.1f ± %.1f (min=%d, max=%d)",
                 subset["accuracy"].mean(), subset["accuracy"].std(),
                 subset["accuracy"].min(), subset["accuracy"].max())
        log.info("  Fluency:   %.1f ± %.1f (min=%d, max=%d)",
                 subset["fluency"].mean(), subset["fluency"].std(),
                 subset["fluency"].min(), subset["fluency"].max())
        log.info("  Meaning:   %.1f ± %.1f (min=%d, max=%d)",
                 subset["meaning"].mean(), subset["meaning"].std(),
                 subset["meaning"].min(), subset["meaning"].max())
        log.info("  Avg:       %.1f ± %.1f", subset["avg_score"].mean(), subset["avg_score"].std())

        low = subset[subset["avg_score"] <= 2.5]
        log.info("  Low quality (avg ≤ 2.5): %d / %d (%.0f%%)",
                 len(low), len(subset), 100 * len(low) / len(subset))
        if len(low) > 0:
            for _, r in low.head(5).iterrows():
                log.info("    %s (task=%s): %.1f — %s",
                         r["sample_id"], r["intended_task"], r["avg_score"], r["issues"][:80])

    log.info("")
    log.info("Saved to %s", out_path)
    log.info("Done in %.1f minutes", (time.time() - t0) / 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
