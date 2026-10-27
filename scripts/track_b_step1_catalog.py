#!/usr/bin/env python3
"""Step 1: Build the Track B catalog for a given split.

Default: test split (270 × 3 = 810 rows).
Use --split train for train split (777 × 3 = 2331 rows).
"""

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("track_b_catalog")

from src.paper_benchmark import load_paper_benchmark
from src.track_b_prompts import build_full_track_b_catalog

OUT_DIR = Path(__file__).resolve().parent.parent / "results"
LOG_DIR = Path(__file__).resolve().parent.parent / "logs"

def main():
    parser = argparse.ArgumentParser(description="Build Track B catalog")
    parser.add_argument(
        "--split", choices=["test", "train"], default="test",
        help="Dataset split (default: test)",
    )
    args = parser.parse_args()

    t0 = time.time()
    LOG_DIR.mkdir(exist_ok=True)

    log.info("=" * 60)
    log.info("STEP 1: BUILD TRACK B CATALOG (split=%s)", args.split)
    log.info("=" * 60)

    bench = load_paper_benchmark(split=args.split)
    # Exclude cryptanalysis (non-benign for our purposes)
    bench = bench[bench["intended_task"] != "cryptanalysis"].copy()
    log.info("Loaded %d rows from benchmark (excluded cryptanalysis)", len(bench))

    catalog = build_full_track_b_catalog(bench)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"track_b_catalog_{args.split}.csv"
    catalog.to_csv(out_path, index=False)

    # Summary stats
    n_total = len(catalog)
    n_ids = catalog["sample_id"].nunique()
    n_degen = (catalog["is_degenerate_translate"] == "True").sum()

    log.info("-" * 60)
    log.info("CATALOG SUMMARY (split=%s)", args.split)
    log.info("-" * 60)
    log.info("  Total rows:        %d", n_total)
    log.info("  Unique sample_ids: %d", n_ids)
    log.info("  Languages:         english, hindi, hinglish")
    log.info("  Degenerate rows:   %d (Hindi/Hinglish prompt → Hindi target)", n_degen)

    log.info("")
    log.info("Rows by language:")
    for lang, count in catalog["language"].value_counts().items():
        log.info("  %12s: %d", lang, count)

    log.info("")
    log.info("Rows by task × language:")
    pivot = catalog.groupby(["language", "intended_task"]).size().unstack(fill_value=0)
    log.info("\n%s", pivot.to_string())

    log.info("")
    log.info("Rows by text_type:")
    for tt, count in catalog["text_type"].value_counts().items():
        log.info("  %25s: %d", tt, count)

    log.info("")
    log.info("Saved catalog to %s", out_path)
    log.info("Step 1 completed in %.1f seconds", time.time() - t0)
    log.info("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
