#!/usr/bin/env python3
"""Run ensemble judge on paper raw outputs (Track A)."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from refusal_judge import judge_responses_ensemble_df


def main():
    parser = argparse.ArgumentParser(description="Ensemble judge on paper raw")
    parser.add_argument("--raw-csv", default="results/paper_raw.csv")
    parser.add_argument("--out-csv", default="results/paper_judgments.csv")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--n-calls", type=int, default=3)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.raw_csv, dtype=str, keep_default_na=False)
    print(f"Loaded {len(df)} rows from {args.raw_csv}")

    result_df = judge_responses_ensemble_df(
        df,
        model=args.model,
        out_path=Path(args.out_csv),
        resume=not args.no_resume,
        n_calls=args.n_calls,
    )
    print(f"Wrote {len(result_df)} judgments to {args.out_csv}")

    ri_counts = result_df["ri"].value_counts()
    print("\nLabel distribution:")
    for label, count in ri_counts.items():
        print(f"  {label}: {count}")

    if "ensemble_agreement" in result_df.columns:
        avg_agree = result_df["ensemble_agreement"].astype(float).mean()
        print(f"\nAvg ensemble agreement: {avg_agree:.3f}")


if __name__ == "__main__":
    main()
