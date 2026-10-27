#!/usr/bin/env python3
"""Calibrate ensemble judge against golden set."""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from refusal_judge import classify_refusal_ensemble, RI_VALUES


def main():
    parser = argparse.ArgumentParser(description="Calibrate ensemble judge")
    parser.add_argument("--golden", default="data/golden/judge_calibration.csv")
    parser.add_argument("--model", default="gpt-oss:20b")
    parser.add_argument("--n-calls", type=int, default=3)
    args = parser.parse_args()

    df = pd.read_csv(args.golden, dtype=str, keep_default_na=False)
    print(f"Loaded {len(df)} golden examples from {args.golden}")

    correct = 0
    total = 0
    errors = []

    for _, row in df.iterrows():
        q = row["question"]
        r = row["response"]
        gold = row["gold_ri"].strip()
        rationale = row.get("rationale", "")

        result = classify_refusal_ensemble(q, r, model=args.model, n_calls=args.n_calls)
        pred = result["ri"]
        match = pred == gold
        correct += int(match)
        total += 1

        status = "OK" if match else "FAIL"
        print(
            f"  [{status}] {row['sample_id']} | gold={gold} pred={pred} | "
            f"votes={result['ensemble_labels']} agree={result['ensemble_agreement']:.2f}"
        )
        if not match:
            errors.append(
                {
                    "id": row["sample_id"],
                    "gold": gold,
                    "pred": pred,
                    "rationale": rationale,
                    "votes": result["ensemble_labels"],
                }
            )

    acc = correct / total if total else 0
    print(f"\nAccuracy: {correct}/{total} = {acc:.3f}")
    print(f"Target: >= 0.80 for publication")

    if errors:
        print("\nMismatches:")
        for e in errors:
            print(f"  {e['id']}: gold={e['gold']} pred={e['pred']} ({e['rationale']})")

    return 0 if acc >= 0.80 else 1


if __name__ == "__main__":
    sys.exit(main())
