"""Runnable demo for the SafeConstellations toy pipeline (CPU-only, no downloads).

Run from the project root::

    python -m src.safeconstellations.demo
    python -m src.safeconstellations.demo --alpha 2.5 --threshold 0.85

This exercises the full plumbing on synthetic data and prints baseline vs steered
over-refusal / harmful-refusal rates so you can verify the pipeline works without
a GPU. Use ``--backend hf`` only if you intend to load a real model (opt-in).
"""

from __future__ import annotations

import argparse
import logging

from src.safeconstellations.config import HiddenStateConfig, SteeringConfig, ToyWorldConfig
from src.safeconstellations.pipeline import format_report, run_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="SafeConstellations toy steering demo (CPU)")
    parser.add_argument("--backend", default="toy", choices=("toy", "hf"),
                        help="Hidden-state backend (default: toy synthetic; 'hf' is opt-in)")
    parser.add_argument("--alpha", type=float, default=2.5, help="Steering strength")
    parser.add_argument("--threshold", type=float, default=0.85,
                        help="Task-detection confidence gate")
    parser.add_argument("--mode", default="additive", choices=("additive", "ablate"),
                        help="Steering mode")
    parser.add_argument("--seed", type=int, default=0, help="Toy world seed")
    parser.add_argument("--quiet", action="store_true", help="Suppress INFO logs")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    hidden_config = HiddenStateConfig(backend=args.backend, toy=ToyWorldConfig(seed=args.seed))
    steering_config = SteeringConfig(alpha=args.alpha, detection_threshold=args.threshold, mode=args.mode)

    result = run_pipeline(hidden_config=hidden_config, steering_config=steering_config)
    print()
    print(format_report(result))

    # Plumbing sanity check: steering should not *increase* over-refusal.
    bor = result.summary._pct(result.summary.baseline_or)
    sor = result.summary._pct(result.summary.steered_or)
    print()
    print(f"OR change: {bor:.0%} -> {sor:.0%} (Δ {sor - bor:+.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
