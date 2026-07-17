"""CLI: ``python -m analysis [--limit 5]``."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from analysis.config import DEFAULT_OUT_DIR, SMOKE_DISCLAIMER
from analysis.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="End-to-end multilingual over-refusal analysis (Track A + Track B)",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Smoke test: keep first N sample_ids per track (NOT publishable)",
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR, help="Output directory")
    p.add_argument("--confidence", type=float, default=0.95, help="Wilson CI level")
    p.add_argument("--label", type=str, default=None, help="Run label for reports/meta")
    p.add_argument(
        "--tracks",
        nargs="+",
        default=["track_a_replicate", "track_a_phase1", "track_b"],
        choices=["track_a_replicate", "track_a_phase1", "track_b"],
        help="Which result streams to ingest",
    )
    p.add_argument("--no-report", action="store_true", help="Skip ANALYSIS_REPORT.md")
    p.add_argument("--no-figures", action="store_true", help="Skip PNG figures")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.limit is not None:
        logger.warning(SMOKE_DISCLAIMER)

    label = args.label or (f"SMOKE n={args.limit}" if args.limit else "full run")

    result = run_pipeline(
        out_dir=args.out,
        include_tracks=tuple(args.tracks),
        limit=args.limit,
        confidence=args.confidence,
        label=label,
        write_report=not args.no_report,
        write_figures=not args.no_figures,
    )

    print(result.summary())
    if result.meta.get("is_smoke"):
        print(f"\n⚠️  {SMOKE_DISCLAIMER}")

    print("\nOver-refusal by language:")
    print(result.by_language.to_string(index=False))

    print("\nChi-square tests:")
    print(result.chi_square.to_string(index=False))

    print("\nWrote:")
    for name, path in sorted(result.paths.items()):
        print(f"  {name}: {path}")

    return 0 if not result.merged.empty else 1


if __name__ == "__main__":
    sys.exit(main())
