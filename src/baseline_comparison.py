"""Benchmark comparison table vs published SafeConstellations baselines (template only)."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUT_PATH = PROJECT_ROOT / "results" / "baseline_comparison.md"

# Published reference points from SafeConstellations paper (Table 2); not filled with local runs.
PUBLISHED_BASELINES = [
    {
        "source": "SafeConstellations (Sakonii et al.)",
        "model": "Llama-3.1-8B-Instruct",
        "metric": "Overall OR rate (test)",
        "value": "17.77%",
        "citation": "Table 2",
    },
    {
        "source": "SafeConstellations (Sakonii et al.)",
        "model": "Qwen2.5-7B-Instruct",
        "metric": "Overall OR rate (test)",
        "value": "8.15%",
        "citation": "Table 2",
    },
    {
        "source": "SafeConstellations (Sakonii et al.)",
        "model": "Llama-3.1-8B-Instruct",
        "metric": "Sentiment OR rate",
        "value": "36.4%",
        "citation": "Table 2 / task breakdown",
    },
    {
        "source": "SafeConstellations (Sakonii et al.)",
        "model": "Llama-3.1-8B-Instruct",
        "metric": "Translation OR rate",
        "value": "46.7%",
        "citation": "Table 2 / task breakdown",
    },
]


def render_baseline_table(
  *,
    local_run_label: str = "This replication (fill after full run)",
    languages: list[str] | None = None,
) -> str:
    """
    Markdown comparison table: published baselines vs empty local columns.

    No fake numbers — local cells are placeholders until a full run completes.
    """
    languages = languages or ["english", "hindi", "hinglish"]

    lines = [
        "# Benchmark comparison vs published baselines",
        "",
        "Template for comparing local evaluator output to published SafeConstellations results.",
        "**Do not treat smoke-test or partial runs as publishable numbers.**",
        "",
        "## Overall over-refusal rate (OR)",
        "",
        "| Source | Model | Metric | Published | "
        f"{local_run_label} | Δ (local − published) |",
        "| --- | --- | --- | ---: | ---: | ---: |",
    ]

    for row in PUBLISHED_BASELINES:
        if "Overall" not in row["metric"]:
            continue
        lines.append(
            f"| {row['source']} | {row['model']} | {row['metric']} ({row['citation']}) | "
            f"{row['value']} | — | — |"
        )
    lines.append(
        f"| Local track | (your model) | Overall OR rate (test) | — | — | — |"
    )

    lines.extend(
        [
            "",
            "## Task-level OR (Llama-3.1-8B-Instruct)",
            "",
            "| Task | Published (SafeConstellations) | Local replication | Δ |",
            "| --- | ---: | ---: | ---: |",
            "| Sentiment | 36.4% | — | — |",
            "| Translation | 46.7% | — | — |",
            "| Cryptanalysis | (see paper) | — | — |",
            "| RAG QA | (see paper) | — | — |",
            "",
            "## Multilingual extension (Track B)",
            "",
            "| Language | Task | Published baseline | Local OR rate | n | Notes |",
            "| --- | --- | ---: | ---: | ---: | --- |",
        ]
    )

    for lang in languages:
        for task in ("sentiment", "translation"):
            pub = "— (English-only in paper)" if lang != "english" else "see Table 2"
            lines.append(f"| {lang} | {task} | {pub} | — | — | fill after QA-passing run |")

    lines.extend(
        [
            "",
            "## Judge calibration",
            "",
            "| Check | Published / reference | Local |",
            "| --- | --- | --- |",
            "| OR-Bench judge agreement (manual audit) | (not reported) | — |",
            "| Inter-annotator κ (human ri labels) | — | — |",
            "| Judge ensemble disagreement rate | — | — |",
            "",
            "## How to fill",
            "",
            "1. Run full replication: `python main.py replicate --force-full` (human-approved).",
            "2. Run Track B with reviewed translations passing QA.",
            "3. Paste metrics from `results/paper_metrics.csv` and `results/track_b_metrics.csv`.",
            "4. Re-generate: `python main.py baseline-table`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_baseline_comparison(out_path: Path | None = None) -> Path:
    """Write the baseline comparison markdown template."""
    out_path = out_path or DEFAULT_OUT_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(render_baseline_table(), encoding="utf-8")
    logger.info("Wrote baseline comparison template to %s", out_path)
    return out_path
