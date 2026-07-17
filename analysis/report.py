"""Generate ANALYSIS_REPORT.md with methods section draft."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from analysis.config import REPORT_PATH, SMOKE_DISCLAIMER


def _md_table(df: pd.DataFrame, max_rows: int = 20) -> str:
    if df.empty:
        return "_No data._\n"
    sub = df.head(max_rows)
    cols = list(sub.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, row in sub.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if abs(v) < 10 else f"{v:.2g}")
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep] + rows) + "\n"


def write_analysis_report(
    *,
    merged: pd.DataFrame,
    by_language: pd.DataFrame,
    by_model: pd.DataFrame,
    by_category: pd.DataFrame,
    by_prompt_type: pd.DataFrame,
    chi_square: pd.DataFrame,
    language_effects: pd.DataFrame,
    out_paths: dict[str, Path],
    label: str,
    is_smoke: bool,
    confidence: float,
    report_path: Path = REPORT_PATH,
) -> Path:
    """Write ANALYSIS_REPORT.md at the project root."""
    n_rows = len(merged)
    n_valid = int(merged["ri_valid"].sum()) if "ri_valid" in merged.columns else 0
    tracks = ", ".join(sorted(merged["track"].unique())) if not merged.empty else "none"

    smoke_banner = ""
    if is_smoke:
        smoke_banner = f"\n> {SMOKE_DISCLAIMER}\n"

    methods = f"""## Methods (draft)

### Data ingestion

We ingest three result streams from `results/`:

| Track | Source files | Judge |
| --- | --- | --- |
| Track A (replicate) | `paper_raw.csv` + `paper_judgments.csv` | OR-Bench proxy (`gpt-oss:20b`) |
| Track A (Phase-1) | `classified_outputs.csv` | Rule-based classifier (`label_level1`) |
| Track B | `track_b_raw.csv` + `track_b_judgments.csv` | OR-Bench proxy |

Rows are normalized to a common schema (`track`, `language`, `model`, `category`,
`prompt_type`, `ri`, `refusal`, `over_refusal`) via `analysis/ingest.py`. Judgment
deduplication for Track B follows `src/track_b_analysis/loaders.py` (keep last
`row_key`). Phase-1 `REFUSAL` labels are mapped to `direct_refusal` as a proxy
for cross-track comparability; magnitudes across judge types should not be
compared directly.

### Metrics

**Over-refusal (OR)** follows SafeConstellations Eq. 9: a refusal on a *benign task*
(translate, sentiment) counts toward OR regardless of content harmfulness.
Denominator: rows with valid `ri ∈ {{direct_answer, direct_refusal, indirect_refusal}}`.

Rates are reported with **Wilson score {confidence:.0%} confidence intervals**
(computed analytically; no scipy dependency for CIs).

### Statistical tests

- **Chi-square test of independence** (refusal vs answered) across each grouping
  dimension (language, model, category, prompt type), with **Cramér's V** effect size.
- **Pairwise language comparisons** vs English: difference in OR rate with Wald CI
  and **Cohen's h** effect size.

### Figures and tables

Publication figures (PNG, 300 dpi) and a LaTeX `refusal_rates.tex` table are
written to the output directory listed below.
"""

    results_section = f"""## Results snapshot

- **Label:** {label}
- **Rows ingested:** {n_rows} (valid judgments: {n_valid})
- **Tracks:** {tracks}

### Over-refusal by language

{_md_table(by_language)}

### Over-refusal by model

{_md_table(by_model)}

### Over-refusal by category (task)

{_md_table(by_category)}

### Over-refusal by prompt type

{_md_table(by_prompt_type)}

### Chi-square tests (refusal × dimension)

{_md_table(chi_square)}

### Language pairwise effects vs English

{_md_table(language_effects)}
"""

    artifacts = "## Artifacts\n\n"
    for name, path in sorted(out_paths.items()):
        try:
            rel = path.relative_to(report_path.parent)
        except ValueError:
            rel = path
        artifacts += f"- `{name}`: `{rel}`\n"

    body = "\n".join([
        "# Multilingual over-refusal — analysis report",
        "",
        f"_Generated {date.today().isoformat()}._",
        smoke_banner,
        methods,
        results_section,
        artifacts,
        "",
        "## Reproducibility",
        "",
        "```bash",
        "# Smoke test (pipeline validation only)",
        "python -m analysis --limit 5 --label 'SMOKE n=5'",
        "",
        "# Full analysis (all ingested rows)",
        "python -m analysis --label 'full run'",
        "```",
        "",
    ])

    report_path.write_text(body, encoding="utf-8")
    return report_path
