"""LaTeX table generation for refusal-rate summaries."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def _fmt_rate(row: pd.Series) -> str:
    rate = row.get("rate", float("nan"))
    lo = row.get("ci_low", float("nan"))
    hi = row.get("ci_high", float("nan"))
    n = int(row.get("n", 0))
    if pd.isna(rate):
        return f"--- & {n}"
    return f"{rate:.1%} [{lo:.1%}, {hi:.1%}] & {n}"


def rates_to_latex(
    table: pd.DataFrame,
    *,
    group_cols: list[str],
    caption: str,
    label: str = "tab:refusal_rates",
) -> str:
    """Render a booktabs-style LaTeX table from a rates DataFrame."""
    if table.empty:
        return (
            f"% Empty table: {caption}\n"
            f"\\begin{{table}}[t]\n\\centering\n\\caption{{{caption}}}\n"
            f"\\label{{{label}}}\n\\begin{{tabular}}{{l r}}\n\\toprule\n"
            f"No data & --- \\\\\n\\bottomrule\n\\end{{tabular}}\n\\end{{table}}\n"
        )

    col_headers = " & ".join(group_cols) + r" & OR rate [95\% CI] & $n$ \\"
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{'l' * len(group_cols)}lr}}",
        r"\toprule",
        col_headers,
        r"\midrule",
    ]
    for _, row in table.iterrows():
        group_vals = " & ".join(str(row[c]) for c in group_cols)
        lines.append(f"{group_vals} & {_fmt_rate(row)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_latex_tables(
  *,
    by_language: pd.DataFrame,
    by_model: pd.DataFrame,
    by_category: pd.DataFrame,
    by_prompt_type: pd.DataFrame,
    full_breakdown: pd.DataFrame,
    out_dir: Path,
    smoke_note: str = "",
) -> Path:
    """Write the main LaTeX table file (full breakdown)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = "Over-refusal rates by track, language, model, category, and prompt type."
    if smoke_note:
        caption += f" {smoke_note}"
    tex = rates_to_latex(
        full_breakdown,
        group_cols=["track", "language", "model", "category", "prompt_type"],
        caption=caption,
        label="tab:or_rates_full",
    )
    # Append compact summary tables
    tex += "\n% --- Summary slices ---\n\n"
    for name, tbl, cols, lab in (
        ("language", by_language, ["language"], "tab:or_by_language"),
        ("model", by_model, ["model"], "tab:or_by_model"),
        ("category", by_category, ["category"], "tab:or_by_category"),
        ("prompt_type", by_prompt_type, ["prompt_type"], "tab:or_by_prompt_type"),
    ):
        tex += rates_to_latex(tbl, group_cols=cols, caption=f"OR rate by {name}.", label=lab)

    main_path = out_dir / "refusal_rates.tex"
    main_path.write_text(tex, encoding="utf-8")
    return main_path
