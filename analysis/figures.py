"""Publication-ready matplotlib figures for the analysis pipeline."""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.track_b_analysis import plots as tb_plots

# Consistent styling for paper figures
PUBLICATION_RC = {
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def _apply_style() -> None:
    mpl.rcParams.update(PUBLICATION_RC)


def _rate_bar_with_ci(
    table: pd.DataFrame,
    x_col: str,
    *,
    ax: plt.Axes,
    title: str,
    ylabel: str = "Over-refusal rate",
    color: str = "#4C72B0",
) -> None:
    if table.empty:
        ax.set_title(title + " (no data)")
        return
    sub = table.sort_values(x_col)
    x = np.arange(len(sub))
    rates = sub["rate"].to_numpy(dtype=float)
    lo = sub["ci_low"].to_numpy(dtype=float)
    hi = sub["ci_high"].to_numpy(dtype=float)
    err = np.vstack([np.clip(rates - lo, 0, None), np.clip(hi - rates, 0, None)])
    labels = sub[x_col].astype(str).tolist()
    ax.bar(x, rates, color=color, yerr=err, capsize=4, error_kw={"elinewidth": 1, "alpha": 0.85})
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", linestyle=":", alpha=0.4)


def save_all_figures(
    *,
    by_language: pd.DataFrame,
    by_model: pd.DataFrame,
    by_category: pd.DataFrame,
    by_prompt_type: pd.DataFrame,
    language_task: pd.DataFrame,
    language_effects: pd.DataFrame,
    harmful: pd.DataFrame,
    out_dir: Path,
    smoke_label: str | None = None,
) -> dict[str, Path]:
    """Render the standard figure set and return path mapping."""
    _apply_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    caption = smoke_label or None
    written: dict[str, Path] = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    _rate_bar_with_ci(by_language, "language", ax=axes[0, 0], title="By language", color="#4C72B0")
    _rate_bar_with_ci(by_model, "model", ax=axes[0, 1], title="By model", color="#DD8452")
    _rate_bar_with_ci(by_category, "category", ax=axes[1, 0], title="By category (task)", color="#55A868")
    _rate_bar_with_ci(by_prompt_type, "prompt_type", ax=axes[1, 1], title="By prompt type", color="#C44E52")
    if caption:
        fig.suptitle(caption, fontsize=9, color="#666", y=0.02)
    fig.tight_layout()
    written["rates_grid"] = out_dir / "fig_rates_grid.png"
    fig.savefig(written["rates_grid"], bbox_inches="tight")
    plt.close(fig)

    if not language_task.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        tb_plots.plot_or_bars(language_task.rename(columns={"category": "task_label"}), ax=ax,
                              title="Over-refusal by language × category", caption=caption)
        written["language_category"] = out_dir / "fig_language_category.png"
        fig.savefig(written["language_category"], bbox_inches="tight")
        plt.close(fig)

    if not language_effects.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        effects = language_effects.rename(columns={"delta": "delta_or"})
        tb_plots.plot_delta_or(effects, ax=ax, title="Δ over-refusal vs English")
        written["language_effects"] = out_dir / "fig_language_effects.png"
        fig.savefig(written["language_effects"], bbox_inches="tight")
        plt.close(fig)

    if not language_task.empty and not harmful.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        benign_for_plot = language_task.rename(columns={"category": "task_label"})
        harmful_for_plot = harmful.rename(columns={"category": "task_label"})
        tb_plots.plot_refusal_breakdown(benign_for_plot, harmful_for_plot, ax=ax)
        written["refusal_breakdown"] = out_dir / "fig_refusal_breakdown.png"
        fig.savefig(written["refusal_breakdown"], bbox_inches="tight")
        plt.close(fig)

    return written
