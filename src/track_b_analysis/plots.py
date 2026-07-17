"""Matplotlib figures for Track B analysis (dependency-light).

All functions accept an optional ``ax`` so they compose in a notebook grid, and
return the Matplotlib ``Axes``. CI columns (``ci_low``/``ci_high``) from the
metrics module are rendered as asymmetric error bars.
"""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LANGUAGE_ORDER = ("english", "hindi", "hinglish")
_LANG_COLORS = {
    "english": "#4C72B0",
    "hindi": "#DD8452",
    "hinglish": "#55A868",
}


def _ordered(values: Sequence[str], order: Sequence[str]) -> list[str]:
    present = list(values)
    head = [v for v in order if v in present]
    tail = sorted(v for v in present if v not in order)
    return head + tail


def _asym_err(rates: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    lower = np.clip(rates - lo, 0, None)
    upper = np.clip(hi - rates, 0, None)
    return np.vstack([lower, upper])


def plot_or_bars(
    or_table: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "Over-refusal rate by language × task",
    caption: str | None = None,
) -> plt.Axes:
    """Grouped bar chart of over-refusal rate with Wilson CI error bars.

    Expects columns: ``language``, ``task_label`` (or ``task``), ``rate``,
    ``ci_low``, ``ci_high``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    table = or_table.copy()
    task_col = "task_label" if "task_label" in table.columns else "task"
    languages = _ordered(table["language"].unique(), LANGUAGE_ORDER)
    tasks = sorted(table[task_col].unique())

    n_lang = len(languages)
    width = 0.8 / max(n_lang, 1)
    x = np.arange(len(tasks))

    for i, lang in enumerate(languages):
        sub = table[table["language"] == lang].set_index(task_col)
        rates = np.array([sub.loc[t, "rate"] if t in sub.index else 0.0 for t in tasks], dtype=float)
        lo = np.array([sub.loc[t, "ci_low"] if t in sub.index else 0.0 for t in tasks], dtype=float)
        hi = np.array([sub.loc[t, "ci_high"] if t in sub.index else 0.0 for t in tasks], dtype=float)
        offset = (i - (n_lang - 1) / 2) * width
        ax.bar(
            x + offset, rates, width,
            label=lang,
            color=_LANG_COLORS.get(lang, None),
            yerr=_asym_err(rates, lo, hi),
            capsize=4,
            error_kw={"elinewidth": 1, "alpha": 0.8},
        )

    ax.set_xticks(x)
    ax.set_xticklabels(tasks)
    ax.set_ylabel("Over-refusal rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(title="language")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    if caption:
        ax.text(0.0, -0.18, caption, transform=ax.transAxes, fontsize=8, color="#666")
    return ax


def plot_or_heatmap(
    or_table: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "Over-refusal rate heatmap",
    cmap: str = "YlOrRd",
) -> plt.Axes:
    """Language × task heatmap of over-refusal rate with annotated cells."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))

    table = or_table.copy()
    task_col = "task_label" if "task_label" in table.columns else "task"
    languages = _ordered(table["language"].unique(), LANGUAGE_ORDER)
    tasks = sorted(table[task_col].unique())

    grid = np.full((len(languages), len(tasks)), np.nan)
    for r, lang in enumerate(languages):
        for c, task in enumerate(tasks):
            cell = table[(table["language"] == lang) & (table[task_col] == task)]
            if len(cell):
                grid[r, c] = float(cell["rate"].iloc[0])

    im = ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(tasks)))
    ax.set_xticklabels(tasks)
    ax.set_yticks(np.arange(len(languages)))
    ax.set_yticklabels(languages)
    ax.set_title(title)

    for r in range(len(languages)):
        for c in range(len(tasks)):
            if not np.isnan(grid[r, c]):
                val = grid[r, c]
                ax.text(c, r, f"{val:.0%}", ha="center", va="center",
                        color="white" if val > 0.5 else "black", fontsize=10)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("OR rate")
    return ax


def plot_delta_or(
    delta_table: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "Δ over-refusal vs English",
) -> plt.Axes:
    """Horizontal bar chart of Δ OR (language − baseline) with CI whiskers.

    Expects columns from :func:`metrics.compare_languages`: ``comparison``,
    ``task``, ``delta_or``, ``delta_ci_low``, ``delta_ci_high``.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    table = delta_table.copy()
    if table.empty:
        ax.set_title(title + " (no data)")
        return ax

    table["label"] = table["comparison"].astype(str) + " · " + table["task"].astype(str)
    table = table.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(table))
    deltas = table["delta_or"].to_numpy(dtype=float)
    lo = table["delta_ci_low"].to_numpy(dtype=float)
    hi = table["delta_ci_high"].to_numpy(dtype=float)
    err = np.vstack([np.clip(deltas - lo, 0, None), np.clip(hi - deltas, 0, None)])
    colors = ["#C44E52" if d > 0 else "#4C72B0" for d in deltas]

    ax.barh(y, deltas, color=colors, xerr=err, capsize=4,
            error_kw={"elinewidth": 1, "alpha": 0.8})
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(table["label"])
    ax.set_xlabel("Δ over-refusal rate (localized − English)")
    ax.set_title(title)
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    return ax


def plot_refusal_breakdown(
    benign_table: pd.DataFrame,
    harmful_table: pd.DataFrame,
    *,
    ax: plt.Axes | None = None,
    title: str = "Benign over-refusal vs harmful-content refusal",
) -> plt.Axes:
    """Side-by-side refusal rates: benign-task OR vs harmful-content refusal, by language."""
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    def _by_lang(table: pd.DataFrame) -> dict[str, float]:
        if table is None or table.empty:
            return {}
        agg = (
            table.assign(_num=table["rate"] * table["n"])
            .groupby("language")[["_num", "n"]]
            .sum()
        )
        return {lang: (row["_num"] / row["n"] if row["n"] else 0.0) for lang, row in agg.iterrows()}

    benign = _by_lang(benign_table)
    harmful = _by_lang(harmful_table)
    languages = _ordered(set(benign) | set(harmful), LANGUAGE_ORDER)
    x = np.arange(len(languages))
    width = 0.38

    ax.bar(x - width / 2, [benign.get(l, 0.0) for l in languages], width,
           label="benign-task OR (lower=better)", color="#C44E52")
    ax.bar(x + width / 2, [harmful.get(l, 0.0) for l in languages], width,
           label="harmful-content refusal (higher=better)", color="#55A868")
    ax.set_xticks(x)
    ax.set_xticklabels(languages)
    ax.set_ylabel("rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    return ax
