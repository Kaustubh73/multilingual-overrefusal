"""Summary tables and plots for Phase 1 multilingual experiments."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.response_classifier import LEVEL1_LABELS, LEVEL2_LABELS

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SUMMARY_RATES_PATH = PROJECT_ROOT / "results" / "summary_rates.csv"
DEFAULT_FAILURE_DIST_PATH = PROJECT_ROOT / "results" / "failure_distribution.csv"
DEFAULT_PLOTS_DIR = PROJECT_ROOT / "results" / "plots"


def _rate(n: int, count: int) -> float:
    return count / n if n > 0 else 0.0


def _level1_column(df: pd.DataFrame) -> str:
    if "label_level1" in df.columns:
        return "label_level1"
    if "label" in df.columns:
        return "label"
    raise ValueError("classified_df must have label_level1 or label")


def build_summary_rates(classified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Summary table: language × model with success/refusal/failure rates.
    """
    l1_col = _level1_column(classified_df)
    rows = []
    for (language, model), group in classified_df.groupby(["language", "model"]):
        n = len(group)
        counts = group[l1_col].value_counts()
        rows.append(
            {
                "language": language,
                "model": model,
                "success_rate": _rate(n, int(counts.get("SUCCESS", 0))),
                "refusal_rate": _rate(n, int(counts.get("REFUSAL", 0))),
                "failure_rate": _rate(n, int(counts.get("FAILURE", 0))),
                "total": n,
            }
        )
    return pd.DataFrame(rows).sort_values(["language", "model"])


def build_failure_distribution(classified_df: pd.DataFrame) -> pd.DataFrame:
    """
    Failure distribution among FAILURE rows only: task_dev, semantic, repetition, other.
    """
    l1_col = _level1_column(classified_df)
    l2_col = "label_level2" if "label_level2" in classified_df.columns else None

    rows = []
    for (language, model), group in classified_df.groupby(["language", "model"]):
        failures = group[group[l1_col] == "FAILURE"]
        n_fail = len(failures)
        if n_fail == 0 or l2_col is None:
            rows.append(
                {
                    "language": language,
                    "model": model,
                    "task_dev": 0.0,
                    "semantic": 0.0,
                    "repetition": 0.0,
                    "other": 0.0,
                    "n_failures": n_fail,
                }
            )
            continue
        l2_counts = failures[l2_col].value_counts()
        rows.append(
            {
                "language": language,
                "model": model,
                "task_dev": _rate(n_fail, int(l2_counts.get("TASK_DEVIATION", 0))),
                "semantic": _rate(n_fail, int(l2_counts.get("SEMANTIC_DISTORTION", 0))),
                "repetition": _rate(n_fail, int(l2_counts.get("REPETITION", 0))),
                "other": _rate(n_fail, int(l2_counts.get("OTHER", 0))),
                "n_failures": n_fail,
            }
        )
    return pd.DataFrame(rows).sort_values(["language", "model"])


def save_summary_tables(
    classified_df: pd.DataFrame,
    rates_path: Path | None = None,
    failure_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rates_path = rates_path or DEFAULT_SUMMARY_RATES_PATH
    failure_path = failure_path or DEFAULT_FAILURE_DIST_PATH
    rates_path.parent.mkdir(parents=True, exist_ok=True)

    rates_df = build_summary_rates(classified_df)
    failure_df = build_failure_distribution(classified_df)

    rates_df.to_csv(rates_path, index=False)
    failure_df.to_csv(failure_path, index=False)
    logger.info("Saved summary rates to %s", rates_path)
    logger.info("Saved failure distribution to %s", failure_path)
    return rates_df, failure_df


# Backward-compatible alias
def build_language_model_summary(classified_df: pd.DataFrame) -> pd.DataFrame:
    return build_summary_rates(classified_df)


def save_summary(
    classified_df: pd.DataFrame,
    out_path: Path | None = None,
) -> pd.DataFrame:
    rates_df, _ = save_summary_tables(classified_df)
    return rates_df


def generate_plots(
    classified_df: pd.DataFrame,
    summary_df: pd.DataFrame | None = None,
    plots_dir: Path | None = None,
) -> None:
    """Bar plots and heatmaps using two-level labels."""
    plots_dir = plots_dir or DEFAULT_PLOTS_DIR
    plots_dir.mkdir(parents=True, exist_ok=True)

    l1_col = _level1_column(classified_df)
    summary_df = summary_df or build_summary_rates(classified_df)

    # --- Bar: success rate by language × model ---
    fig, ax = plt.subplots(figsize=(10, 5))
    languages = sorted(classified_df["language"].unique())
    models = sorted(classified_df["model"].unique())
    x = np.arange(len(languages))
    width = 0.8 / max(len(models), 1)

    for i, model in enumerate(models):
        sub = summary_df[summary_df["model"] == model]
        vals = [
            float(sub.loc[sub["language"] == lang, "success_rate"].iloc[0])
            if len(sub.loc[sub["language"] == lang]) > 0
            else 0.0
            for lang in languages
        ]
        ax.bar(x + i * width, vals, width, label=model)

    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels(languages)
    ax.set_ylabel("Rate")
    ax.set_title("Success rate by language and model")
    ax.legend(fontsize=8, loc="upper right")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(plots_dir / "bar_success_rate.png", dpi=150)
    plt.close(fig)

    # --- Heatmap: language × model refusal rate ---
    for metric, fname in (
        ("refusal_rate", "heatmap_refusal_rate.png"),
        ("failure_rate", "heatmap_failure_rate.png"),
    ):
        if metric not in summary_df.columns:
            continue
        pivot = summary_df.pivot(index="language", columns="model", values=metric)
        fig, ax = plt.subplots(figsize=(8, 4))
        im = ax.imshow(pivot.values, aspect="auto", cmap="Reds", vmin=0, vmax=1)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns, rotation=30, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        ax.set_title(metric.replace("_", " "))
        fig.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(plots_dir / fname, dpi=150)
        plt.close(fig)

    # --- Heatmap: language × level1 label ---
    lang_pivot = pd.crosstab(
        classified_df["language"],
        classified_df[l1_col],
        normalize="index",
    )
    for label in LEVEL1_LABELS:
        if label not in lang_pivot.columns:
            lang_pivot[label] = 0.0
    lang_pivot = lang_pivot[list(LEVEL1_LABELS)]

    fig, ax = plt.subplots(figsize=(8, 4))
    im = ax.imshow(lang_pivot.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax.set_xticks(range(len(LEVEL1_LABELS)))
    ax.set_xticklabels(LEVEL1_LABELS, rotation=45, ha="right")
    ax.set_yticks(range(len(lang_pivot.index)))
    ax.set_yticklabels(lang_pivot.index)
    ax.set_title("Level-1 label distribution by language")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(plots_dir / "heatmap_language_level1.png", dpi=150)
    plt.close(fig)

    # --- Bar: level-2 among failures ---
    if "label_level2" in classified_df.columns:
        failures = classified_df[classified_df[l1_col] == "FAILURE"]
        if len(failures) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            dist = failures["label_level2"].value_counts().reindex(LEVEL2_LABELS, fill_value=0)
            ax.bar(dist.index, dist.values, color="steelblue")
            ax.set_title("Failure subtype distribution (all slices)")
            ax.tick_params(axis="x", rotation=45)
            fig.tight_layout()
            fig.savefig(plots_dir / "bar_failure_level2.png", dpi=150)
            plt.close(fig)

    logger.info("Saved plots to %s", plots_dir)
