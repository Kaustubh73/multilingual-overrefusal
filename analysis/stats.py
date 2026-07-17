"""Chi-square tests and effect sizes for refusal-rate comparisons."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from src.track_b_analysis.metrics import z_for_confidence


def _chi2_sf(chi2: float, df: int) -> float:
    """Survival function P(Chi2_df > chi2) via regularized incomplete gamma."""
    if df <= 0 or chi2 < 0:
        return float("nan")
    if chi2 == 0:
        return 1.0
    # Use scipy if available; otherwise Wilson-Hilferty normal approximation.
    try:
        from scipy.stats import chi2 as chi2_dist

        return float(chi2_dist.sf(chi2, df))
    except ImportError:
        # Wilson–Hilferty: (chi2/df)^(1/3) ~ N(1 - 2/(9df), 2/(9df))
        z = (chi2 / df) ** (1 / 3) - (1 - 2 / (9 * df))
        z /= math.sqrt(2 / (9 * df))
        # Standard normal survival via error function
        return 0.5 * math.erfc(z / math.sqrt(2))


def chi_square_independence(table: np.ndarray) -> dict:
    """Pearson chi-square test of independence on a contingency table.

    Returns chi2, df, p_value, cramers_v, n.
    """
    obs = np.asarray(table, dtype=float)
    if obs.ndim != 2 or obs.size == 0:
        return {"chi2": float("nan"), "df": 0, "p_value": float("nan"),
                "cramers_v": float("nan"), "n": 0}

    n = obs.sum()
    if n == 0:
        return {"chi2": float("nan"), "df": 0, "p_value": float("nan"),
                "cramers_v": float("nan"), "n": 0}

    row_sums = obs.sum(axis=1, keepdims=True)
    col_sums = obs.sum(axis=0, keepdims=True)
    expected = row_sums @ col_sums / n
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = float(np.nansum((obs - expected) ** 2 / expected))

    r, c = obs.shape
    df = (r - 1) * (c - 1)
    p_value = _chi2_sf(chi2, df) if df > 0 else float("nan")
    min_dim = min(r - 1, c - 1)
    cramers_v = math.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else float("nan")

    return {"chi2": chi2, "df": df, "p_value": p_value, "cramers_v": cramers_v, "n": int(n)}


def cohens_h(p1: float, p2: float) -> float:
    """Cohen's h for difference of two proportions."""
    p1 = min(max(p1, 0.0), 1.0)
    p2 = min(max(p2, 0.0), 1.0)
    return 2 * math.asin(math.sqrt(p1)) - 2 * math.asin(math.sqrt(p2))


def refusal_contingency(
    df: pd.DataFrame,
    row_col: str,
    *,
    success_col: str = "over_refusal",
    eligible_col: str | None = "ri_valid",
) -> pd.DataFrame:
    """Build a refusal (yes/no) × ``row_col`` contingency table."""
    work = df.copy()
    if eligible_col and eligible_col in work.columns:
        work = work[work[eligible_col].astype(bool)]
    if work.empty or row_col not in work.columns:
        return pd.DataFrame()

    work["_refused"] = work[success_col].astype(bool).map({True: "refused", False: "answered"})
    return pd.crosstab(work[row_col], work["_refused"])


def chi_square_by_dimension(
    df: pd.DataFrame,
    dimension: str,
    *,
    success_col: str = "over_refusal",
    eligible_col: str | None = "ri_valid",
) -> dict:
    """Chi-square test: refusal vs non-refusal across levels of ``dimension``."""
    table = refusal_contingency(df, dimension, success_col=success_col, eligible_col=eligible_col)
    if table.empty or table.shape[0] < 2 or table.shape[1] < 2:
        return {"dimension": dimension, "chi2": float("nan"), "df": 0,
                "p_value": float("nan"), "cramers_v": float("nan"), "n": 0}
    result = chi_square_independence(table.to_numpy())
    result["dimension"] = dimension
    return result


def pairwise_language_effects(
    df: pd.DataFrame,
    *,
    baseline: str = "english",
    confidence: float = 0.95,
) -> pd.DataFrame:
    """Pairwise Cohen's h and Wald CI for over-refusal rate vs baseline language."""
    work = df.copy()
    if "ri_valid" in work.columns:
        work = work[work["ri_valid"].astype(bool)]
    work["_or"] = work["over_refusal"].astype(bool)

    def _rate(sub: pd.DataFrame) -> tuple[int, int, float]:
        n = len(sub)
        c = int(sub["_or"].sum())
        return c, n, c / n if n else float("nan")

    base_sub = work[work["language"] == baseline]
    bc, bn, bp = _rate(base_sub)
    z = z_for_confidence(confidence)
    rows: list[dict] = []
    for lang in sorted(work["language"].unique()):
        if lang == baseline:
            continue
        lc, ln, lp = _rate(work[work["language"] == lang])
        if bn == 0 or ln == 0:
            continue
        delta = lp - bp
        se = math.sqrt(bp * (1 - bp) / bn + lp * (1 - lp) / ln)
        rows.append({
            "comparison": f"{lang} vs {baseline}",
            "language": lang,
            "task": "overall",
            "baseline_rate": bp,
            "language_rate": lp,
            "delta": delta,
            "delta_ci_low": delta - z * se,
            "delta_ci_high": delta + z * se,
            "cohens_h": cohens_h(lp, bp),
            "n_baseline": bn,
            "n_language": ln,
        })
    return pd.DataFrame(rows)


def run_all_statistical_tests(
    df: pd.DataFrame,
    dimensions: Sequence[str] = ("language", "model", "category", "prompt_type"),
) -> pd.DataFrame:
    """Chi-square + Cramér's V for each grouping dimension."""
    rows = [chi_square_by_dimension(df, dim) for dim in dimensions]
    return pd.DataFrame(rows)
