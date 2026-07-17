"""Inter-annotator agreement metrics for OR-Bench refusal labels (ri)."""

from __future__ import annotations

import logging
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Matches refusal_judge.RI_VALUES and manual_audit.RI_VALUES
RI_VALUES = frozenset({"direct_answer", "direct_refusal", "indirect_refusal"})

ANNOTATOR_ID_COL = "annotator"
ITEM_ID_COL = "item_id"
LABEL_COL = "ri"


def _cohen_kappa(labels_a: Sequence[str], labels_b: Sequence[str]) -> float:
    """Cohen's kappa for two raters (scipy-free)."""
    if len(labels_a) != len(labels_b):
        raise ValueError("label sequences must be the same length")
    n = len(labels_a)
    if n == 0:
        return float("nan")
    agreements = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agreements / n
    all_labels = sorted(set(labels_a) | set(labels_b))
    pe = 0.0
    for lab in all_labels:
        p_a = sum(1 for x in labels_a if x == lab) / n
        p_b = sum(1 for x in labels_b if x == lab) / n
        pe += p_a * p_b
    if pe >= 1.0:
        return float("nan")
    return (po - pe) / (1 - pe)


def _fleiss_kappa(matrix: list[list[int]]) -> float:
    """
    Fleiss' kappa for 3+ raters rating the same items.

    ``matrix`` is n_items × n_categories count table.
    """
    if not matrix:
        return float("nan")
    n_items = len(matrix)
    n_categories = len(matrix[0])
    if n_categories == 0:
        return float("nan")

    n_raters = sum(matrix[0])
    if n_raters < 2:
        return float("nan")

    p_i_sum = 0.0
    for row in matrix:
        if sum(row) != n_raters:
            raise ValueError("each item must have the same number of ratings")
        p_i = sum(c * c - c for c in row) / (n_raters * (n_raters - 1))
        p_i_sum += p_i
    p_bar = p_i_sum / n_items

    col_totals = [sum(matrix[i][j] for i in range(n_items)) for j in range(n_categories)]
    p_e = sum(c * c for c in col_totals) / (n_items * n_raters) ** 2

    if p_e >= 1.0:
        return float("nan")
    return (p_bar - p_e) / (1 - p_e)


def percent_agreement(labels: Sequence[Sequence[str]]) -> dict:
    """
    Percent agreement across multiple raters on the same items.

    ``labels`` is a list of per-rater label lists (equal length).
    """
    if not labels:
        return {"n_items": 0, "agreement_rate": float("nan"), "unanimous": 0}
    n_items = len(labels[0])
    for seq in labels:
        if len(seq) != n_items:
            raise ValueError("all rater sequences must have equal length")
    unanimous = 0
    for i in range(n_items):
        votes = {seq[i] for seq in labels}
        if len(votes) == 1:
            unanimous += 1
    return {
        "n_items": n_items,
        "unanimous": unanimous,
        "agreement_rate": unanimous / n_items if n_items else float("nan"),
    }


def pairwise_cohen_kappa(
    annotations: pd.DataFrame,
    *,
    annotator_col: str = ANNOTATOR_ID_COL,
    item_col: str = ITEM_ID_COL,
    label_col: str = LABEL_COL,
) -> pd.DataFrame:
    """Cohen's kappa for every annotator pair."""
    pivot = annotations.pivot_table(
        index=item_col,
        columns=annotator_col,
        values=label_col,
        aggfunc="first",
    )
    annotators = list(pivot.columns)
    rows: list[dict] = []
    for a, b in combinations(annotators, 2):
        pair = pivot[[a, b]].dropna(how="any")
        if pair.empty:
            continue
        kappa = _cohen_kappa(pair[a].tolist(), pair[b].tolist())
        agree = (pair[a] == pair[b]).mean()
        rows.append(
            {
                "annotator_a": a,
                "annotator_b": b,
                "n_items": len(pair),
                "agreement_rate": float(agree),
                "cohen_kappa": float(kappa),
            }
        )
    return pd.DataFrame(rows)


def fleiss_kappa_from_annotations(
    annotations: pd.DataFrame,
    *,
    annotator_col: str = ANNOTATOR_ID_COL,
    item_col: str = ITEM_ID_COL,
    label_col: str = LABEL_COL,
    categories: Iterable[str] | None = None,
) -> dict:
    """Fleiss' kappa when 3+ annotators rate the same items."""
    categories = list(categories or sorted(RI_VALUES))
    cat_index = {c: i for i, c in enumerate(categories)}

    grouped = annotations.groupby([item_col, annotator_col])[label_col].first()
    items = sorted(annotations[item_col].unique())
    annotators = sorted(annotations[annotator_col].unique())
    n_raters = len(annotators)

    if n_raters < 2:
        return {"n_items": 0, "n_raters": n_raters, "fleiss_kappa": float("nan")}

    matrix: list[list[int]] = []
    for item in items:
        counts = [0] * len(categories)
        for ann in annotators:
            key = (item, ann)
            if key not in grouped.index:
                continue
            lab = grouped[key]
            if lab in cat_index:
                counts[cat_index[lab]] += 1
        if sum(counts) == n_raters:
            matrix.append(counts)

    return {
        "n_items": len(matrix),
        "n_raters": n_raters,
        "categories": categories,
        "fleiss_kappa": _fleiss_kappa(matrix) if matrix else float("nan"),
    }


def load_annotations_long(path: Path) -> pd.DataFrame:
    """
    Load long-format annotations CSV: item_id, annotator, ri.

  Also accepts wide format with columns annotator_*_ri and converts to long.
    """
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if ANNOTATOR_ID_COL in df.columns and LABEL_COL in df.columns:
        return df

    wide_cols = [c for c in df.columns if c.endswith("_ri") and c != LABEL_COL]
    if not wide_cols:
        raise ValueError(
            f"Expected long format ({ITEM_ID_COL}, {ANNOTATOR_ID_COL}, {LABEL_COL}) "
            f"or wide annotator columns ending in _ri"
        )

    id_col = ITEM_ID_COL if ITEM_ID_COL in df.columns else df.columns[0]
    rows: list[dict] = []
    for _, row in df.iterrows():
        item = str(row[id_col])
        for col in wide_cols:
            ann = col[: -len("_ri")]
            lab = str(row[col]).strip()
            if lab in RI_VALUES:
                rows.append({ITEM_ID_COL: item, ANNOTATOR_ID_COL: ann, LABEL_COL: lab})
    return pd.DataFrame(rows)


def compute_agreement_report(
    annotations: pd.DataFrame,
    *,
    annotator_col: str = ANNOTATOR_ID_COL,
    item_col: str = ITEM_ID_COL,
    label_col: str = LABEL_COL,
) -> dict:
    """
    Full agreement report for refusal labels from multiple annotators.

    Works with the manual_audit ``ri`` schema (direct_answer / direct_refusal /
    indirect_refusal).
    """
    work = annotations[
        annotations[label_col].astype(str).str.strip().isin(RI_VALUES)
    ].copy()
    if work.empty:
        return {"n_annotations": 0, "error": "no valid ri labels"}

    annotators = sorted(work[annotator_col].unique())
    pivot = work.pivot_table(
        index=item_col,
        columns=annotator_col,
        values=label_col,
        aggfunc="first",
    )
    label_matrix = [pivot[a].dropna().tolist() for a in annotators if a in pivot.columns]
    # Align by item for percent agreement
    complete = pivot.dropna(how="any")
    rater_labels = [complete[a].tolist() for a in annotators if a in complete.columns]

    report: dict = {
        "n_annotations": len(work),
        "n_items": work[item_col].nunique(),
        "n_annotators": len(annotators),
        "annotators": annotators,
        "label_distribution": work[label_col].value_counts().to_dict(),
        "percent_agreement": percent_agreement(rater_labels) if rater_labels else {},
        "pairwise_kappa": pairwise_cohen_kappa(
            work, annotator_col=annotator_col, item_col=item_col, label_col=label_col
        ).to_dict(orient="records"),
    }

    if len(annotators) >= 3 and not complete.empty:
        report["fleiss_kappa"] = fleiss_kappa_from_annotations(
            work[work[item_col].isin(complete.index)],
            annotator_col=annotator_col,
            item_col=item_col,
            label_col=label_col,
        )

    # Items with disagreement
    disagree_rows: list[dict] = []
    for item_id, row in complete.iterrows():
        votes = Counter(str(row[a]) for a in annotators)
        if len(votes) > 1:
            disagree_rows.append(
                {
                    item_col: item_id,
                    "votes": dict(votes),
                    "majority": votes.most_common(1)[0][0],
                }
            )
    report["disagreement_items"] = disagree_rows
    report["n_disagreements"] = len(disagree_rows)

    return report


def write_agreement_report(
    path: Path,
    out_path: Path | None = None,
) -> Path:
    """Load annotations from CSV and write JSON + markdown summary."""
    import json

    out_path = out_path or (PROJECT_ROOT / "results" / "refusal_agreement_report.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    annotations = load_annotations_long(path)
    report = compute_agreement_report(annotations)

    # JSON-serializable (DataFrame already converted)
    out_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote refusal agreement report to %s", out_path)
    return out_path
