"""Round-trip (back-translate) similarity metrics for EN ↔ back_en.

Primary: chrF++ (lexical) + embedding cosine (semantic).
Secondary: BLEU + length ratio.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

import numpy as np

DEFAULT_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def length_ratio(en: str, back: str) -> float:
    n = len(en)
    if n == 0:
        return float("nan")
    return len(back) / n


def sentence_chrf(hypothesis: str, reference: str) -> float:
    """chrF++ (word_order=2). Score on 0–100 scale (sacrebleu)."""
    from sacrebleu.metrics import CHRF

    metric = CHRF(word_order=2)
    return float(metric.sentence_score(hypothesis, [reference]).score)


def sentence_bleu(hypothesis: str, reference: str) -> float:
    """Sentence BLEU on 0–100 scale (sacrebleu)."""
    from sacrebleu.metrics import BLEU

    metric = BLEU(effective_order=True)
    return float(metric.sentence_score(hypothesis, [reference]).score)


@lru_cache(maxsize=2)
def _embedder(model_name: str = DEFAULT_EMBED_MODEL):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


def embedding_cosine_batch(
    references: Sequence[str],
    hypotheses: Sequence[str],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
) -> np.ndarray:
    """Cosine similarity between paired EN reference and back-translated strings."""
    if len(references) != len(hypotheses):
        raise ValueError("references and hypotheses must have the same length")
    if not references:
        return np.array([], dtype=np.float64)

    model = _embedder(model_name)
    ref_emb = model.encode(
        list(references),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(references) > 200,
    )
    hyp_emb = model.encode(
        list(hypotheses),
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=len(hypotheses) > 200,
    )
    return np.sum(ref_emb * hyp_emb, axis=1).astype(np.float64)


def score_pairs(
    en_texts: Iterable[str],
    back_texts: Iterable[str],
    *,
    model_name: str = DEFAULT_EMBED_MODEL,
    batch_size: int = 64,
) -> dict[str, list[float]]:
    """Score aligned EN / back-EN pairs. Returns parallel lists of metrics."""
    ens = [str(x) for x in en_texts]
    backs = [str(x) for x in back_texts]
    if len(ens) != len(backs):
        raise ValueError("en_texts and back_texts length mismatch")

    chrf_scores = [sentence_chrf(h, r) for r, h in zip(ens, backs)]
    bleu_scores = [sentence_bleu(h, r) for r, h in zip(ens, backs)]
    cosines = embedding_cosine_batch(
        ens, backs, model_name=model_name, batch_size=batch_size
    ).tolist()
    ratios = [length_ratio(r, h) for r, h in zip(ens, backs)]
    return {
        "chrf": chrf_scores,
        "bleu": bleu_scores,
        "embed_cosine": cosines,
        "len_ratio": ratios,
    }
