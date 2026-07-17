"""Pure vector math for representation-space steering.

All functions operate on numpy arrays and have no model/IO dependencies so they
are cheap to unit-test. Vectors are 1-D ``(d,)`` arrays; stacks of vectors are
2-D ``(n, d)`` arrays (one row per example).
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def unit_vector(v: np.ndarray) -> np.ndarray:
    """Return v / ||v||. A zero vector is returned unchanged (norm guarded)."""
    v = np.asarray(v, dtype=np.float64)
    norm = np.linalg.norm(v)
    if norm < EPS:
        return v.copy()
    return v / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity in [-1, 1]; 0.0 if either vector is ~zero."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < EPS or nb < EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def centroid(matrix: np.ndarray) -> np.ndarray:
    """Mean vector over rows of a ``(n, d)`` stack."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(f"centroid expects a 2-D (n, d) array, got shape {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("centroid of an empty stack is undefined")
    return matrix.mean(axis=0)


def refusal_direction(refusal_centroid: np.ndarray, target_centroid: np.ndarray) -> np.ndarray:
    """Unit vector pointing from the non-refusal manifold toward refusals.

    Defined as ``unit(refusal_centroid - target_centroid)``.
    """
    return unit_vector(np.asarray(refusal_centroid, np.float64) - np.asarray(target_centroid, np.float64))


def steering_direction(refusal_centroid: np.ndarray, target_centroid: np.ndarray) -> np.ndarray:
    """Unit vector pointing toward the non-refusal (target) manifold.

    This is the negative of :func:`refusal_direction`.
    """
    return unit_vector(np.asarray(target_centroid, np.float64) - np.asarray(refusal_centroid, np.float64))


def projection_scalar(v: np.ndarray, direction: np.ndarray) -> float:
    """Signed length of v along ``direction`` (direction need not be unit)."""
    u = unit_vector(direction)
    return float(np.dot(np.asarray(v, np.float64), u))


def project_onto(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Vector component of v along ``direction``: (v·u) u with u = unit(direction)."""
    u = unit_vector(direction)
    return float(np.dot(np.asarray(v, np.float64), u)) * u


def remove_component(v: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Ablate the component of v along ``direction`` (orthogonal projection)."""
    return np.asarray(v, np.float64) - project_onto(v, direction)


def steer_additive(v: np.ndarray, direction: np.ndarray, alpha: float) -> np.ndarray:
    """Push v by ``alpha`` units along the (unit-normalized) ``direction``."""
    return np.asarray(v, np.float64) + alpha * unit_vector(direction)


def steer_toward_target(
    v: np.ndarray,
    refusal_centroid: np.ndarray,
    target_centroid: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Move v toward the non-refusal manifold by ``alpha`` units.

    This is the additive SafeConstellations-style intervention.
    """
    return steer_additive(v, steering_direction(refusal_centroid, target_centroid), alpha)
