"""Unit tests for the steering vector math (numpy/pytest)."""

from __future__ import annotations

import numpy as np
import pytest

from src.safeconstellations import linalg as L


def test_unit_vector_has_unit_norm():
    v = np.array([3.0, 4.0])
    u = L.unit_vector(v)
    assert np.isclose(np.linalg.norm(u), 1.0)
    np.testing.assert_allclose(u, np.array([0.6, 0.8]))


def test_unit_vector_zero_is_safe():
    z = np.zeros(5)
    np.testing.assert_array_equal(L.unit_vector(z), z)


def test_cosine_similarity_bounds_and_known_values():
    a = np.array([1.0, 0.0])
    assert L.cosine_similarity(a, a) == pytest.approx(1.0)
    assert L.cosine_similarity(a, np.array([0.0, 1.0])) == pytest.approx(0.0)
    assert L.cosine_similarity(a, np.array([-1.0, 0.0])) == pytest.approx(-1.0)
    # Zero vector -> defined as 0.0, not NaN.
    assert L.cosine_similarity(a, np.zeros(2)) == 0.0


def test_centroid_is_mean_over_rows():
    m = np.array([[0.0, 0.0], [2.0, 4.0], [4.0, 8.0]])
    np.testing.assert_allclose(L.centroid(m), np.array([2.0, 4.0]))


def test_centroid_rejects_bad_shapes():
    with pytest.raises(ValueError):
        L.centroid(np.array([1.0, 2.0, 3.0]))  # 1-D
    with pytest.raises(ValueError):
        L.centroid(np.zeros((0, 4)))  # empty


def test_refusal_and_steering_directions_are_opposite_units():
    refusal_c = np.array([1.0, 2.0, 3.0])
    target_c = np.array([0.0, 0.0, 0.0])
    rd = L.refusal_direction(refusal_c, target_c)
    sd = L.steering_direction(refusal_c, target_c)
    assert np.isclose(np.linalg.norm(rd), 1.0)
    assert np.isclose(np.linalg.norm(sd), 1.0)
    np.testing.assert_allclose(rd, -sd)
    # Refusal direction points from target toward refusal.
    assert L.cosine_similarity(rd, refusal_c - target_c) == pytest.approx(1.0)


def test_projection_scalar_and_vector_consistent():
    v = np.array([2.0, 3.0])
    d = np.array([0.0, 5.0])  # +y axis (non-unit)
    assert L.projection_scalar(v, d) == pytest.approx(3.0)
    np.testing.assert_allclose(L.project_onto(v, d), np.array([0.0, 3.0]))


def test_remove_component_is_orthogonal():
    v = np.array([2.0, 3.0, 4.0])
    d = np.array([1.0, 0.0, 0.0])
    resid = L.remove_component(v, d)
    np.testing.assert_allclose(resid, np.array([0.0, 3.0, 4.0]))
    # Residual is orthogonal to the removed direction.
    assert np.dot(resid, d) == pytest.approx(0.0)


def test_steer_additive_moves_by_alpha_along_unit_direction():
    v = np.array([1.0, 1.0])
    d = np.array([0.0, 10.0])  # +y, non-unit; only direction matters
    out = L.steer_additive(v, d, alpha=2.0)
    np.testing.assert_allclose(out, np.array([1.0, 3.0]))


def test_steer_toward_target_reduces_refusal_projection():
    # Build a refusal axis and a point sitting on the refusal side.
    refusal_c = np.array([0.0, 5.0])  # high on +y (refusal)
    target_c = np.array([0.0, 0.0])  # non-refusal manifold
    axis = L.refusal_direction(refusal_c, target_c)  # ~ +y unit

    v = np.array([1.0, 3.0])  # projection onto axis = 3.0
    before = L.projection_scalar(v, axis)
    steered = L.steer_toward_target(v, refusal_c, target_c, alpha=2.0)
    after = L.projection_scalar(steered, axis)

    assert before == pytest.approx(3.0)
    # Steering toward target lowers the refusal-axis projection by exactly alpha.
    assert after == pytest.approx(before - 2.0)


def test_steering_preserves_orthogonal_components():
    refusal_c = np.array([0.0, 5.0, 0.0])
    target_c = np.array([0.0, 0.0, 0.0])
    v = np.array([7.0, 3.0, -4.0])
    steered = L.steer_toward_target(v, refusal_c, target_c, alpha=2.0)
    # x and z (orthogonal to the y refusal axis) are untouched.
    assert steered[0] == pytest.approx(7.0)
    assert steered[2] == pytest.approx(-4.0)
