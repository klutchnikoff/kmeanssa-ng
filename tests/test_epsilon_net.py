"""Tests for epsilon-net placement strategies."""

import numpy as np
import pytest

from kmeanssa_ng import create_sphere, create_hyperbolic_space
from kmeanssa_ng.riemannian_manifold import (
    UniformNet,
    RepulsionNet,
    RepulsionNetExtrinsicSpeedup,
    FibonacciNet,
)


def _min_max_nn_ratio(points):
    """min/max nearest-neighbour geodesic distance on the sphere (1 = regular)."""
    d = np.arccos(np.clip(points @ points.T, -1, 1))
    np.fill_diagonal(d, np.inf)
    nn = d.min(axis=1)
    return nn.min() / nn.max()


def _on_sphere(points):
    return np.allclose(np.linalg.norm(points, axis=1), 1.0, atol=1e-6)


class TestUniformNet:
    def test_shape_and_on_manifold(self):
        pts = UniformNet(random_state=0).build(create_sphere(2), 100)
        assert pts.shape == (100, 3)
        assert _on_sphere(pts)

    def test_reproducible(self):
        a = UniformNet(random_state=7).build(create_sphere(2), 50)
        b = UniformNet(random_state=7).build(create_sphere(2), 50)
        np.testing.assert_array_equal(a, b)


class TestFibonacciNet:
    def test_shape_and_deterministic(self):
        a = FibonacciNet().build(create_sphere(2), 200)
        b = FibonacciNet().build(create_sphere(2), 200)
        assert a.shape == (200, 3)
        assert _on_sphere(a)
        np.testing.assert_array_equal(a, b)

    def test_is_a_good_net(self):
        # The Fibonacci lattice is near-optimal: very regular spacing.
        assert _min_max_nn_ratio(FibonacciNet().build(create_sphere(2), 300)) > 0.7

    def test_rejects_non_sphere(self):
        with pytest.raises(ValueError, match="2-sphere"):
            FibonacciNet().build(create_hyperbolic_space(2), 100)

    def test_rejects_wrong_dimension(self):
        with pytest.raises(ValueError, match="2-sphere"):
            FibonacciNet().build(create_sphere(3), 100)


class TestRepulsionNet:
    def test_shape_and_on_manifold(self):
        pts = RepulsionNet(n_iter=50, random_state=0).build(create_sphere(2), 100)
        assert pts.shape == (100, 3)
        assert _on_sphere(pts)

    def test_reproducible(self):
        a = RepulsionNet(n_iter=50, random_state=1).build(create_sphere(2), 80)
        b = RepulsionNet(n_iter=50, random_state=1).build(create_sphere(2), 80)
        np.testing.assert_array_equal(a, b)

    def test_more_regular_than_uniform(self):
        sphere = create_sphere(2)
        n = 200
        uniform = UniformNet(random_state=2).build(sphere, n)
        repelled = RepulsionNet(n_iter=200, random_state=2).build(sphere, n)
        assert _min_max_nn_ratio(repelled) > _min_max_nn_ratio(uniform)
        assert _min_max_nn_ratio(repelled) > 0.4


class TestRepulsionNetExtrinsicSpeedup:
    def test_shape_and_on_manifold(self):
        pts = RepulsionNetExtrinsicSpeedup(n_iter=50, random_state=0).build(
            create_sphere(2), 100
        )
        assert pts.shape == (100, 3)
        assert _on_sphere(pts)

    def test_reproducible(self):
        a = RepulsionNetExtrinsicSpeedup(n_iter=50, random_state=1).build(
            create_sphere(2), 80
        )
        b = RepulsionNetExtrinsicSpeedup(n_iter=50, random_state=1).build(
            create_sphere(2), 80
        )
        np.testing.assert_array_equal(a, b)

    def test_equivalent_to_intrinsic_on_sphere(self):
        # On S^2 the embedding is isometric and chordal order == geodesic order,
        # so the extrinsic KD-tree and the intrinsic search select the same
        # neighbours: the two strategies must reach nets of equivalent regularity.
        sphere = create_sphere(2)
        n = 200
        intrinsic = RepulsionNet(n_iter=150, random_state=3).build(sphere, n)
        extrinsic = RepulsionNetExtrinsicSpeedup(n_iter=150, random_state=3).build(
            sphere, n
        )
        r_int = _min_max_nn_ratio(intrinsic)
        r_ext = _min_max_nn_ratio(extrinsic)
        assert r_int > 0.4 and r_ext > 0.4
        assert abs(r_int - r_ext) < 0.1
