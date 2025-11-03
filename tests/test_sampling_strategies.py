"""Tests for sampling strategies."""

import pytest

from kmeanssa_ng.core.strategies.sampling import SamplingStrategy, UniformSampling
from kmeanssa_ng.riemannian_manifold import create_sphere


class TestSamplingStrategy:
    """Tests for abstract SamplingStrategy."""

    def test_abstract_cannot_instantiate(self):
        """Cannot instantiate abstract SamplingStrategy."""
        with pytest.raises(TypeError):
            SamplingStrategy()


class TestUniformSampling:
    """Tests for UniformSampling strategy."""

    def test_uniform_sampling_sphere(self):
        """Test uniform sampling on sphere."""
        sphere = create_sphere(dim=2)
        strategy = UniformSampling()

        points = sphere.sample_points(100, strategy=strategy)

        assert len(points) == 100
        assert all(hasattr(p, "coordinates") for p in points)

    def test_sample_points_requires_strategy(self):
        """Test that sample_points requires strategy parameter."""
        sphere = create_sphere(dim=2)

        # Should fail without strategy
        with pytest.raises(TypeError, match="missing.*required.*argument"):
            sphere.sample_points(100)

    def test_uniform_sampling_multiple_calls(self):
        """Test multiple uniform sampling calls."""
        sphere = create_sphere(dim=2)
        strategy = UniformSampling()

        points1 = sphere.sample_points(50, strategy=strategy)
        points2 = sphere.sample_points(50, strategy=strategy)

        assert len(points1) == 50
        assert len(points2) == 50
        # Points should be different (with very high probability)
        assert points1[0].coordinates[0] != points2[0].coordinates[0]
