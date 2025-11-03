"""Tests for sampling strategies."""

import pytest

from kmeanssa_ng.core.strategies.sampling import SamplingStrategy
from kmeanssa_ng.quantum_graph import generate_simple_graph
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.riemannian_manifold import create_sphere
from kmeanssa_ng.riemannian_manifold.sampling import UniformManifoldSampling


class TestSamplingStrategy:
    """Tests for abstract SamplingStrategy."""

    def test_abstract_cannot_instantiate(self):
        """Cannot instantiate abstract SamplingStrategy."""
        with pytest.raises(TypeError):
            SamplingStrategy()


class TestUniformNodeSampling:
    """Tests for UniformNodeSampling strategy (QuantumGraph)."""

    def test_uniform_node_sampling_graph(self):
        """Test uniform node sampling on quantum graph."""
        graph = generate_simple_graph(n_a=3)
        strategy = UniformNodeSampling()

        points = graph.sample_points(100, strategy=strategy)

        assert len(points) == 100
        assert all(hasattr(p, "edge") for p in points)
        # All points should be at nodes (position 0)
        assert all(p.position == 0.0 for p in points)

    def test_uniform_node_sampling_multiple_calls(self):
        """Test multiple uniform node sampling calls."""
        graph = generate_simple_graph(n_a=3)
        strategy = UniformNodeSampling()

        points1 = graph.sample_points(50, strategy=strategy)
        points2 = graph.sample_points(50, strategy=strategy)

        assert len(points1) == 50
        assert len(points2) == 50
        # Points should be distributed across nodes


class TestUniformManifoldSampling:
    """Tests for UniformManifoldSampling strategy (RiemannianManifold)."""

    def test_uniform_manifold_sampling_sphere(self):
        """Test uniform manifold sampling on sphere."""
        sphere = create_sphere(dim=2)
        strategy = UniformManifoldSampling()

        points = sphere.sample_points(100, strategy=strategy)

        assert len(points) == 100
        assert all(hasattr(p, "coordinates") for p in points)

    def test_uniform_manifold_sampling_multiple_calls(self):
        """Test multiple uniform manifold sampling calls."""
        sphere = create_sphere(dim=2)
        strategy = UniformManifoldSampling()

        points1 = sphere.sample_points(50, strategy=strategy)
        points2 = sphere.sample_points(50, strategy=strategy)

        assert len(points1) == 50
        assert len(points2) == 50
        # Points should be different (with very high probability)
        assert points1[0].coordinates[0] != points2[0].coordinates[0]


class TestStrategyRequired:
    """Test that strategy parameter is required."""

    def test_strategy_required_quantum_graph(self):
        """Test that sample_points requires strategy for quantum graphs."""
        graph = generate_simple_graph(n_a=3)

        with pytest.raises(TypeError, match="missing.*required.*argument"):
            graph.sample_points(100)

    def test_strategy_required_riemannian_manifold(self):
        """Test that sample_points requires strategy for Riemannian manifolds."""
        sphere = create_sphere(dim=2)

        with pytest.raises(TypeError, match="missing.*required.*argument"):
            sphere.sample_points(100)
