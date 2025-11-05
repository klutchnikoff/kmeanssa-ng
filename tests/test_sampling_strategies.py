"""Tests for sampling strategies."""

import networkx as nx
import pytest

from kmeanssa_ng.core.strategies.sampling import SamplingStrategy
from kmeanssa_ng.quantum_graph import generate_simple_graph
from kmeanssa_ng.quantum_graph.sampling import (
    UniformEdgeSampling,
    UniformNodeSampling,
    WeightedNodeSampling,
)
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

    def test_uniform_node_sampling_tracks_observations(self):
        """Test that uniform node sampling tracks nb_obs."""
        graph = generate_simple_graph(n_a=3)
        strategy = UniformNodeSampling()

        graph.sample_points(100, strategy=strategy)

        # Check that nb_obs attributes were set
        nb_obs = nx.get_node_attributes(graph, "nb_obs")
        assert len(nb_obs) > 0
        assert sum(nb_obs.values()) == 100


class TestUniformEdgeSampling:
    """Tests for UniformEdgeSampling strategy (QuantumGraph)."""

    def test_uniform_edge_sampling_graph(self):
        """Test uniform edge sampling on quantum graph."""
        graph = generate_simple_graph(n_a=3, n_aa=2, bridge_length=2.0)
        strategy = UniformEdgeSampling()

        points = graph.sample_points(100, strategy=strategy)

        assert len(points) == 100
        assert all(hasattr(p, "edge") for p in points)
        assert all(hasattr(p, "position") for p in points)

    def test_uniform_edge_sampling_positions(self):
        """Test that edge sampling produces positions along edges."""
        graph = generate_simple_graph(n_a=2, bridge_length=5.0)
        strategy = UniformEdgeSampling()

        points = graph.sample_points(50, strategy=strategy)

        # At least some points should not be at position 0
        positions = [p.position for p in points]
        assert any(pos > 0 for pos in positions)

    def test_uniform_edge_sampling_no_edges_raises(self):
        """Test that sampling from empty graph raises ValueError."""
        from kmeanssa_ng import QuantumGraph

        graph = QuantumGraph()
        strategy = UniformEdgeSampling()

        with pytest.raises(ValueError, match="Cannot sample from graph with no edges"):
            graph.sample_points(10, strategy=strategy)

    def test_uniform_edge_sampling_multiple_calls(self):
        """Test multiple uniform edge sampling calls."""
        graph = generate_simple_graph(n_a=3)
        strategy = UniformEdgeSampling()

        points1 = graph.sample_points(50, strategy=strategy)
        points2 = graph.sample_points(50, strategy=strategy)

        assert len(points1) == 50
        assert len(points2) == 50


class TestWeightedNodeSampling:
    """Tests for WeightedNodeSampling strategy (QuantumGraph)."""

    def test_weighted_node_sampling_with_weights(self):
        """Test weighted node sampling with node weights."""
        graph = generate_simple_graph(n_a=3)

        # Set weights: node 0 should be sampled 10x more than others
        nx.set_node_attributes(graph, {0: 10.0, 1: 1.0, 2: 1.0}, "weight")

        strategy = WeightedNodeSampling()
        points = graph.sample_points(120, strategy=strategy)

        assert len(points) == 120
        assert all(p.position == 0.0 for p in points)  # All at nodes

    def test_weighted_node_sampling_no_weights_raises(self):
        """Test that sampling without weights raises ValueError."""
        from kmeanssa_ng import QuantumGraph

        # Create graph without automatic weights
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        strategy = WeightedNodeSampling()

        with pytest.raises(ValueError, match="Nodes must have 'weight' attribute"):
            graph.sample_points(10, strategy=strategy)

    def test_weighted_node_sampling_negative_weights_raises(self):
        """Test that negative weights raise ValueError."""
        from kmeanssa_ng import QuantumGraph

        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        nx.set_node_attributes(graph, {0: 1.0, 1: -1.0, 2: 1.0}, "weight")

        strategy = WeightedNodeSampling()

        with pytest.raises(ValueError, match="All node weights must be positive"):
            graph.sample_points(10, strategy=strategy)

    def test_weighted_node_sampling_zero_weights_raises(self):
        """Test that zero weights raise ValueError."""
        from kmeanssa_ng import QuantumGraph

        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        nx.set_node_attributes(graph, {0: 1.0, 1: 0.0, 2: 1.0}, "weight")

        strategy = WeightedNodeSampling()

        with pytest.raises(ValueError, match="All node weights must be positive"):
            graph.sample_points(10, strategy=strategy)

    def test_weighted_node_sampling_tracks_observations(self):
        """Test that weighted sampling tracks nb_obs."""
        graph = generate_simple_graph(n_a=3)
        nx.set_node_attributes(graph, {0: 1.0, 1: 1.0, 2: 1.0}, "weight")

        strategy = WeightedNodeSampling()
        graph.sample_points(100, strategy=strategy)

        # Check that nb_obs attributes were set
        nb_obs = nx.get_node_attributes(graph, "nb_obs")
        assert len(nb_obs) > 0
        assert sum(nb_obs.values()) == 100

    def test_weighted_node_sampling_multiple_calls(self):
        """Test multiple weighted node sampling calls."""
        graph = generate_simple_graph(n_a=3)
        nx.set_node_attributes(graph, {0: 1.0, 1: 2.0, 2: 1.0}, "weight")

        strategy = WeightedNodeSampling()
        points1 = graph.sample_points(50, strategy=strategy)
        points2 = graph.sample_points(50, strategy=strategy)

        assert len(points1) == 50
        assert len(points2) == 50


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
