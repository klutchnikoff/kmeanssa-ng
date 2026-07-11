"""Tests for sampling strategies."""

import networkx as nx
import numpy as np
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

    def test_int_seed_is_normalized_once(self):
        """Successive draws on one instance must differ (RNG-2 regression).

        The int seed was re-wrapped in a fresh default_rng on every draw, so
        two sample() calls returned identical points. It is now normalized to
        one advancing Generator at construction.
        """
        graph = generate_simple_graph(n_a=5)
        strategy = UniformNodeSampling(random_state=0)

        first = [p.edge[0] for p in graph.sample_points(30, strategy=strategy)]
        second = [p.edge[0] for p in graph.sample_points(30, strategy=strategy)]
        assert first != second

        # Same seed, two instances -> reproducible first draw.
        again = UniformNodeSampling(random_state=0)
        assert [p.edge[0] for p in graph.sample_points(30, strategy=again)] == first

    def test_random_state_property_is_a_generator(self):
        """random_state is normalized to a Generator, however it was set."""
        strategy = UniformNodeSampling(random_state=7)
        assert isinstance(strategy.random_state, np.random.Generator)
        rng = np.random.default_rng(1)
        strategy.random_state = rng  # as run_parallel reassigns it per worker
        assert strategy.random_state is rng


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

    def test_uniform_node_sampling_does_not_mutate_the_space(self):
        """Sampling is a pure draw: it must not touch the graph's node measure.

        A strategy silently stamping obs_weight would let two algorithms
        sharing one space overwrite each other's reference measure; the
        measure is now always declared explicitly (register_observations or
        the obs_weight attribute).
        """
        graph = generate_simple_graph(n_a=3)
        nx.set_node_attributes(graph, {"A0": 0.7}, "obs_weight")

        graph.sample_points(100, strategy=UniformNodeSampling(random_state=0))

        assert nx.get_node_attributes(graph, "obs_weight") == {"A0": 0.7}


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

    def test_weighted_node_sampling_does_not_mutate_the_space(self):
        """Sampling is a pure draw: it must not touch the graph's node measure."""
        graph = generate_simple_graph(n_a=3)
        nx.set_node_attributes(graph, {0: 1.0, 1: 1.0, 2: 1.0}, "weight")
        nx.set_node_attributes(graph, {"A0": 0.7}, "obs_weight")

        graph.sample_points(100, strategy=WeightedNodeSampling(random_state=0))

        assert nx.get_node_attributes(graph, "obs_weight") == {"A0": 0.7}

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

    def test_uniform_manifold_sampling_reproducible(self):
        """random_state makes sphere sampling reproducible, independent of global RNG.

        Sampling must go through the strategy's generator, not geomstats' global
        RNG: a given random_state yields the same points even when the global
        numpy RNG is perturbed between calls.
        """
        sphere = create_sphere(dim=2)

        def coords(seed):
            pts = sphere.sample_points(
                50, strategy=UniformManifoldSampling(random_state=seed)
            )
            return np.array([p.coordinates for p in pts])

        np.random.seed(1)
        first = coords(7)
        np.random.seed(123_456)  # perturb the global RNG between the two draws
        second = coords(7)

        np.testing.assert_array_equal(first, second)


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
