"""Tests for Lloyd's algorithm."""

import pytest
import random
import numpy as np

from kmeanssa_ng import (
    generate_sbm,
    Lloyd,
    MostFrequentNodeUpdate,
    RiemannianManifold,
    RiemannianCenter,
    KarcherFrechetMean,
)
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.riemannian_manifold.sampling import UniformManifoldSampling
from kmeanssa_ng.core.strategies.initialization import RandomInit
from geomstats.geometry.hypersphere import Hypersphere


def test_lloyd_on_quantum_graph():
    """Test Lloyd's algorithm on a quantum graph."""
    random.seed(42)
    np.random.seed(42)
    graph = generate_sbm(sizes=[20, 20], p=[[0.8, 0.1], [0.1, 0.8]])
    points = graph.sample_points(100, strategy=UniformNodeSampling(random_state=42))

    lloyd = Lloyd(
        points,
        k=2,
        update_strategy=MostFrequentNodeUpdate(random_state=42),
        random_state=42,
    )
    centers = lloyd.run(initialization_strategy=RandomInit())

    # For now, just check that the algorithm runs and returns the correct number of centers.
    # A more robust test would check for convergence or energy decrease.
    assert len(centers) == 2


def test_lloyd_on_riemannian_manifold():
    """Test Lloyd's algorithm on a Riemannian manifold."""
    sphere = Hypersphere(dim=2)
    space = RiemannianManifold(sphere)
    points = space.sample_points(100, strategy=UniformManifoldSampling(random_state=42))

    lloyd = Lloyd(points, k=2, update_strategy=KarcherFrechetMean(), random_state=42)
    centers = lloyd.run(initialization_strategy=RandomInit())

    assert len(centers) == 2
    assert all(isinstance(c, RiemannianCenter) for c in centers)


def test_lloyd_empty_points_raises():
    """Test that Lloyd raises ValueError with empty points list."""
    with pytest.raises(ValueError, match="Input points list cannot be empty."):
        Lloyd([], k=2, update_strategy=MostFrequentNodeUpdate())


def test_lloyd_invalid_k_raises():
    """Test that Lloyd raises ValueError with invalid k."""
    graph = generate_sbm(sizes=[10], p=[[1.0]])
    points = graph.sample_points(10, strategy=UniformNodeSampling())
    with pytest.raises(ValueError, match="Number of clusters k must be positive."):
        Lloyd(points, k=0, update_strategy=MostFrequentNodeUpdate())


def test_lloyd_reseeds_empty_clusters():
    """k never silently shrinks: an empty cluster is reseeded, not dropped.

    Ten points on two far-apart nodes with k=4 force at least two empty
    clusters at every assignment step.
    """
    graph = generate_sbm(sizes=[10, 10], p=[[0.9, 0.1], [0.1, 0.9]])
    nodes = list(graph.nodes())
    points = graph.sample_points(10, strategy=UniformNodeSampling(random_state=0))
    # Pile every point onto two nodes only
    for i, point in enumerate(points):
        target = nodes[0] if i % 2 == 0 else nodes[-1]
        neighbor = next(graph.neighbors(target))
        point._edge = (target, neighbor)
        point.position = 0.0

    lloyd = Lloyd(
        points,
        k=4,
        update_strategy=MostFrequentNodeUpdate(random_state=0),
        random_state=0,
    )
    centers = lloyd.run(initialization_strategy=RandomInit(), max_iterations=5)

    assert len(centers) == 4


def test_simultaneously_empty_clusters_get_distinct_reseeds():
    """Two clusters empty at the same iteration must not share one reseed.

    Regression (2026-07-09 review): every empty cluster was reseeded on the
    point farthest from the *pre-iteration* centers, so simultaneous empties
    all landed on the same point and k stayed shrunk.
    """
    from kmeanssa_ng import QuantumGraph, QGPoint
    from kmeanssa_ng.core.strategies.initialization import InitializationStrategy

    graph = QuantumGraph()
    for v in range(4):
        graph.add_edge(v, v + 1, length=1.0)
    graph.precomputing()

    # 3 points at node 0, 2 at node 2, 2 at node 4.
    points = (
        [QGPoint(graph, (0, 1), 0.0)] * 3
        + [QGPoint(graph, (2, 3), 0.0)] * 2
        + [QGPoint(graph, (4, 3), 0.0)] * 2
    )

    class AllAtNode2(InitializationStrategy):
        """Degenerate start: every center on node 2 -> clusters 1 and 2 empty."""

        def initialize_centers(self, algo):
            return [algo.space.node_as_center(2) for _ in range(algo.k)]

    lloyd = Lloyd(points, k=3, update_strategy=MostFrequentNodeUpdate(random_state=0))
    centers = lloyd.run(initialization_strategy=AllAtNode2(), max_iterations=1)

    assert sorted(c.closest_node() for c in centers) == [0, 2, 4]
