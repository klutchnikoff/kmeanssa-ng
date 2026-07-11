"""Tests for update strategies."""

import pytest

import numpy as np
from kmeanssa_ng import (
    QGPoint,
    RiemannianManifold,
    KarcherFrechetMean,
    RiemannianPoint,
    QuantumGraph,
    MostFrequentNodeUpdate,
    MinimizeEnergyNodeUpdate,
    SimulatedAnnealingFrechetMean,
)
from kmeanssa_ng.quantum_graph.generators import generate_simple_graph
from geomstats.geometry.hypersphere import Hypersphere


class TestMostFrequentNodeUpdate:
    def test_update_with_empty_points(self):
        """Test that update with empty points list returns None."""
        graph = generate_simple_graph()
        strategy = MostFrequentNodeUpdate()
        center = strategy.update([], graph)
        assert center is None

    def test_update_with_wrong_point_type(self):
        """Test that update with wrong point type raises TypeError."""
        graph = generate_simple_graph()
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        points = [RiemannianPoint(space, np.array([1, 0, 0]))]
        strategy = MostFrequentNodeUpdate()
        with pytest.raises(TypeError):
            strategy.update(points, graph)


class TestKarcherFrechetMean:
    def test_update_with_empty_points(self):
        """Test that update with empty points list returns None."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        strategy = KarcherFrechetMean()
        center = strategy.update([], space)
        assert center is None

    def test_update_with_wrong_point_type(self):
        """Test that update with wrong point type raises TypeError."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        graph = generate_simple_graph()
        graph.add_edge("A0", "A1", length=1.0)
        points = [QGPoint(graph, ("A0", "A1"), 0.5)]
        strategy = KarcherFrechetMean()
        with pytest.raises(TypeError):
            strategy.update(points, space)


class TestMinimizeEnergyNodeUpdate:
    def test_update_with_empty_points(self):
        """Test that update with empty points list returns None."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()
        strategy = MinimizeEnergyNodeUpdate()
        center = strategy.update([], graph)
        assert center is None

    def test_update_without_precomputation(self):
        """Test that update raises ValueError if distances are not precomputed."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        points = [QGPoint(graph, (0, 1), 0.1)]
        strategy = MinimizeEnergyNodeUpdate()
        with pytest.raises(ValueError, match="requires precomputed distances"):
            strategy.update(points, graph)

    def test_update_logic(self):
        """Test the core logic of the MinimizeEnergyNodeUpdate strategy."""
        # 1. Setup graph and points
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        graph.add_edge(2, 3, length=1.0)
        graph.add_edge(3, 4, length=1.0)
        graph.precomputing()

        # 3 points near node 1, 1 point near node 4
        points = [
            QGPoint(graph, (0, 1), 0.9),  # closest_node is 1
            QGPoint(graph, (1, 2), 0.1),  # closest_node is 1
            QGPoint(graph, (1, 2), 0.2),  # closest_node is 1
            QGPoint(graph, (3, 4), 0.9),  # closest_node is 4
        ]

        # Expected nearest nodes: [1, 1, 1, 4]
        # Energy calculation:
        # Candidate 0: 3*d(0,1)^2 + 1*d(0,4)^2 = 3*1 + 1*16 = 19
        # Candidate 1: 3*d(1,1)^2 + 1*d(1,4)^2 = 3*0 + 1*9 = 9
        # Candidate 2: 3*d(2,1)^2 + 1*d(2,4)^2 = 3*1 + 1*4 = 7
        # Candidate 3: 3*d(3,1)^2 + 1*d(3,4)^2 = 3*4 + 1*1 = 13
        # Candidate 4: 3*d(4,1)^2 + 1*d(4,4)^2 = 3*9 + 1*0 = 27
        # Minimum energy is for node 2.

        # 2. Run strategy
        strategy = MinimizeEnergyNodeUpdate()
        new_center = strategy.update(points, graph)

        # 3. Assert result
        assert new_center is not None
        # The center is a point at position 0 on an arbitrary edge starting at the best node
        assert new_center.closest_node() == 2


class TestSimulatedAnnealingFrechetMean:
    def test_update_on_sphere(self):
        """Test that the SA-based Fréchet mean strategy finds a reasonable center."""
        # 1. Setup space and points
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)

        # Create a cluster of points close to the north pole ([0, 0, 1]);
        # points must lie exactly on the sphere (membership is validated)
        def unit(v):
            v = np.asarray(v, dtype=float)
            return v / np.linalg.norm(v)

        points = [
            RiemannianPoint(space, unit([0.1, 0, 0.995])),
            RiemannianPoint(space, unit([-0.1, 0, 0.995])),
            RiemannianPoint(space, unit([0, 0.1, 0.995])),
            RiemannianPoint(space, unit([0, -0.1, 0.995])),
            RiemannianPoint(space, np.array([0, 0, 1.0])),
        ]

        # The Fréchet mean should be very close to the north pole
        expected_mean = np.array([0, 0, 1.0])

        # 2. Run strategy
        # Use a small number of samples for a fast test
        strategy = SimulatedAnnealingFrechetMean(
            n_samples=20, lambda0=0.5, beta0=3.0, random_state=0
        )
        new_center = strategy.update(points, space)

        # 3. Assert result
        assert new_center is not None
        # Check that the result is close to the expected mean
        distance_to_expected = space.distance(
            new_center, RiemannianPoint(space, expected_mean)
        )
        assert distance_to_expected < 0.15


class TestFrechetMeanSelectionEnergy:
    """The inner SA must select on the *cluster's* empirical energy.

    Regression (2026-07-09 review): on a graph, the selection went through
    the old "obs"/uniform energy — the mean squared distance to *all* graph
    nodes — so the returned "Fréchet mean" was biased toward the global
    center of the graph (a 9-node path returned node 2 instead of node 1 for
    the cluster {0, 1, 2}).
    """

    def test_returns_the_cluster_mean_not_the_graph_center(self):
        graph = QuantumGraph()
        for v in range(8):
            graph.add_edge(v, v + 1, length=1.0)
        graph.precomputing()

        cluster = [
            QGPoint(graph, (v, v + 1), 0.0) for v in (0, 1, 2)
        ]  # true Fréchet mean: node 1 (empirical energy 2/3; node 2 gives 5/3)

        strategy = SimulatedAnnealingFrechetMean(
            random_state=np.random.default_rng(0), n_samples=40
        )
        for _ in range(3):  # three clusters' worth of calls, all must stay local
            center = strategy.update(cluster, graph)
            energy = graph.calculate_energy(
                [center], how="empirical", observations=cluster
            )
            assert energy < 1.0  # node 1 -> 0.667, the biased node 2 -> 1.667

    def test_rejects_a_foreign_energy_mode(self):
        graph = generate_simple_graph()
        points = [QGPoint(graph, ("A0", "B0"), 0.0)]
        strategy = SimulatedAnnealingFrechetMean(random_state=0, energy_mode="uniform")
        with pytest.raises(ValueError, match="empirical"):
            strategy.update(points, graph)

    def test_int_seed_decorrelates_successive_updates(self):
        """With an int seed, consecutive update() calls must differ (RNG-1).

        The int seed used to rebuild default_rng(seed) inside every update(),
        so each cluster and each Lloyd iteration reused an identical stream
        (same resample indices, same inner SA trajectory). One owned Generator
        now advances across calls.
        """
        space = RiemannianManifold(Hypersphere(dim=2))
        rng = np.random.default_rng(1)
        cloud = []
        for _ in range(15):
            v = np.array([1.0, 0.0, 0.0]) + 0.05 * rng.standard_normal(3)
            cloud.append(RiemannianPoint(space, v / np.linalg.norm(v)))

        strategy = SimulatedAnnealingFrechetMean(random_state=42, n_samples=30)
        first = strategy.update(cloud, space).coordinates
        second = strategy.update(cloud, space).coordinates
        assert not np.allclose(first, second)

        # Two instances with the same int seed reproduce the first update.
        again = SimulatedAnnealingFrechetMean(random_state=42, n_samples=30)
        assert np.allclose(again.update(cloud, space).coordinates, first)
