"""Tests for approximating a manifold by an epsilon-net quantum graph."""

import networkx as nx
import numpy as np
import pytest

from kmeanssa_ng import (
    create_sphere,
    SimulatedAnnealing,
    KMeansPlusPlus,
    MinimizeEnergy,
)
from kmeanssa_ng.quantum_graph import QuantumGraph
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.riemannian_manifold import (
    FibonacciNet,
    RepulsionNet,
    approximate_geodesic_space,
    build_epsilon_net_graph,
    estimate_covering_radius,
)


class TestCoveringRadius:
    def test_positive_and_reproducible(self):
        sphere = create_sphere(2)
        points = FibonacciNet().build(sphere, 300)
        a = estimate_covering_radius(sphere, points, n_test=4000, random_state=0)
        b = estimate_covering_radius(sphere, points, n_test=4000, random_state=0)
        assert a > 0
        assert a == b

    def test_shrinks_with_more_points(self):
        sphere = create_sphere(2)
        coarse = estimate_covering_radius(
            sphere, FibonacciNet().build(sphere, 100), n_test=4000, random_state=0
        )
        fine = estimate_covering_radius(
            sphere, FibonacciNet().build(sphere, 500), n_test=4000, random_state=0
        )
        assert fine < coarse


class TestBuildEpsilonNetGraph:
    def _graph(self, n=300):
        sphere = create_sphere(2)
        points = FibonacciNet().build(sphere, n)
        return sphere, points, build_epsilon_net_graph(sphere, points, random_state=0)

    def test_is_connected_quantum_graph(self):
        _, points, qg = self._graph()
        assert isinstance(qg, QuantumGraph)
        assert qg.number_of_nodes() == len(points)
        assert nx.is_connected(qg)

    def test_edges_have_geometric_attributes(self):
        sphere = create_sphere(2)
        points = FibonacciNet().build(sphere, 300)
        ell = 0.4
        qg = build_epsilon_net_graph(sphere, points, ell=ell)
        for _, _, data in qg.edges(data=True):
            assert data["weight"] == 1.0
            assert "distribution" in data
            assert data["length"] <= ell + 1e-9
        assert all(w == 1.0 for w in nx.get_node_attributes(qg, "weight").values())

    def test_reproducible_structure(self):
        sphere = create_sphere(2)
        points = FibonacciNet().build(sphere, 300)
        e1 = sorted(build_epsilon_net_graph(sphere, points, random_state=0).edges())
        e2 = sorted(build_epsilon_net_graph(sphere, points, random_state=0).edges())
        assert e1 == e2

    def test_disconnected_raises(self):
        sphere = create_sphere(2)
        points = FibonacciNet().build(sphere, 300)
        with pytest.raises(ValueError, match="disconnected"):
            build_epsilon_net_graph(sphere, points, ell=0.01)


class TestApproximateGeodesicSpace:
    def test_returns_connected_graph(self):
        sphere = create_sphere(2)
        qg = approximate_geodesic_space(
            sphere, 200, net=RepulsionNet(n_iter=60, random_state=1), random_state=1
        )
        assert isinstance(qg, QuantumGraph)
        assert qg.number_of_nodes() == 200
        assert nx.is_connected(qg)

    def test_supports_clustering(self):
        sphere = create_sphere(2)
        qg = approximate_geodesic_space(
            sphere, 200, net=RepulsionNet(n_iter=60, random_state=1), random_state=1
        )
        obs = qg.sample_points(120, strategy=UniformNodeSampling(random_state=2))
        sa = SimulatedAnnealing(
            obs,
            k=3,
            lambda0=1.0,
            beta0=0.5,
            step_size=0.05,
            energy_mode="obs",
            random_state=3,
        )
        centers = sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1)
        assert len(centers) == 3
