"""Tests for Lloyd's algorithm."""

import pytest

from kmeanssa_ng import (
    generate_sbm,
    Lloyd,
    MostFrequentNodeUpdate,
    RiemannianManifold,
    RiemannianCenter,
    FrechetMeanUpdate,
)
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.riemannian_manifold.sampling import UniformManifoldSampling
from kmeanssa_ng.core.strategies.initialization import RandomInit
from geomstats.geometry.hypersphere import Hypersphere


def test_lloyd_on_quantum_graph():
    """Test Lloyd's algorithm on a quantum graph."""
    graph = generate_sbm(sizes=[20, 20], p=[[0.8, 0.1], [0.1, 0.8]])
    points = graph.sample_points(100, strategy=UniformNodeSampling())

    lloyd = Lloyd(points, k=2, update_strategy=MostFrequentNodeUpdate())
    centers = lloyd.run(initialization_strategy=RandomInit())

    # For now, just check that the algorithm runs and returns the correct number of centers.
    # A more robust test would check for convergence or energy decrease.
    assert len(centers) == 2

def test_lloyd_on_riemannian_manifold():
    """Test Lloyd's algorithm on a Riemannian manifold."""
    sphere = Hypersphere(dim=2)
    space = RiemannianManifold(sphere)
    points = space.sample_points(100, strategy=UniformManifoldSampling())

    lloyd = Lloyd(points, k=2, update_strategy=FrechetMeanUpdate())
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
