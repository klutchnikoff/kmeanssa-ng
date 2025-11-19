"""Tests for update strategies."""

import pytest

import numpy as np
from kmeanssa_ng import (
    generate_simple_graph,
    MostFrequentNodeUpdate,
    QGPoint,
    RiemannianManifold,
    FrechetMeanUpdate,
    RiemannianPoint,
)
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

class TestFrechetMeanUpdate:
    def test_update_with_empty_points(self):
        """Test that update with empty points list returns None."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        strategy = FrechetMeanUpdate()
        center = strategy.update([], space)
        assert center is None

    def test_update_with_wrong_point_type(self):
        """Test that update with wrong point type raises TypeError."""
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        graph = generate_simple_graph()
        graph.add_edge(0, 1, length=1.0)
        points = [QGPoint(graph, (0, 1), 0.5)]
        strategy = FrechetMeanUpdate()
        with pytest.raises(TypeError):
            strategy.update(points, space)
