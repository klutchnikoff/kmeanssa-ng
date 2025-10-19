"""Test quantum graph functionality."""

import pytest

from kmeanssa_ng import QGCenter, QGPoint, QuantumGraph, generate_sbm, generate_simple_graph


class TestQuantumGraph:
    """Tests for QuantumGraph class."""

    def test_create_simple_graph(self):
        """Test creating a simple quantum graph."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.add_edge(0, 2, length=3.0)

        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3

    def test_precomputing(self):
        """Test precomputing pairwise distances."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)

        graph.precomputing()
        assert graph._pairwise_nodes_distance is not None

    def test_distance_between_nodes(self):
        """Test distance computation between nodes."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.precomputing()

        dist = graph.distance_between_nodes(0, 2)
        assert dist == 3.0

    def test_get_edge_length(self):
        """Test getting edge length."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.5)

        length = graph.get_edge_length(0, 1)
        assert length == 2.5


class TestQGPoint:
    """Tests for QGPoint class."""

    def test_create_point(self):
        """Test creating a point on a quantum graph."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        assert point.edge == (0, 1)
        assert point.position == 0.5
        assert point.space == graph

    def test_closest_node(self):
        """Test finding closest node to a point."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        # Point closer to node 0
        point1 = QGPoint(graph, edge=(0, 1), position=0.3)
        assert point1._closest_node() == 0

        # Point closer to node 1
        point2 = QGPoint(graph, edge=(0, 1), position=0.7)
        assert point2._closest_node() == 1

    def test_reverse(self):
        """Test reversing edge orientation."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.3)
        point.reverse()

        assert point.edge == (1, 0)
        assert abs(point.position - 0.7) < 1e-10


class TestQGCenter:
    """Tests for QGCenter class."""

    def test_create_center(self):
        """Test creating a center from a point."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        assert center.edge == (0, 1)
        assert center.position == 0.5

    def test_brownian_motion(self):
        """Test Brownian motion (just check it doesn't crash)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        initial_edge = center.edge
        center.brownian_motion(0.01)  # Small time step

        # Position should have changed (probabilistically)
        # Just check it doesn't crash for now
        assert center.edge is not None

    def test_drift(self):
        """Test drift toward target point."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)

        center = QGCenter(center_point)
        initial_pos = center.position

        # Drift halfway toward target
        center.drift(target_point, 0.5)

        # Should have moved closer to target
        assert center.position > initial_pos


class TestGenerators:
    """Tests for graph generators."""

    def test_generate_simple_graph(self):
        """Test simple graph generator."""
        graph = generate_simple_graph(n_a=3, n_aa=2, bridge_length=2.0)

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        assert graph._pairwise_nodes_distance is not None  # Should be precomputed

    def test_generate_sbm(self):
        """Test SBM generator."""
        graph = generate_sbm(sizes=[10, 10], p=[[0.7, 0.1], [0.1, 0.7]])

        assert graph.number_of_nodes() == 20
        assert graph._pairwise_nodes_distance is not None

    def test_sample_points(self):
        """Test sampling points from a graph."""
        graph = generate_simple_graph(n_a=3)

        points = graph.sample_points(10)
        assert len(points) == 10
        assert all(isinstance(p, QGPoint) for p in points)

    def test_sample_centers(self):
        """Test sampling centers from a graph."""
        graph = generate_simple_graph(n_a=3)

        centers = graph.sample_centers(3)
        assert len(centers) == 3
        assert all(isinstance(c, QGCenter) for c in centers)

    def test_sample_kpp_centers(self):
        """Test k-means++ center initialization."""
        graph = generate_simple_graph(n_a=5)

        centers = graph.sample_kpp_centers(3)
        assert len(centers) == 3
        assert all(isinstance(c, QGCenter) for c in centers)


class TestDistance:
    """Tests for distance computation."""

    def test_distance_same_edge(self):
        """Test distance between points on the same edge."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        p1 = QGPoint(graph, edge=(0, 1), position=0.2)
        p2 = QGPoint(graph, edge=(0, 1), position=0.8)

        dist = graph.distance(p1, p2)
        assert abs(dist - 0.6) < 1e-10

    def test_distance_different_edges(self):
        """Test distance between points on different edges."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        graph.precomputing()

        p1 = QGPoint(graph, edge=(0, 1), position=0.5)
        p2 = QGPoint(graph, edge=(1, 2), position=0.5)

        dist = graph.distance(p1, p2)
        # Should be (0.5 from p1 to node 1) + (0.5 from node 1 to p2) = 1.0
        assert abs(dist - 1.0) < 1e-10
