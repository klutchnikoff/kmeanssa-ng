"""Test quantum graph functionality."""

import networkx as nx
import networkx as nx
import numpy as np
import pytest

from kmeanssa_ng import (
    QGCenter,
    QGPoint,
    QuantumGraph,
    generate_sbm,
    generate_simple_graph,
    generate_random_sbm,
    as_quantum_graph,
    complete_quantum_graph,
)


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

    def test_calculate_energy_with_no_observations(self):
        """Test energy calculation with how='obs' and no observations."""
        # Create a minimal graph that has no 'nb_obs' attributes set
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        centers = [QGCenter(QGPoint(graph, (0, 1), 0.5))]

        # When how="obs" and no nodes have `nb_obs`, total_obs should be 0
        energy = graph.calculate_energy_graph(centers, how="obs")
        assert energy == 0.0

    def test_batch_distances_special_cases(self):
        """Test batch_distances for same-edge and reversed-edge cases."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.precomputing()

        # Case 1: Center and target on the same edge
        c1 = QGCenter(QGPoint(graph, (0, 1), 0.2))
        p1 = QGPoint(graph, (0, 1), 0.8)
        dist1 = graph.batch_distances_from_centers([c1], p1)
        assert np.isclose(dist1[0], 0.6)

        # Case 2: Center and target on reversed edge
        c2 = QGCenter(QGPoint(graph, (0, 1), 0.2))
        p2 = QGPoint(graph, (1, 0), 0.2)  # Same as (0, 1) at 0.8
        dist2 = graph.batch_distances_from_centers([c2], p2)
        assert np.isclose(dist2[0], 0.6)


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

    def test_create_point_with_none_graph_raises(self):
        """Test that creating a point with None graph raises ValueError."""
        with pytest.raises(ValueError, match="quantum_graph cannot be None"):
            QGPoint(None, edge=(0, 1), position=0.5)

    def test_create_point_with_invalid_edge_type_raises(self):
        """Test that invalid edge type raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="must be a tuple of two nodes"):
            QGPoint(graph, edge=[0, 1], position=0.5)

    def test_create_point_with_nonexistent_edge_raises(self):
        """Test that point on non-existent edge raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="does not exist in the graph"):
            QGPoint(graph, edge=(2, 3), position=0.5)

    def test_create_point_with_negative_position_raises(self):
        """Test that negative position raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="must be non-negative"):
            QGPoint(graph, edge=(0, 1), position=-0.5)

    def test_create_point_with_position_exceeding_length_raises(self):
        """Test that position exceeding edge length raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="exceeds edge length"):
            QGPoint(graph, edge=(0, 1), position=1.5)

    def test_create_point_with_non_numeric_position_raises(self):
        """Test that non-numeric position raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="must be a number"):
            QGPoint(graph, edge=(0, 1), position="invalid")

    def test_create_point_at_edge_boundaries(self):
        """Test creating points at edge boundaries (0 and edge_length)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)

        # Position at start
        point1 = QGPoint(graph, edge=(0, 1), position=0.0)
        assert point1.position == 0.0

        # Position at end
        point2 = QGPoint(graph, edge=(0, 1), position=2.0)
        assert point2.position == 2.0

    def test_set_edge_with_nonexistent_edge_raises(self):
        """Test that setting a non-existent edge raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)

        point = QGPoint(graph, edge=(0, 1), position=1.5)

        # Try to change to non-existent edge - should fail
        with pytest.raises(ValueError, match="does not belong to the graph"):
            point.edge = (5, 6)

    def test_set_edge_succeeds_and_edge_property_checks(self):
        """Test that setting edge and edge property coverage (covers 104-107, 153-154)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)
        graph.add_edge(1, 2, length=3.0)

        point = QGPoint(graph, edge=(0, 1), position=1.5)
        point.edge = (1, 2)  # Should succeed

        assert point.edge == (1, 2)
        # Position is not automatically updated when changing edge
        assert point.position == 1.5

        # Cover __str__ representation path (includes graph name if present)
        graph.name = "TestGraph"
        s = str(point)
        assert "TestGraph" in s

        # Force edge property to pass through "edge not in graph" branch with self-loop
        point.edge = (3, 3)  # Self-loop allowed by property
        assert point.edge == (3, 3)

    def test_edge_property_raises_when_not_in_graph(self):
        """Test edge property raises when edge not in graph (covers line 107)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)

        # Force the edge to be something not in the graph and not a self-loop
        point._edge = (5, 6)  # Set directly to bypass setter validation

        # Now accessing the property should raise
        with pytest.raises(ValueError, match="does not belong to the graph"):
            _ = point.edge


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

    def test_brownian_motion_with_negative_time_raises(self):
        """Test that negative time_to_travel raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        with pytest.raises(ValueError, match="must be non-negative"):
            center.brownian_motion(-0.1)

    def test_brownian_motion_with_non_numeric_time_raises(self):
        """Test that non-numeric time_to_travel raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        with pytest.raises(ValueError, match="must be a number"):
            center.brownian_motion("invalid")

    def test_brownian_motion_with_zero_time_succeeds(self):
        """Test that zero time_to_travel is valid."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        initial_position = center.position
        center.brownian_motion(0.0)  # Should not crash

        # Position might change slightly due to random normal, but should be close
        assert center.position is not None

    def test_drift_with_none_target_raises(self):
        """Test that None target_point raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        with pytest.raises(ValueError, match="target_point cannot be None"):
            center.drift(None, 0.5)

    def test_drift_with_negative_prop_raises(self):
        """Test that negative prop_to_travel raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)
        center = QGCenter(center_point)

        with pytest.raises(ValueError, match="must be in"):
            center.drift(target_point, -0.1)

    def test_drift_with_prop_greater_than_one_raises(self):
        """Test that prop_to_travel > 1 raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)
        center = QGCenter(center_point)

        with pytest.raises(ValueError, match="must be in"):
            center.drift(target_point, 1.5)

    def test_drift_with_non_numeric_prop_raises(self):
        """Test that non-numeric prop_to_travel raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)
        center = QGCenter(center_point)

        with pytest.raises(ValueError, match="must be a number"):
            center.drift(target_point, "invalid")

    def test_drift_with_boundary_values_succeeds(self):
        """Test that prop_to_travel at boundaries (0 and 1) succeed."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)

        # Test with prop = 0 (no movement)
        center1 = QGCenter(center_point)
        initial_pos = center1.position
        center1.drift(target_point, 0.0)
        assert center1.position == initial_pos

        # Test with prop = 1 (full movement)
        center2 = QGCenter(center_point)
        center2.drift(target_point, 1.0)
        assert abs(center2.position - target_point.position) < 1e-10

    def test_find_best_neighbor_same_nodes(self):
        """Test _find_best_neighbor when n1 == n2 (covers line 65)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(0, 2, length=2.0)
        graph.precomputing()

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        # When n1 == n2, should return n1
        result = center._find_best_neighbor(0, 0)
        assert result == 0

    def test_drift_on_same_edge_different_parametrizations(self):
        """Test drift when edges have different parametrizations (covers lines 131-137)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        # Create points on "same" edge but different parametrizations
        center_point = QGPoint(graph, edge=(0, 1), position=0.3)
        target_point = QGPoint(graph, edge=(1, 0), position=0.2)  # Reversed edge

        center = QGCenter(center_point)
        initial_pos = center.position

        # This should trigger the "different parametrization" branch
        center.drift(target_point, 0.1)

        # Position should have changed
        assert center.position != initial_pos

        # Test the other condition: center.position > target.position (line 135)
        center_point2 = QGPoint(graph, edge=(0, 1), position=0.7)
        target_point2 = QGPoint(graph, edge=(1, 0), position=0.2)  # Reversed edge

        center2 = QGCenter(center_point2)
        initial_pos2 = center2.position

        # This should trigger line 135
        center2.drift(target_point2, 0.1)
        assert center2.position != initial_pos2

        # Test line 131: center.position > target.position on same orientation
        center_point3 = QGPoint(graph, edge=(0, 1), position=0.8)
        target_point3 = QGPoint(graph, edge=(0, 1), position=0.2)

        center3 = QGCenter(center_point3)
        initial_pos3 = center3.position

        # This should trigger line 131
        center3.drift(target_point3, 0.1)
        assert center3.position < initial_pos3  # Should move backward


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

    @pytest.mark.parametrize(
        "invalid_graph, expected_error_match",
        [
            (None, "`graph` must be a networkx.Graph object."),
            ("not a graph", "`graph` must be a networkx.Graph object."),
            (123, "`graph` must be a networkx.Graph object."),
        ],
    )
    def test_as_quantum_graph_invalid_graph_raises_value_error(
        self, invalid_graph, expected_error_match
    ):
        """Test that invalid 'graph' raises ValueError."""
        with pytest.raises(ValueError, match=expected_error_match):
            as_quantum_graph(invalid_graph)

    @pytest.mark.parametrize(
        "invalid_weight, expected_error_match",
        [
            (0, "`node_weight` must be a positive number."),
            (-1, "`node_weight` must be a positive number."),
            ("invalid", "`node_weight` must be a positive number."),
        ],
    )
    def test_as_quantum_graph_invalid_node_weight_raises_value_error(
        self, invalid_weight, expected_error_match
    ):
        """Test that invalid 'node_weight' raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(ValueError, match=expected_error_match):
            as_quantum_graph(G, node_weight=invalid_weight)

    @pytest.mark.parametrize(
        "invalid_length, expected_error_match",
        [
            (0, "`edge_length` must be a positive number."),
            (-1, "`edge_length` must be a positive number."),
            ("invalid", "`edge_length` must be a positive number."),
        ],
    )
    def test_as_quantum_graph_invalid_edge_length_raises_value_error(
        self, invalid_length, expected_error_match
    ):
        """Test that invalid 'edge_length' raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(ValueError, match=expected_error_match):
            as_quantum_graph(G, edge_length=invalid_length)

    @pytest.mark.parametrize(
        "invalid_weight, expected_error_match",
        [
            (0, "`edge_weight` must be a positive number."),
            (-1, "`edge_weight` must be a positive number."),
            ("invalid", "`edge_weight` must be a positive number."),
        ],
    )
    def test_as_quantum_graph_invalid_edge_weight_raises_value_error(
        self, invalid_weight, expected_error_match
    ):
        """Test that invalid 'edge_weight' raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(ValueError, match=expected_error_match):
            as_quantum_graph(G, edge_weight=invalid_weight)

    @pytest.mark.parametrize(
        "invalid_objects, expected_error_match",
        [
            ([], "`objects` must be a non-empty list."),
            ("not a list", "`objects` must be a non-empty list."),
            (123, "`objects` must be a non-empty list."),
        ],
    )
    def test_complete_quantum_graph_invalid_objects_raises_value_error(
        self, invalid_objects, expected_error_match
    ):
        """Test that invalid 'objects' raise ValueError."""
        with pytest.raises(ValueError, match=expected_error_match):
            complete_quantum_graph(invalid_objects)

    @pytest.mark.parametrize(
        "invalid_similarities, expected_error_match",
        [
            ("not an array", "`similarities` must be a numpy array."),
            (
                np.array([[1, 2]]),
                "`similarities` must be a square matrix of size 2x2.",
            ),  # Not square
            (
                np.array([[-1, 2], [3, 4]]),
                "Elements of `similarities` must be non-negative.",
            ),  # Negative value
        ],
    )
    def test_complete_quantum_graph_invalid_similarities_raises_value_error(
        self, invalid_similarities, expected_error_match
    ):
        """Test that invalid 'similarities' raise ValueError."""
        objects = [1, 2]
        with pytest.raises(ValueError, match=expected_error_match):
            complete_quantum_graph(objects, similarities=invalid_similarities)

    @pytest.mark.parametrize(
        "invalid_labels, expected_error_match",
        [
            ("not a list", "`true_labels` must be a list."),
            (
                [1],
                r"`true_labels` must have the same length as `objects` \(\d+\).",
            ),  # Incorrect length
        ],
    )
    def test_complete_quantum_graph_invalid_true_labels_raises_value_error(
        self, invalid_labels, expected_error_match
    ):
        """Test that invalid 'true_labels' raise ValueError."""
        objects = [1, 2]
        with pytest.raises(ValueError, match=expected_error_match):
            complete_quantum_graph(objects, true_labels=invalid_labels)


class TestGenerateRandomSBM:
    """Tests for generate_random_sbm input validation."""

    def test_generate_random_sbm_default_params(self):
        """Test generate_random_sbm with default parameters."""
        graph = generate_random_sbm()
        assert graph.number_of_nodes() == 100  # 50 + 50
        assert graph.number_of_edges() > 0

    @pytest.mark.parametrize(
        "invalid_sizes, expected_error_match",
        [
            (None, None),  # Default behavior, no error
            ([], "`sizes` must be a non-empty list of positive integers."),
            ([0], "`sizes` must be a non-empty list of positive integers."),
            ([-10], "`sizes` must be a non-empty list of positive integers."),
            ([10, -5], "`sizes` must be a non-empty list of positive integers."),
            ([10.5, 20], "`sizes` must be a non-empty list of positive integers."),
            ("not a list", "`sizes` must be a non-empty list of positive integers."),
            ([10, "invalid"], "`sizes` must be a non-empty list of positive integers."),
        ],
    )
    def test_generate_random_sbm_invalid_sizes_raises_value_error(
        self, invalid_sizes, expected_error_match
    ):
        """Test that invalid 'sizes' raise ValueError."""
        if expected_error_match is None:
            graph = generate_random_sbm(sizes=invalid_sizes)
            assert graph.number_of_nodes() == 100
        else:
            with pytest.raises(ValueError, match=expected_error_match):
                # For empty sizes, provide empty p, weights, lengths to avoid early validation errors
                if invalid_sizes == []:
                    generate_random_sbm(
                        sizes=invalid_sizes, p=[], weights=[], lengths=[]
                    )
                else:
                    generate_random_sbm(sizes=invalid_sizes)

    @pytest.mark.parametrize(
        "invalid_p, expected_error_match",
        [
            (None, None),  # Default behavior, no error
            ([], r"`p` must be a square matrix of size \d+x\d+."),
            ([[0.5]], r"`p` must be a square matrix of size \d+x\d+."),  # Not square
            (
                [[0.5, 0.5]],
                r"`p` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[0.5, 0.5], [0.5]],
                r"`p` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[0.5, 0.5], [0.5, 1.5]],
                "Elements of `p` must be floats or integers between 0 and 1.",
            ),  # Value > 1
            (
                [[-0.1, 0.5], [0.5, 0.5]],
                "Elements of `p` must be floats or integers between 0 and 1.",
            ),  # Value < 0
            (
                [[0.5, "invalid"], [0.5, 0.5]],
                "Elements of `p` must be floats or integers between 0 and 1.",
            ),  # Non-numeric
            ("not a list", r"`p` must be a square matrix of size \d+x\d+."),
        ],
    )
    def test_generate_random_sbm_invalid_p_raises_value_error(
        self, invalid_p, expected_error_match
    ):
        """Test that invalid 'p' raises ValueError."""
        if expected_error_match is None:
            graph = generate_random_sbm(p=invalid_p)
            assert graph.number_of_nodes() == 100
        else:
            with pytest.raises(ValueError, match=expected_error_match):
                generate_random_sbm(sizes=[50, 50], p=invalid_p)

    @pytest.mark.parametrize(
        "invalid_weights, expected_error_match",
        [
            (None, None),  # Default behavior, no error
            ([], r"`weights` must be a list of size \d+."),
            ([0], r"`weights` must be a list of size \d+."),
            ([-1], r"`weights` must be a list of size \d+."),
            ([1, -0.5], "Elements of `weights` must be positive numbers."),
            ([1, "invalid"], "Elements of `weights` must be positive numbers."),
            ("not a list", r"`weights` must be a list of size \d+."),
        ],
    )
    def test_generate_random_sbm_invalid_weights_raises_value_error(
        self, invalid_weights, expected_error_match
    ):
        """Test that invalid 'weights' raise ValueError."""
        if expected_error_match is None:
            graph = generate_random_sbm(weights=invalid_weights)
            assert graph.number_of_nodes() == 100
        else:
            with pytest.raises(ValueError, match=expected_error_match):
                generate_random_sbm(sizes=[50, 50], weights=invalid_weights)

    @pytest.mark.parametrize(
        "invalid_lengths, expected_error_match",
        [
            (None, None),  # Default behavior, no error
            ([], r"`lengths` must be a square matrix of size \d+x\d+."),
            (
                [[1]],
                r"`lengths` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[1, 2]],
                r"`lengths` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[1, 2], [3]],
                r"`lengths` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[1, -2], [3, 4]],
                "Elements of `lengths` must be positive numbers.",
            ),  # Negative value
            (
                [[1, "invalid"], [3, 4]],
                "Elements of `lengths` must be positive numbers.",
            ),  # Non-numeric
            ("not a list", r"`lengths` must be a square matrix of size \d+x\d+."),
        ],
    )
    def test_generate_random_sbm_invalid_lengths_raises_value_error(
        self, invalid_lengths, expected_error_match
    ):
        """Test that invalid 'lengths' raise ValueError."""
        if expected_error_match is None:
            graph = generate_random_sbm(lengths=invalid_lengths)
            assert graph.number_of_nodes() == 100
        else:
            with pytest.raises(ValueError, match=expected_error_match):
                generate_random_sbm(sizes=[50, 50], lengths=invalid_lengths)


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


class TestValidation:
    """Tests for input validation in QuantumGraph."""

    def test_add_edge_without_length_raises(self):
        """Test that adding an edge without length raises ValueError."""
        graph = QuantumGraph()
        with pytest.raises(ValueError, match="must have a 'length' attribute"):
            graph.add_edge(0, 1)

    def test_add_edge_with_zero_length_raises(self):
        """Test that zero length raises ValueError."""
        graph = QuantumGraph()
        with pytest.raises(ValueError, match="must be positive"):
            graph.add_edge(0, 1, length=0)

    def test_add_edge_with_negative_length_raises(self):
        """Test that negative length raises ValueError."""
        graph = QuantumGraph()
        with pytest.raises(ValueError, match="must be positive"):
            graph.add_edge(0, 1, length=-1.5)

    def test_add_edge_with_non_numeric_length_raises(self):
        """Test that non-numeric length raises ValueError."""
        graph = QuantumGraph()
        with pytest.raises(ValueError, match="must be a number"):
            graph.add_edge(0, 1, length="invalid")

    def test_validate_edge_lengths_with_missing_length(self):
        """Test validation fails when edge is missing length attribute."""
        graph = QuantumGraph()
        # Bypass validation by using parent class method
        nx.Graph.add_edge(graph, 0, 1, weight=1.0)  # No length attribute

        with pytest.raises(ValueError, match="missing 'length' attribute"):
            graph.validate_edge_lengths()

    def test_validate_edge_lengths_with_invalid_length(self):
        """Test validation fails when edge has invalid length."""
        graph = QuantumGraph()
        # Bypass validation by using parent class method
        nx.Graph.add_edge(graph, 0, 1, length=-2.0)

        with pytest.raises(ValueError, match="invalid length.*must be positive"):
            graph.validate_edge_lengths()

    def test_validate_edge_lengths_success(self):
        """Test validation succeeds with valid edges."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)

        # Should not raise
        graph.validate_edge_lengths()

    def test_precomputing_disconnected_graph_raises(self):
        """Test that precomputing on disconnected graph raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(2, 3, length=1.0)  # Separate component

        with pytest.raises(
            ValueError, match="must be connected.*2 connected components"
        ):
            graph.precomputing()

    def test_precomputing_invalid_edges_raises(self):
        """Test that precomputing validates edge lengths."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        # Manually corrupt an edge length
        graph[0][1]["length"] = -1.0

        with pytest.raises(ValueError, match="invalid length"):
            graph.precomputing()

    def test_precomputing_connected_graph_success(self):
        """Test that precomputing succeeds on valid connected graph."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.add_edge(0, 2, length=3.0)

        # Should not raise
        graph.precomputing()
        assert graph._pairwise_nodes_distance is not None
