"""Additional tests for QuantumGraph space.py to improve coverage."""

import networkx as nx
import pytest
from unittest.mock import patch

from kmeanssa_ng import QuantumGraph, QGPoint, QGCenter


class TestQuantumGraphDiameter:
    """Tests for diameter property."""

    def test_diameter_calculation(self):
        """Test diameter calculation and caching."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)
        graph.add_edge(1, 2, length=3.0)
        graph.add_edge(0, 2, length=1.0)  # Shorter path
        graph.precomputing()

        # First call should calculate
        diameter = graph.diameter
        assert (
            diameter == 3.0
        )  # 0 -> 1 -> 2 (2 + 3 = 5) vs 0 -> 2 -> 1 (1 + 3 = 4) vs direct paths

        # Second call should use cached value
        assert graph._diameter == 3.0
        diameter2 = graph.diameter
        assert diameter2 == diameter

    def test_diameter_empty_graph(self):
        """Test diameter on empty graph."""
        graph = QuantumGraph()
        diameter = graph.diameter
        assert diameter == 0.0


class TestValidateEdgeLengths:
    """Tests for edge length validation."""

    def test_validate_edge_lengths_invalid_type(self):
        """Test validation with non-numeric edge length."""
        graph = QuantumGraph()
        # Use parent class to bypass validation
        nx.Graph.add_edge(graph, 0, 1, length="invalid")

        with pytest.raises(ValueError, match="length must be a number"):
            graph.validate_edge_lengths()

    def test_validate_edge_lengths_zero_length(self):
        """Test validation with zero edge length."""
        graph = QuantumGraph()
        # Use parent class to bypass validation
        nx.Graph.add_edge(graph, 0, 1, length=0.0)

        with pytest.raises(ValueError, match="invalid length.*must be positive"):
            graph.validate_edge_lengths()


class TestPrecomputingConnectivity:
    """Tests for precomputing connectivity checks."""

    def test_precomputing_disconnected_graph(self):
        """Test that precomputing raises error for disconnected graph."""
        graph = QuantumGraph()
        # Create two disconnected components
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(2, 3, length=1.0)

        with pytest.raises(
            ValueError, match="Graph must be connected.*Found 2 connected components"
        ):
            graph.precomputing()

    def test_precomputing_single_node(self):
        """Test precomputing with single node graph."""
        graph = QuantumGraph()
        graph.add_node(0)

        # Should work fine - single node is connected
        graph.precomputing()
        assert graph._pairwise_nodes_distance is not None


class TestQuantumPathEdgeCases:
    """Tests for quantum_path edge cases."""

    def test_quantum_path_same_edge_direct(self):
        """Test quantum path between points on same edge."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=5.0)
        graph.precomputing()

        p1 = QGPoint(graph, edge=(0, 1), position=1.0)
        p2 = QGPoint(graph, edge=(0, 1), position=3.0)

        result = graph.quantum_path(p1, p2)
        assert result["distance"] == 2.0  # |3.0 - 1.0|
        assert result["path"] is None


class TestNodePosition:
    """Tests for node_position property."""

    def test_node_position_calculation(self):
        """Test node position calculation for visualization."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.add_edge(0, 2, length=1.5)

        # First call should calculate positions
        positions = graph.node_position
        assert isinstance(positions, dict)
        assert len(positions) == 3
        assert all(node in positions for node in [0, 1, 2])

        # Each position should be (x, y) tuple
        for pos in positions.values():
            assert len(pos) == 2
            assert all(isinstance(coord, (int, float)) for coord in pos)

        # Second call should use cached value
        positions2 = graph.node_position
        assert positions2 is positions  # Same object

    def test_node_position_caching(self):
        """Test that node positions are properly cached."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        # Check caching works
        assert graph._node_position is None
        pos1 = graph.node_position
        assert graph._node_position is not None
        pos2 = graph.node_position
        assert pos1 is pos2


class TestSamplingEdgeCases:
    """Tests for sampling edge cases and error conditions."""

    def test_nodes_as_points(self):
        """Test converting nodes to points."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        points = graph.nodes_as_points()
        assert len(points) == 3
        assert all(isinstance(p, QGPoint) for p in points)
        assert all(p.position == 0.0 for p in points)  # All at nodes

    def test_node_as_center(self):
        """Test creating center at specific node."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        center = graph.node_as_center(1)
        assert isinstance(center, QGCenter)
        assert center.position == 0.0
        assert center.edge[0] == 1 or center.edge[1] == 1  # One end is node 1

    def test_compute_clusters(self):
        """Test cluster assignment."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        graph.precomputing()

        # Create centers at nodes 0 and 2
        center1 = graph.node_as_center(0)
        center2 = graph.node_as_center(2)
        centers = [center1, center2]

        graph.compute_clusters(centers)

        # Check that cluster attributes were set
        cluster_attrs = nx.get_node_attributes(graph, "cluster")
        assert len(cluster_attrs) == 3
        assert all(cluster in [0, 1] for cluster in cluster_attrs.values())

    def test_calculate_energy_uniform(self):
        """Test energy calculation with uniform weighting."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        graph.precomputing()

        center = graph.node_as_center(1)  # Center at middle node
        energy = graph.calculate_energy([center], how="uniform")

        assert isinstance(energy, float)
        assert energy >= 0

    def test_calculate_energy_obs_no_observations(self):
        """Test energy calculation with obs weighting but no observations."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center = graph.node_as_center(0)
        energy = graph.calculate_energy([center], how="obs")

        assert energy == 0.0  # No observations

    def test_calculate_energy_obs_with_observations(self):
        """Test energy calculation with obs weighting and observations."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        # Set some observations
        nx.set_node_attributes(graph, {0: {"nb_obs": 5}, 1: {"nb_obs": 3}})

        center = graph.node_as_center(1)
        energy = graph.calculate_energy([center], how="obs")

        assert isinstance(energy, float)
        assert energy >= 0

    def test_distance_matrix_no_precomputing(self):
        """Test matrix distance computation without precomputing."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="Must call precomputing\\(\\) first"):
            graph.distance_matrix()

    def test_distance_matrix(self):
        """Test matrix distance computation."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)
        graph.add_edge(1, 2, length=3.0)
        graph.precomputing()

        matrix = graph.distance_matrix()

        assert matrix.shape == (3, 3)
        assert matrix[0, 1] == 2.0
        assert matrix[1, 2] == 3.0
        assert matrix[0, 2] == 5.0  # 2 + 3

        # Check symmetry
        assert matrix[0, 1] == matrix[1, 0]

        # Check diagonal is zero
        assert all(matrix[i, i] == 0 for i in range(3))

    def test_index_to_centers(self):
        """Test converting indices to centers."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        centers = graph.index_to_centers([0, 2])

        assert len(centers) == 2
        assert all(isinstance(c, QGCenter) for c in centers)

        # Check positions are at nodes (position 0)
        assert all(c.position == 0.0 for c in centers)

    def test_light_sample_points(self):
        """Test light sampling of points."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        points = graph.light_sample_points(5)

        assert len(points) == 5
        assert all(isinstance(p, QGPoint) for p in points)
        assert all(p.position == 0.0 for p in points)  # All at nodes


class TestDrawingCoverage:
    """Tests for drawing method coverage."""

    def test_draw_raises_importerror_if_matplotlib_not_installed(self):
        """Test that draw() raises ImportError if matplotlib is not installed."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        # Simulate ImportError for matplotlib
        with patch.dict("sys.modules", {"matplotlib.pyplot": None}):
            with pytest.raises(ImportError, match="Plotting requires matplotlib"):
                graph.draw()

    def test_draw_prints_warning_for_missing_attributes(self, capsys):
        """Test draw() prints warnings to stdout for missing attributes."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        # Mock the plotting call to avoid rendering
        with patch("matplotlib.pyplot.subplots") as mock_subplots:
            mock_ax = mock_subplots.return_value[1]

            # Test warning for missing color_by attribute
            graph.draw(color_by="non_existent", ax=mock_ax)
            captured = capsys.readouterr()
            assert "Warning: Node attribute 'non_existent' not found" in captured.out

            # Test warning for missing nb_obs attribute
            graph.draw(node_size_by_obs=True, ax=mock_ax)
            captured = capsys.readouterr()
            assert "Warning: Node attribute 'nb_obs' not found" in captured.out
