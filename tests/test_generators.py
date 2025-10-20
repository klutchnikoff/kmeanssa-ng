"""Tests for quantum graph generators to improve coverage."""

import networkx as nx
import numpy as np
import pytest

from kmeanssa_ng import (
    QuantumGraph,
    generate_simple_graph,
    generate_simple_random_graph,
    generate_sbm,
    as_quantum_graph,
    complete_quantum_graph,
)


class TestGenerateSimpleGraphErrors:
    """Tests for error handling in generate_simple_graph."""

    def test_generate_simple_graph_invalid_n_a_type(self):
        """Test that invalid n_a type raises ValueError."""
        with pytest.raises(ValueError, match="n_a must be an integer"):
            generate_simple_graph(n_a="invalid")

    def test_generate_simple_graph_negative_n_a(self):
        """Test that negative n_a raises ValueError."""
        with pytest.raises(ValueError, match="n_a must be non-negative"):
            generate_simple_graph(n_a=-1)

    def test_generate_simple_graph_invalid_n_aa_type(self):
        """Test that invalid n_aa type raises ValueError."""
        with pytest.raises(ValueError, match="n_aa must be an integer"):
            generate_simple_graph(n_aa="invalid")

    def test_generate_simple_graph_negative_n_aa(self):
        """Test that negative n_aa raises ValueError."""
        with pytest.raises(ValueError, match="n_aa must be non-negative"):
            generate_simple_graph(n_aa=-1)

    def test_generate_simple_graph_invalid_bridge_length_type(self):
        """Test that invalid bridge_length type raises ValueError."""
        with pytest.raises(ValueError, match="bridge_length must be a number"):
            generate_simple_graph(bridge_length="invalid")

    def test_generate_simple_graph_zero_bridge_length(self):
        """Test that zero bridge_length raises ValueError."""
        with pytest.raises(ValueError, match="bridge_length must be positive"):
            generate_simple_graph(bridge_length=0)

    def test_generate_simple_graph_negative_bridge_length(self):
        """Test that negative bridge_length raises ValueError."""
        with pytest.raises(ValueError, match="bridge_length must be positive"):
            generate_simple_graph(bridge_length=-1.5)


class TestGenerateSimpleRandomGraph:
    """Tests for generate_simple_random_graph function."""

    def test_generate_simple_random_graph_default_params(self):
        """Test generate_simple_random_graph with default parameters."""
        graph = generate_simple_random_graph()

        # Should have at least central nodes A0 and B0
        assert graph.number_of_nodes() >= 2
        assert graph.has_node("A0")
        assert graph.has_node("B0")
        assert graph.has_edge("A0", "B0")

        # Check that graph has precomputed distances
        assert graph._pairwise_nodes_distance is not None

    def test_generate_simple_random_graph_custom_params(self):
        """Test generate_simple_random_graph with custom parameters."""
        graph = generate_simple_random_graph(
            n_a=3, n_b=2, lam_a=2, lam_b=1, bridge_length=5.0
        )

        # Should have A0, B0, A1-A3, B1-B2
        expected_nodes = ["A0", "B0", "A1", "A2", "A3", "B1", "B2"]
        for node in expected_nodes:
            assert graph.has_node(node)

        # Check bridge exists with correct length range
        bridge_data = graph.get_edge_data("A0", "B0")
        assert 4.5 <= bridge_data["length"] <= 5.5  # 0.9*5 to 1.1*5

    def test_generate_simple_random_graph_zero_poisson(self):
        """Test generate_simple_random_graph with zero Poisson parameters."""
        graph = generate_simple_random_graph(n_a=2, n_b=2, lam_a=0, lam_b=0)

        # Should only have first and second level nodes (no third level)
        expected_nodes = ["A0", "B0", "A1", "A2", "B1", "B2"]
        assert graph.number_of_nodes() == len(expected_nodes)


class TestGenerateSbmErrors:
    """Tests for error handling in generate_sbm."""

    def test_generate_sbm_default_params(self):
        """Test generate_sbm with default parameters (None values)."""
        graph = generate_sbm()
        assert graph.number_of_nodes() == 100  # Default [50, 50]
        assert graph.number_of_edges() > 0

    def test_generate_sbm_custom_none_params(self):
        """Test generate_sbm with explicit None parameters."""
        graph = generate_sbm(sizes=None, p=None)
        assert graph.number_of_nodes() == 100  # Default [50, 50]

    def test_generate_sbm_empty_sizes(self):
        """Test that empty sizes raises ValueError."""
        with pytest.raises(ValueError, match="sizes must be a non-empty list"):
            generate_sbm(sizes=[])

    def test_generate_sbm_non_list_sizes(self):
        """Test that non-list sizes raises ValueError."""
        with pytest.raises(ValueError, match="sizes must be a non-empty list"):
            generate_sbm(sizes="not a list")

    def test_generate_sbm_invalid_size_type(self):
        """Test that invalid size type raises ValueError."""
        with pytest.raises(ValueError, match="sizes\\[0\\] must be an integer"):
            generate_sbm(sizes=["invalid"])

    def test_generate_sbm_zero_size(self):
        """Test that zero size raises ValueError."""
        with pytest.raises(ValueError, match="sizes\\[0\\] must be positive"):
            generate_sbm(sizes=[0])

    def test_generate_sbm_empty_p(self):
        """Test that empty p raises ValueError."""
        with pytest.raises(ValueError, match="p must be a non-empty list"):
            generate_sbm(sizes=[50], p=[])

    def test_generate_sbm_non_list_p(self):
        """Test that non-list p raises ValueError."""
        with pytest.raises(ValueError, match="p must be a non-empty list"):
            generate_sbm(sizes=[50], p="not a list")

    def test_generate_sbm_wrong_p_size(self):
        """Test that wrong p size raises ValueError."""
        with pytest.raises(ValueError, match="p must have 2 rows to match sizes"):
            generate_sbm(sizes=[50, 50], p=[[0.5]])

    def test_generate_sbm_non_list_p_row(self):
        """Test that non-list p row raises ValueError."""
        with pytest.raises(ValueError, match="p\\[0\\] must be a list"):
            generate_sbm(sizes=[50], p=["not a list"])

    def test_generate_sbm_wrong_p_columns(self):
        """Test that wrong p columns raises ValueError."""
        with pytest.raises(ValueError, match="p\\[0\\] must have 1 columns"):
            generate_sbm(sizes=[50], p=[[0.5, 0.6]])

    def test_generate_sbm_invalid_prob_type(self):
        """Test that invalid probability type raises ValueError."""
        with pytest.raises(ValueError, match="p\\[0\\]\\[0\\] must be a number"):
            generate_sbm(sizes=[50], p=[["invalid"]])

    def test_generate_sbm_prob_out_of_range_high(self):
        """Test that probability > 1 raises ValueError."""
        with pytest.raises(ValueError, match="p\\[0\\]\\[0\\] must be in \\[0, 1\\]"):
            generate_sbm(sizes=[50], p=[[1.5]])

    def test_generate_sbm_prob_out_of_range_low(self):
        """Test that probability < 0 raises ValueError."""
        with pytest.raises(ValueError, match="p\\[0\\]\\[0\\] must be in \\[0, 1\\]"):
            generate_sbm(sizes=[50], p=[[-0.1]])


class TestAsQuantumGraph:
    """Tests for as_quantum_graph function."""

    def test_as_quantum_graph_basic(self):
        """Test basic conversion of NetworkX graph."""
        G = nx.Graph()
        G.add_edge(0, 1)
        G.add_edge(1, 2)

        qg = as_quantum_graph(G, node_weight=2.0, edge_length=1.5, edge_weight=0.8)

        # Check structure preserved
        assert qg.number_of_nodes() == 3
        assert qg.number_of_edges() == 2
        assert qg.has_edge(0, 1)
        assert qg.has_edge(1, 2)

        # Check attributes set correctly
        for node in qg.nodes():
            assert qg.nodes[node]["weight"] == 2.0

        for edge in qg.edges():
            edge_data = qg.get_edge_data(*edge)
            assert edge_data["length"] == 1.5
            assert edge_data["weight"] == 0.8
            assert "distribution" in edge_data

    def test_as_quantum_graph_karate_club(self):
        """Test conversion of Karate Club graph."""
        G = nx.karate_club_graph()
        qg = as_quantum_graph(G, edge_length=1.0)

        assert qg.number_of_nodes() == G.number_of_nodes()
        assert qg.number_of_edges() == G.number_of_edges()

        # All edges should have length 1.0
        for edge in qg.edges():
            assert qg.get_edge_data(*edge)["length"] == 1.0

    def test_as_quantum_graph_invalid_graph(self):
        """Test that invalid graph raises ValueError."""
        with pytest.raises(ValueError, match="`graph` must be a networkx.Graph object"):
            as_quantum_graph("not a graph")

    def test_as_quantum_graph_invalid_node_weight(self):
        """Test that invalid node_weight raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)

        with pytest.raises(ValueError, match="`node_weight` must be a positive number"):
            as_quantum_graph(G, node_weight=0)

    def test_as_quantum_graph_invalid_edge_length(self):
        """Test that invalid edge_length raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)

        with pytest.raises(ValueError, match="`edge_length` must be a positive number"):
            as_quantum_graph(G, edge_length=-1)

    def test_as_quantum_graph_invalid_edge_weight(self):
        """Test that invalid edge_weight raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)

        with pytest.raises(ValueError, match="`edge_weight` must be a positive number"):
            as_quantum_graph(G, edge_weight="invalid")


class TestCompleteQuantumGraph:
    """Tests for complete_quantum_graph function."""

    def test_complete_quantum_graph_basic(self):
        """Test basic complete graph creation."""
        objects = [1, 2, 3]
        graph = complete_quantum_graph(objects)

        # Should be complete graph with 3 nodes, 3 edges
        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3

        # Check all edges exist
        expected_edges = [(0, 1), (0, 2), (1, 2)]
        for edge in expected_edges:
            assert graph.has_edge(*edge)
            assert graph.get_edge_data(*edge)["length"] == 1.0

    def test_complete_quantum_graph_with_similarities(self):
        """Test complete graph with similarity matrix."""
        objects = [1, 2, 3]
        similarities = np.array([[0, 2, 3], [2, 0, 1], [3, 1, 0]])

        graph = complete_quantum_graph(objects, similarities=similarities)

        # Check edge lengths match similarities
        assert graph.get_edge_data(0, 1)["length"] == 2
        assert graph.get_edge_data(0, 2)["length"] == 3
        assert graph.get_edge_data(1, 2)["length"] == 1

    def test_complete_quantum_graph_with_labels(self):
        """Test complete graph with true labels."""
        objects = [1, 2, 3, 4]
        labels = [0, 0, 1, 1]

        graph = complete_quantum_graph(objects, true_labels=labels)

        # Check nodes have group attribute
        for i, expected_group in enumerate(labels):
            assert graph.nodes[i]["group"] == expected_group

    def test_complete_quantum_graph_empty_objects(self):
        """Test that empty objects list raises ValueError."""
        with pytest.raises(ValueError, match="`objects` must be a non-empty list"):
            complete_quantum_graph([])

    def test_complete_quantum_graph_non_list_objects(self):
        """Test that non-list objects raises ValueError."""
        with pytest.raises(ValueError, match="`objects` must be a non-empty list"):
            complete_quantum_graph("not a list")

    def test_complete_quantum_graph_invalid_similarities_type(self):
        """Test that non-array similarities raises ValueError."""
        objects = [1, 2]
        with pytest.raises(ValueError, match="`similarities` must be a numpy array"):
            complete_quantum_graph(objects, similarities="not an array")

    def test_complete_quantum_graph_wrong_similarities_shape(self):
        """Test that wrong similarities shape raises ValueError."""
        objects = [1, 2]
        similarities = np.array([[1, 2, 3]])  # Wrong shape

        with pytest.raises(
            ValueError, match="`similarities` must be a square matrix of size 2x2"
        ):
            complete_quantum_graph(objects, similarities=similarities)

    def test_complete_quantum_graph_negative_similarities(self):
        """Test that negative similarities raises ValueError."""
        objects = [1, 2]
        similarities = np.array([[0, -1], [1, 0]])  # Negative value

        with pytest.raises(
            ValueError, match="Elements of `similarities` must be non-negative"
        ):
            complete_quantum_graph(objects, similarities=similarities)

    def test_complete_quantum_graph_invalid_labels_type(self):
        """Test that non-list labels raises ValueError."""
        objects = [1, 2]
        with pytest.raises(ValueError, match="`true_labels` must be a list"):
            complete_quantum_graph(objects, true_labels="not a list")

    def test_complete_quantum_graph_wrong_labels_length(self):
        """Test that wrong labels length raises ValueError."""
        objects = [1, 2]
        labels = [0]  # Wrong length

        with pytest.raises(
            ValueError, match="`true_labels` must have the same length as `objects`"
        ):
            complete_quantum_graph(objects, true_labels=labels)

    def test_complete_quantum_graph_precomputes_distances(self):
        """Test that complete graph precomputes distances."""
        objects = [1, 2, 3]
        graph = complete_quantum_graph(objects)

        # Should have precomputed distances
        assert graph._pairwise_nodes_distance is not None
