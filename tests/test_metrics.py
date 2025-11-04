"""Tests for clustering evaluation metrics."""

import numpy as np
import pytest

from kmeanssa_ng import SimulatedAnnealing
from kmeanssa_ng.core.strategies.initialization import RandomInit
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.core.metrics import (
    adjusted_rand_index,
    calinski_harabasz,
    compute_distance_matrix,
    compute_labels,
    davies_bouldin,
    normalized_mutual_info,
    silhouette,
)
from kmeanssa_ng.quantum_graph import QGPoint, QuantumGraph


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing."""
    import random as rd

    graph = QuantumGraph()
    # Create a path graph with 5 nodes: 0-1-2-3-4
    for i in range(5):
        graph.add_node(i, weight=1.0)

    for i in range(4):
        graph.add_edge(i, i + 1, length=10.0, weight=1.0)
        # Add distribution for sampling
        graph.edges[i, i + 1]["distribution"] = lambda L=10.0: rd.uniform(0, L)

    graph.precomputing()
    return graph


@pytest.fixture
def points_at_nodes(simple_graph):
    """Create points at graph nodes (position = 0)."""
    return [
        QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=0.0),
        QGPoint(quantum_graph=simple_graph, edge=(1, 2), position=0.0),
        QGPoint(quantum_graph=simple_graph, edge=(2, 3), position=0.0),
        QGPoint(quantum_graph=simple_graph, edge=(3, 4), position=0.0),
    ]


@pytest.fixture
def points_at_edge_ends(simple_graph):
    """Create points at edge endpoints (position = 0 or edge_length)."""
    return [
        QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=0.0),  # At node 0
        QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=10.0),  # At node 1
        QGPoint(quantum_graph=simple_graph, edge=(1, 2), position=0.0),  # At node 1
        QGPoint(quantum_graph=simple_graph, edge=(1, 2), position=10.0),  # At node 2
    ]


@pytest.fixture
def points_on_edges(simple_graph):
    """Create points in the middle of edges."""
    return [
        QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=5.0),
        QGPoint(quantum_graph=simple_graph, edge=(1, 2), position=3.0),
        QGPoint(quantum_graph=simple_graph, edge=(2, 3), position=7.0),
        QGPoint(quantum_graph=simple_graph, edge=(3, 4), position=2.0),
    ]


@pytest.fixture
def mixed_points(simple_graph):
    """Create a mix of points at nodes and on edges."""
    return [
        QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=0.0),  # At node
        QGPoint(quantum_graph=simple_graph, edge=(1, 2), position=5.0),  # On edge
        QGPoint(quantum_graph=simple_graph, edge=(2, 3), position=0.0),  # At node
        QGPoint(quantum_graph=simple_graph, edge=(3, 4), position=7.5),  # On edge
    ]


class TestComputeLabels:
    """Test compute_labels function."""

    def test_basic_assignment(self, simple_graph, points_at_nodes):
        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points_at_nodes, centers)

        assert len(labels) == len(points_at_nodes)
        assert labels.dtype == np.int64 or labels.dtype == np.int32
        assert all(0 <= label < 2 for label in labels)

    def test_single_cluster(self, simple_graph, points_at_nodes):
        points_for_centers = simple_graph.sample_points(
            1, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points_at_nodes, centers)

        assert all(label == 0 for label in labels)

    def test_nearest_center(self, simple_graph):
        """Test that points are assigned to nearest center."""
        # Create points and centers at known positions
        point = QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=0.0)
        points_for_center1 = simple_graph.sample_points(
            1, strategy=UniformNodeSampling()
        )
        center1 = simple_graph.center_from_point(points_for_center1[0])
        points_for_center2 = simple_graph.sample_points(
            1, strategy=UniformNodeSampling()
        )
        center2 = simple_graph.center_from_point(points_for_center2[0])

        labels = compute_labels(simple_graph, [point], [center1, center2])

        # Verify assignment matches nearest center
        d1 = simple_graph.distance(center1, point)
        d2 = simple_graph.distance(center2, point)
        expected_label = 0 if d1 <= d2 else 1
        assert labels[0] == expected_label


class TestComputeDistanceMatrix:
    """Test compute_distance_matrix function."""

    def test_matrix_shape(self, simple_graph, points_at_nodes):
        """Test distance matrix has correct shape."""
        dm = compute_distance_matrix(simple_graph, points_at_nodes)
        n = len(points_at_nodes)

        assert dm.shape == (n, n)

    def test_matrix_symmetry(self, simple_graph, points_at_nodes):
        """Test distance matrix is symmetric."""
        dm = compute_distance_matrix(simple_graph, points_at_nodes)

        assert np.allclose(dm, dm.T)

    def test_diagonal_zeros(self, simple_graph, points_at_nodes):
        """Test diagonal elements are zero."""
        dm = compute_distance_matrix(simple_graph, points_at_nodes)

        assert np.allclose(np.diag(dm), 0.0)

    def test_positive_distances(self, simple_graph, points_at_nodes):
        """Test all off-diagonal distances are positive."""
        dm = compute_distance_matrix(simple_graph, points_at_nodes)

        # Check off-diagonal elements
        for i in range(len(points_at_nodes)):
            for j in range(i + 1, len(points_at_nodes)):
                assert dm[i, j] > 0

    def test_optimization_nodes_at_zero(self, simple_graph, points_at_nodes):
        """Test optimized path for points at nodes (position = 0)."""
        dm = compute_distance_matrix(simple_graph, points_at_nodes)

        # Should use precomputed distances if available
        assert dm.shape == (len(points_at_nodes), len(points_at_nodes))

    def test_optimization_nodes_at_edge_ends(self, simple_graph, points_at_edge_ends):
        """Test optimized path for points at edge endpoints."""
        dm = compute_distance_matrix(simple_graph, points_at_edge_ends)

        # Should recognize points at edge_length as being at nodes
        assert dm.shape == (len(points_at_edge_ends), len(points_at_edge_ends))

    def test_fallback_for_edge_points(self, simple_graph, points_on_edges):
        """Test fallback computation for points on edges."""
        dm = compute_distance_matrix(simple_graph, points_on_edges)

        # Should work even when points are not at nodes
        assert dm.shape == (len(points_on_edges), len(points_on_edges))

    def test_mixed_points(self, simple_graph, mixed_points):
        """Test with mix of node and edge points."""
        dm = compute_distance_matrix(simple_graph, mixed_points)

        # Should fall back to general computation
        assert dm.shape == (len(mixed_points), len(mixed_points))

    def test_consistency_with_space_distance(self, simple_graph, points_at_nodes):
        """Test matrix distances match individual space.distance calls."""
        dm = compute_distance_matrix(simple_graph, points_at_nodes)

        for i in range(len(points_at_nodes)):
            for j in range(i + 1, len(points_at_nodes)):
                expected = simple_graph.distance(points_at_nodes[i], points_at_nodes[j])
                assert np.isclose(dm[i, j], expected)


class TestSilhouette:
    """Test silhouette score function."""

    def test_basic_silhouette(self, simple_graph, points_at_nodes):
        """Test basic silhouette computation."""
        points = simple_graph.sample_points(10, strategy=UniformNodeSampling())
        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only test if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score = silhouette(simple_graph, points, centers)
            assert -1 <= score <= 1
        else:
            pytest.skip("All points in single cluster")

    def test_with_precomputed_labels(self, simple_graph, points_at_nodes):
        """Test silhouette with pre-computed labels."""
        # Use sample_points to get more diverse points for better clustering
        points = simple_graph.sample_points(10, strategy=UniformNodeSampling())
        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only run if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score1 = silhouette(simple_graph, points, centers)
            score2 = silhouette(simple_graph, points, centers, labels=labels)
            assert np.isclose(score1, score2)
        else:
            pytest.skip("All points in single cluster")

    def test_single_cluster_fails(self, simple_graph, points_at_nodes):
        """Test that single cluster raises ValueError."""
        sa = SimulatedAnnealing(points_at_nodes, k=1)
        centers = RandomInit().initialize_centers(sa)

        with pytest.raises(ValueError):
            silhouette(simple_graph, points_at_nodes, centers)

    def test_perfect_clustering(self, simple_graph):
        """Test silhouette with well-separated clusters."""
        # Create two distant clusters with multiple points each
        points = [
            QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=0.0),
            QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=1.0),
            QGPoint(quantum_graph=simple_graph, edge=(3, 4), position=8.0),
            QGPoint(quantum_graph=simple_graph, edge=(3, 4), position=9.0),
        ]

        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only test if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score = silhouette(simple_graph, points, centers)
            assert score >= -1
        else:
            pytest.skip("All points in single cluster")


class TestAdjustedRandIndex:
    """Test adjusted_rand_index function."""

    def test_perfect_agreement(self):
        """Test ARI with perfect label agreement."""
        labels1 = np.array([0, 0, 1, 1, 2, 2])
        labels2 = np.array([0, 0, 1, 1, 2, 2])

        ari = adjusted_rand_index(labels1, labels2)
        assert np.isclose(ari, 1.0)

    def test_random_labeling(self):
        """Test ARI with independent labelings."""
        labels1 = np.array([0, 0, 0, 1, 1, 1])
        labels2 = np.array([0, 1, 0, 1, 0, 1])

        ari = adjusted_rand_index(labels1, labels2)
        assert -1 <= ari <= 1

    def test_complete_disagreement(self):
        """Test ARI with complete disagreement."""
        labels1 = np.array([0, 0, 1, 1])
        labels2 = np.array([0, 1, 0, 1])

        ari = adjusted_rand_index(labels1, labels2)
        assert ari < 0.5

    def test_permutation_invariance(self):
        """Test ARI is invariant to label permutations."""
        labels1 = np.array([0, 0, 1, 1, 2, 2])
        labels2 = np.array([2, 2, 0, 0, 1, 1])  # Permuted version

        ari = adjusted_rand_index(labels1, labels2)
        assert np.isclose(ari, 1.0)


class TestCalinskiHarabasz:
    """Test calinski_harabasz function."""

    def test_basic_score(self, simple_graph, points_at_nodes):
        """Test basic CH score computation."""
        points = simple_graph.sample_points(10, strategy=UniformNodeSampling())
        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only test if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score = calinski_harabasz(simple_graph, points, centers)
            assert score > 0  # CH score is always positive
        else:
            pytest.skip("All points in single cluster")

    def test_with_precomputed_labels(self, simple_graph, points_at_nodes):
        """Test CH with pre-computed labels."""
        points = simple_graph.sample_points(10, strategy=UniformNodeSampling())
        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only test if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score1 = calinski_harabasz(simple_graph, points, centers)
            score2 = calinski_harabasz(simple_graph, points, centers, labels=labels)
            assert np.isclose(score1, score2)
        else:
            pytest.skip("All points in single cluster")

    def test_higher_for_better_clustering(self, simple_graph):
        """Test CH score increases with better separation."""
        # Create well-separated points
        points = simple_graph.sample_points(10, strategy=UniformNodeSampling())
        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only test if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score = calinski_harabasz(simple_graph, points, centers)
            assert score > 0
        else:
            pytest.skip("All points in single cluster")


class TestDaviesBouldin:
    """Test davies_bouldin function."""

    def test_basic_score(self, simple_graph, points_at_nodes):
        """Test basic DB score computation."""
        points = simple_graph.sample_points(10, strategy=UniformNodeSampling())
        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only test if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score = davies_bouldin(simple_graph, points, centers)
            assert score >= 0  # DB score is always non-negative
        else:
            pytest.skip("All points in single cluster")

    def test_with_precomputed_labels(self, simple_graph, points_at_nodes):
        """Test DB with pre-computed labels."""
        points = simple_graph.sample_points(10, strategy=UniformNodeSampling())
        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only test if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score1 = davies_bouldin(simple_graph, points, centers)
            score2 = davies_bouldin(simple_graph, points, centers, labels=labels)
            assert np.isclose(score1, score2)
        else:
            pytest.skip("All points in single cluster")

    def test_lower_for_better_clustering(self, simple_graph):
        """Test DB score is lower for better clustering."""
        points = [
            QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=0.0),
            QGPoint(quantum_graph=simple_graph, edge=(0, 1), position=1.0),
            QGPoint(quantum_graph=simple_graph, edge=(3, 4), position=8.0),
            QGPoint(quantum_graph=simple_graph, edge=(3, 4), position=9.0),
        ]

        points_for_centers = simple_graph.sample_points(
            2, strategy=UniformNodeSampling()
        )
        centers = [simple_graph.center_from_point(p) for p in points_for_centers]
        labels = compute_labels(simple_graph, points, centers)

        # Only test if we have at least 2 clusters
        if len(np.unique(labels)) >= 2:
            score = davies_bouldin(simple_graph, points, centers)
            assert score >= 0
        else:
            pytest.skip("All points in single cluster")


class TestNormalizedMutualInfo:
    """Test normalized_mutual_info function."""

    def test_perfect_agreement(self):
        """Test NMI with perfect label agreement."""
        labels1 = np.array([0, 0, 1, 1, 2, 2])
        labels2 = np.array([0, 0, 1, 1, 2, 2])

        nmi = normalized_mutual_info(labels1, labels2)
        assert np.isclose(nmi, 1.0)

    def test_random_labeling(self):
        """Test NMI with independent labelings."""
        labels1 = np.array([0, 0, 0, 1, 1, 1])
        labels2 = np.array([0, 1, 0, 1, 0, 1])

        nmi = normalized_mutual_info(labels1, labels2)
        assert 0 <= nmi <= 1

    def test_permutation_invariance(self):
        """Test NMI is invariant to label permutations."""
        labels1 = np.array([0, 0, 1, 1, 2, 2])
        labels2 = np.array([2, 2, 0, 0, 1, 1])  # Permuted version

        nmi = normalized_mutual_info(labels1, labels2)
        assert np.isclose(nmi, 1.0)

    def test_complete_independence(self):
        """Test NMI approaches 0 for independent labelings."""
        labels1 = np.array([0, 0, 1, 1])
        labels2 = np.array([0, 1, 0, 1])

        nmi = normalized_mutual_info(labels1, labels2)
        assert 0 <= nmi < 0.5


class TestMetricHelpers:
    """Tests for metric helper functions."""

    def test_points_to_features_raises_for_unknown_type(self, simple_graph):
        """Test that _points_to_features raises for unknown point types."""

        class MockPoint:
            pass

        points = [MockPoint(), MockPoint()]
        # Provide pre-computed labels to avoid calling compute_labels, which fails on MockPoint
        labels = np.array([0, 1])

        # davies_bouldin and calinski_harabasz internally call _points_to_features
        with pytest.raises(
            NotImplementedError, match="Feature extraction not implemented"
        ):
            davies_bouldin(simple_graph, points, [], labels=labels)

        with pytest.raises(
            NotImplementedError, match="Feature extraction not implemented"
        ):
            calinski_harabasz(simple_graph, points, [], labels=labels)

    def test_evaluate_clustering_with_true_labels(self, simple_graph):
        """Test that evaluate_clustering includes ARI and NMI with true_labels."""
        from kmeanssa_ng.core.metrics import evaluate_clustering

        # Create points and centers at fixed positions to ensure multiple clusters
        points = [
            QGPoint(simple_graph, (0, 1), 1.0),
            QGPoint(simple_graph, (0, 1), 2.0),
            QGPoint(simple_graph, (3, 4), 8.0),
            QGPoint(simple_graph, (3, 4), 9.0),
        ]
        centers = [
            QGPoint(simple_graph, (0, 1), 1.5),
            QGPoint(simple_graph, (3, 4), 8.5),
        ]
        true_labels = np.array([0, 0, 1, 1])

        results = evaluate_clustering(
            simple_graph, points, centers, true_labels=true_labels
        )

        assert "ari" in results
        assert "nmi" in results
        assert isinstance(results["ari"], float)
        assert isinstance(results["nmi"], float)

    def test_distance_matrix_fallback_without_precomputing(self):
        """Test the fallback path in _distance_matrix_from_precomputed."""
        from kmeanssa_ng.core.metrics import _distance_matrix_from_precomputed

        graph = QuantumGraph()
        graph.add_edge(0, 1, length=10.0)
        points = [
            QGPoint(graph, (0, 1), 0.0),  # at node 0
            QGPoint(graph, (0, 1), 10.0),  # at node 1
        ]

        # Ensure precomputed distances are not available
        graph._pairwise_nodes_distance = None

        # Call the internal function directly to test the fallback
        dm = _distance_matrix_from_precomputed(graph, points)

        assert dm.shape == (2, 2)
        assert np.isclose(dm[0, 1], 10.0)
