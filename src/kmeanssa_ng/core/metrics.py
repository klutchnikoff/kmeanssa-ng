"""Clustering evaluation metrics for metric spaces.

This module provides metrics to evaluate clustering quality, wrapping
scikit-learn's implementations and adapting them for arbitrary metric spaces.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)

if TYPE_CHECKING:
    from .abstract import Center, Point, Space


def compute_labels(
    space: Space, points: list[Point], centers: list[Center]
) -> np.ndarray:
    """Assign each point to its nearest center.

    Args:
        space: The metric space containing the points.
        points: List of points to assign.
        centers: List of cluster centers.

    Returns:
        Array of cluster labels (integers from 0 to k-1).

    Example:
        ```python
        labels = compute_labels(graph, points, centers)
        print(f"Point 0 belongs to cluster {labels[0]}")
        ```
    """
    labels = []
    for point in points:
        distances = [space.distance(center, point) for center in centers]
        labels.append(np.argmin(distances))
    return np.array(labels)


def compute_distance_matrix(space: Space, points: list[Point]) -> np.ndarray:
    """Compute pairwise distance matrix between points.

    Args:
        space: The metric space containing the points.
        points: List of points.

    Returns:
        n×n symmetric matrix of pairwise distances.

    Note:
        This can be expensive for large datasets (O(n²) distances).
        For quantum graph points at nodes, uses precomputed distances when available.
    """
    # Optimization: if all points are quantum graph nodes, use precomputed distances
    if _all_points_at_nodes(points):
        return _distance_matrix_from_precomputed(space, points)

    # General case: compute all pairwise distances
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = space.distance(points[i], points[j])
            distances[i, j] = d
            distances[j, i] = d
    return distances


def _all_points_at_nodes(points: list[Point]) -> bool:
    """Check if all points are at graph nodes.

    A point is at a node if:
    - position == 0 (at edge[0])
    - position == edge_length (at edge[1])

    Args:
        points: List of points to check.

    Returns:
        True if all points are at nodes, False otherwise.
    """
    if not hasattr(points[0], "position"):
        return False

    for point in points:
        if point.position == 0:
            continue  # At edge[0]

        # Check if at edge[1] (position == edge_length)
        if hasattr(point, "space") and hasattr(point.space, "get_edge_length"):
            edge_length = point.space.get_edge_length(*point.edge)
            if abs(point.position - edge_length) < 1e-10:  # Floating point tolerance
                continue  # At edge[1]

        return False  # Point is in the middle of an edge

    return True


def _distance_matrix_from_precomputed(space: Space, points: list[Point]) -> np.ndarray:
    """Build distance matrix using precomputed node distances.

    Args:
        space: The quantum graph space.
        points: List of points at nodes.

    Returns:
        n×n distance matrix.
    """
    # Try to access precomputed distances
    if hasattr(space, "_pairwise_nodes_distance") and space._pairwise_nodes_distance:
        n = len(points)
        distances = np.zeros((n, n))

        # Extract node IDs from points
        # If position == 0, use edge[0]; if position == edge_length, use edge[1]
        node_ids = []
        for point in points:
            if point.position == 0:
                node_ids.append(point.edge[0])
            else:
                # Point is at edge[1]
                node_ids.append(point.edge[1])

        for i in range(n):
            for j in range(i + 1, n):
                d = space._pairwise_nodes_distance[node_ids[i]][node_ids[j]]
                distances[i, j] = d
                distances[j, i] = d

        return distances

    # Fallback: no precomputed distances available
    n = len(points)
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = space.distance(points[i], points[j])
            distances[i, j] = d
            distances[j, i] = d
    return distances


def silhouette(
    space: Space,
    points: list[Point],
    centers: list[Center],
    labels: np.ndarray | None = None,
) -> float:
    """Calculate the silhouette score for a clustering.

    The silhouette score measures how similar a point is to its own cluster
    compared to other clusters. Values range from -1 to 1:
    - 1: Point is far from neighboring clusters
    - 0: Point is on or very close to the decision boundary
    - -1: Point may have been assigned to the wrong cluster

    Args:
        space: The metric space containing the points.
        points: List of clustered points.
        centers: List of cluster centers.
        labels: Optional pre-computed cluster labels. If None, computed automatically.

    Returns:
        Mean silhouette score across all points.

    Raises:
        ValueError: If fewer than 2 clusters or all points in one cluster.

    Example:
        ```python
        from kmeanssa_ng.metrics import silhouette

        score = silhouette(graph, points, centers)
        print(f"Silhouette score: {score:.3f}")
        # Higher is better (max = 1.0)
        ```
    """
    if labels is None:
        labels = compute_labels(space, points, centers)

    # Silhouette requires precomputed distances
    distance_matrix = compute_distance_matrix(space, points)

    return silhouette_score(distance_matrix, labels, metric="precomputed")


def adjusted_rand_index(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Calculate the Adjusted Rand Index (ARI) between predicted and true labels.

    ARI measures the similarity between two clusterings, adjusted for chance.
    Values range from -1 to 1:
    - 1: Perfect agreement
    - 0: Random labeling
    - Negative: Worse than random

    Args:
        predicted_labels: Cluster labels from the algorithm.
        true_labels: Ground truth cluster labels.

    Returns:
        Adjusted Rand Index.

    Example:
        ```python
        from kmeanssa_ng.metrics import adjusted_rand_index, compute_labels

        predicted = compute_labels(graph, points, centers)
        true = [0, 0, 0, 1, 1, 1]  # Ground truth
        ari = adjusted_rand_index(predicted, true)
        print(f"ARI: {ari:.3f}")
        # 1.0 = perfect, 0.0 = random
        ```
    """
    return adjusted_rand_score(true_labels, predicted_labels)


def normalized_mutual_info(
    predicted_labels: np.ndarray,
    true_labels: np.ndarray,
) -> float:
    """Calculate Normalized Mutual Information (NMI) between predicted and true labels.

    NMI measures how much information is shared between two clusterings,
    normalized to be between 0 and 1.

    Args:
        predicted_labels: Cluster labels from the algorithm.
        true_labels: Ground truth cluster labels.

    Returns:
        Normalized Mutual Information (0 to 1).

    Example:
        ```python
        from kmeanssa_ng.metrics import normalized_mutual_info, compute_labels

        predicted = compute_labels(graph, points, centers)
        true = [0, 0, 0, 1, 1, 1]  # Ground truth
        nmi = normalized_mutual_info(predicted, true)
        print(f"NMI: {nmi:.3f}")
        # 1.0 = perfect agreement
        ```
    """
    return normalized_mutual_info_score(true_labels, predicted_labels)


def davies_bouldin(
    space: Space,
    points: list[Point],
    centers: list[Center],
    labels: np.ndarray | None = None,
) -> float:
    """Calculate the Davies-Bouldin index for a clustering.

    The Davies-Bouldin index evaluates cluster separation and compactness.
    Lower values indicate better clustering (0 is best).

    Args:
        space: The metric space containing the points.
        points: List of clustered points.
        centers: List of cluster centers.
        labels: Optional pre-computed cluster labels. If None, computed automatically.

    Returns:
        Davies-Bouldin index (lower is better).

    Example:
        ```python
        from kmeanssa_ng.metrics import davies_bouldin

        db = davies_bouldin(graph, points, centers)
        print(f"Davies-Bouldin: {db:.3f}")
        # Lower is better (min = 0)
        ```
    """
    if labels is None:
        labels = compute_labels(space, points, centers)

    # Convert points to feature matrix for sklearn
    # We use coordinates in embedding space (point positions)
    # For quantum graphs, we can use edge ID + position as features
    features = _points_to_features(points)

    return davies_bouldin_score(features, labels)


def calinski_harabasz(
    space: Space,
    points: list[Point],
    centers: list[Center],
    labels: np.ndarray | None = None,
) -> float:
    """Calculate the Calinski-Harabasz index (Variance Ratio Criterion).

    This metric evaluates cluster separation. Higher values indicate
    better-defined clusters.

    Args:
        space: The metric space containing the points.
        points: List of clustered points.
        centers: List of cluster centers.
        labels: Optional pre-computed cluster labels. If None, computed automatically.

    Returns:
        Calinski-Harabasz index (higher is better).

    Example:
        ```python
        from kmeanssa_ng.metrics import calinski_harabasz

        ch = calinski_harabasz(graph, points, centers)
        print(f"Calinski-Harabasz: {ch:.3f}")
        # Higher is better
        ```
    """
    if labels is None:
        labels = compute_labels(space, points, centers)

    # Convert points to feature matrix
    features = _points_to_features(points)

    return calinski_harabasz_score(features, labels)


def _points_to_features(points: list[Point]) -> np.ndarray:
    """Convert points to a feature matrix for sklearn metrics.

    For quantum graph points, we use (edge_hash, position) as features.
    For other point types, we try to extract numeric attributes.

    Args:
        points: List of points to convert.

    Returns:
        n×d array of features.
    """
    # Try to detect point type
    if hasattr(points[0], "edge") and hasattr(points[0], "position"):
        # Quantum graph points: use edge hash and position
        features = []
        for point in points:
            edge_hash = hash(point.edge) % 1000000  # Normalize hash
            features.append([edge_hash, point.position])
        return np.array(features)
    else:
        # Generic points: try to extract any numeric attributes
        # This is a fallback - users may need custom feature extraction
        raise NotImplementedError(
            "Feature extraction not implemented for this point type. "
            "Please convert points to features manually before using this metric."
        )


def evaluate_clustering(
    space: Space,
    points: list[Point],
    centers: list[Center],
    true_labels: np.ndarray | None = None,
) -> dict[str, float]:
    """Evaluate clustering quality with multiple metrics.

    Computes both intrinsic metrics (no ground truth needed) and
    extrinsic metrics (if ground truth labels provided).

    Args:
        space: The metric space containing the points.
        points: List of clustered points.
        centers: List of cluster centers.
        true_labels: Optional ground truth cluster labels.

    Returns:
        Dictionary with metric names and scores:
        - 'silhouette': Silhouette score (-1 to 1, higher is better)
        - 'davies_bouldin': Davies-Bouldin index (≥0, lower is better)
        - 'calinski_harabasz': Calinski-Harabasz score (≥0, higher is better)
        - 'ari': Adjusted Rand Index (-1 to 1, higher is better) [if true_labels provided]
        - 'nmi': Normalized Mutual Information (0 to 1, higher is better) [if true_labels provided]

    Example:
        ```python
        from kmeanssa_ng.metrics import evaluate_clustering

        # Without ground truth
        scores = evaluate_clustering(graph, points, centers)
        print(f"Silhouette: {scores['silhouette']:.3f}")

        # With ground truth
        scores = evaluate_clustering(graph, points, centers, true_labels)
        print(f"ARI: {scores['ari']:.3f}")
        print(f"NMI: {scores['nmi']:.3f}")
        ```
    """
    predicted_labels = compute_labels(space, points, centers)

    results = {
        "silhouette": silhouette(space, points, centers, predicted_labels),
        "davies_bouldin": davies_bouldin(space, points, centers, predicted_labels),
        "calinski_harabasz": calinski_harabasz(
            space, points, centers, predicted_labels
        ),
    }

    if true_labels is not None:
        results["ari"] = adjusted_rand_index(predicted_labels, true_labels)
        results["nmi"] = normalized_mutual_info(predicted_labels, true_labels)

    return results
