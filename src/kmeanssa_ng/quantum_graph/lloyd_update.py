"""Update strategies for QuantumGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np

from ..core.strategies.lloyd_update import LloydUpdateStrategy

if TYPE_CHECKING:
    from .center import QGCenter
    from .point import QGPoint
    from .space import QuantumGraph


class MostFrequentNodeUpdate(LloydUpdateStrategy):
    """Update strategy that computes the new center as the most frequent
    closest node to the points in the cluster.
    """

    def __init__(self, random_state: int | np.random.Generator | None = None):
        # Normalize once so the tie-breaking stream advances across update()
        # calls instead of restarting from the same int seed each time.
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    def update(self, points: list[QGPoint], space: "QuantumGraph") -> "QGCenter":
        """Compute the new center for a given cluster of points.

        Args:
            points: A list of points belonging to a single cluster.
            space: The quantum graph in which the points and center exist.

        Returns:
            The new center for the cluster.
        """
        if not points:
            # Or should this raise an error? Or return a random point?
            # For now, let's follow the old logic which would fail on empty.
            # Lloyd's should handle empty clusters.
            return None

        # Find the closest node for each point
        closest_nodes = []
        for p in points:
            if not isinstance(p, space.get_point_type()):
                raise TypeError("All points must be QGPoint instances.")
            closest_nodes.append(p.closest_node())

        # Count occurrences of each node
        node_counts = {}
        for node in closest_nodes:
            node_counts[node] = node_counts.get(node, 0) + 1

        # Find the most frequent node
        if not node_counts:
            raise ValueError("No valid nodes found for Fréchet mean approximation.")

        most_frequent_node = max(node_counts, key=node_counts.get)

        # Return a QGCenter at the most frequent node
        return space.node_as_center(most_frequent_node, rng=self._rng)


class MinimizeEnergyNodeUpdate(LloydUpdateStrategy):
    """Update strategy that finds the node that minimizes k-means energy.

    This strategy first maps each point in the cluster to its nearest node,
    then finds the graph node that minimizes the sum of squared distances
    (k-means energy) to this set of cluster nodes.
    """

    def __init__(self, random_state: int | np.random.Generator | None = None):
        # Normalize once (see MostFrequentNodeUpdate): one advancing stream
        # across update() calls rather than a fresh default_rng each time.
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )

    def update(self, points: list[QGPoint], space: "QuantumGraph") -> "QGCenter":
        """Compute the new center by finding the node that minimizes energy.

        Args:
            points: A list of points belonging to a single cluster.
            space: The quantum graph, which must have distances precomputed.

        Returns:
            The new center for the cluster.
        """
        if not points:
            return None

        if space._pairwise_nodes_distance_array is None:
            raise ValueError("MinimizeEnergyNodeUpdate requires precomputed distances.")

        # 1. Replace each point with its closest node
        cluster_nodes = [p.closest_node() for p in points]

        # 2. Find the node that minimizes the energy
        dist_matrix = space._pairwise_nodes_distance_array
        node_to_idx = space._node_to_index
        all_graph_nodes = list(space.nodes())

        cluster_node_indices = [node_to_idx[n] for n in cluster_nodes]

        min_energy = float("inf")
        best_node = -1

        # Iterate through all nodes of the graph as potential centers
        for i, candidate_node in enumerate(all_graph_nodes):
            candidate_node_idx = i  # Index corresponds to order in all_graph_nodes

            # Calculate energy for this candidate node
            energy = 0.0
            for node_idx in cluster_node_indices:
                energy += dist_matrix[candidate_node_idx, node_idx] ** 2

            if energy < min_energy:
                min_energy = energy
                best_node = candidate_node

        if best_node == -1:
            # This should not happen if there are points
            raise ValueError("Could not determine best node to minimize energy.")

        # 3. Return the new center
        return space.node_as_center(best_node, rng=self._rng)


# Helper method in QuantumGraph to get the point type
# I need to add this to QuantumGraph
# def get_point_type(self) -> type[Point]:
#     return QGPoint
