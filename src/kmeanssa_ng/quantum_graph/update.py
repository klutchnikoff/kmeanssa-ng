"""Update strategies for QuantumGraph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.strategies.update import UpdateStrategy

if TYPE_CHECKING:
    from ..core.abstract import Center, Point, Space
    from .point import QGPoint


class MostFrequentNodeUpdate(UpdateStrategy):
    """Update strategy that computes the new center as the most frequent
    closest node to the points in the cluster.
    """

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
        return space.node_as_center(most_frequent_node)

# Helper method in QuantumGraph to get the point type
# I need to add this to QuantumGraph
# def get_point_type(self) -> type[Point]:
#     return QGPoint
