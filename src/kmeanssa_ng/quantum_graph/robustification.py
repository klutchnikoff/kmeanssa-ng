"""Quantum Graph specific robustification strategies."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from ..core.strategies.robustification import RobustificationStrategy

if TYPE_CHECKING:
    from ..core.simulated_annealing import SimulatedAnnealing


class MostFrequentNode(RobustificationStrategy[list[Any]]):
    """Strategy to find the most frequent node for each center.

    Returns QGCenter objects located at the most frequently visited nodes
    during the robustification phase.
    """

    def initialize(self, sa: SimulatedAnnealing) -> None:
        """Initialize an empty list to store node collections.

        Raises:
            TypeError: If the space is not a QuantumGraph.
        """
        from .space import QuantumGraph

        if not isinstance(sa.space, QuantumGraph):
            raise TypeError(
                "MostFrequentNode strategy can only be used with QuantumGraph spaces."
            )

        self._central_nodes_collections: list[list] = []
        self.sa = sa

    def collect(self, sa: SimulatedAnnealing) -> None:
        """Collect the closest node for each center at the current step."""
        current_nodes = [center._closest_node() for center in sa.centers]
        self._central_nodes_collections.append(current_nodes)

    def get_result(self) -> list[Any]:
        """Return QGCenter objects at the most frequent nodes.

        Always returns a list of QGCenter objects, even for k=1.
        """
        if not self._central_nodes_collections:
            return []

        num_centers = len(self._central_nodes_collections[0])
        transposed_nodes = [
            [nodes[i] for nodes in self._central_nodes_collections]
            for i in range(num_centers)
        ]

        robust_nodes = [
            Counter(center_nodes).most_common(1)[0][0]
            for center_nodes in transposed_nodes
        ]

        # Convert node IDs to QGCenter objects
        from .space import QuantumGraph

        if isinstance(self.sa.space, QuantumGraph):
            robust_centers = [
                self.sa.space.node_as_center(node) for node in robust_nodes
            ]
        else:
            # Fallback for non-QuantumGraph spaces (shouldn't happen with this strategy)
            robust_centers = robust_nodes

        # Always return a list for consistency
        return robust_centers
