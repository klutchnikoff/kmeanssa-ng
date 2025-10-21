"""Quantum Graph specific robustification strategies."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

from ..core.strategies import RobustificationStrategy

if TYPE_CHECKING:
    from ..core.simulated_annealing import SimulatedAnnealing


class MostFrequentNodeStrategy(RobustificationStrategy[list[Any]]):
    """Strategy to find the most frequent node for each center."""

    def initialize(self, sa: "SimulatedAnnealing") -> None:
        """Initialize an empty list to store node collections."""
        self._central_nodes_collections: list[list] = []
        self.sa = sa

    def collect(self, sa: "SimulatedAnnealing") -> None:
        """Collect the closest node for each center at the current step."""
        current_nodes = [center._closest_node() for center in sa.centers]
        self._central_nodes_collections.append(current_nodes)

    def get_result(self) -> list[Any] | Any:
        """Return the most frequent node for each center.

        If k=1, returns a single node ID. Otherwise, returns a list of node IDs.
        """
        if not self._central_nodes_collections:
            return [] if self.sa._k > 1 else None

        num_centers = len(self._central_nodes_collections[0])
        transposed_nodes = [
            [nodes[i] for nodes in self._central_nodes_collections]
            for i in range(num_centers)
        ]

        robust_nodes = [
            Counter(center_nodes).most_common(1)[0][0]
            for center_nodes in transposed_nodes
        ]

        # For k=1 (mean computation), return the single element, not a list
        if len(robust_nodes) == 1:
            return robust_nodes[0]
        return robust_nodes
