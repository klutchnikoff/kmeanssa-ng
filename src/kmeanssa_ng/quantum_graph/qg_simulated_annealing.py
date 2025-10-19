"""Quantum graph specific simulated annealing implementation."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from ..core import SimulatedAnnealing

if TYPE_CHECKING:
    pass


class QGSimulatedAnnealing(SimulatedAnnealing):
    """Simulated annealing specialized for quantum graphs.

    This class extends the core SimulatedAnnealing with quantum graph specific
    methods that rely on node proximity information.

    Example:
        ```python
        from kmeanssa_ng import generate_sbm, QGSimulatedAnnealing

        graph = generate_sbm(sizes=[50, 50], p=[[0.7, 0.1], [0.1, 0.7]])
        points = graph.sample_points(100)

        sa = QGSimulatedAnnealing(points, k=2)
        node_ids = sa.run_for_kmeans(robust_prop=0.1)
        ```
    """

    def run_for_mean(self, robust_prop: float = 0.0) -> int:
        """Run simulated annealing for k=1 and return most frequent closest node.

        This is a special case for computing a robust mean (single cluster)
        on a quantum graph. The result is the node ID that appears most
        frequently as the closest node during the robustification period.

        Args:
            robust_prop: Proportion of final iterations for robustification.

        Returns:
            Index of the most frequent closest node during robustification period.

        Raises:
            ValueError: If robust_prop not in [0, 1] or k != 1.
        """
        if robust_prop < 0 or robust_prop > 1:
            raise ValueError("The proportion must be in [0,1]")
        if self._k != 1:
            raise ValueError("run_for_mean only works with k=1")

        i0 = int(np.floor((self.n - 1) * (1 - robust_prop)))
        self._centers = self._initialize_kpp_centers()

        central_nodes = []
        times = self._initialize_times(self.n)
        time = 0.0

        for i, point in enumerate(self._observations):
            T = times[i]

            while time <= T - self._step_size:
                h = min(time + self._step_size, T) - time
                prop = min(h * self._beta * np.log(1 + time), 1)

                closest_center = None
                min_distance = float("inf")

                for center in self._centers:
                    center.brownian_motion(h)
                    dist = self.space.distance(center, point)
                    if dist < min_distance:
                        closest_center, min_distance = center, dist

                if closest_center is not None:
                    closest_center.drift(point, prop)

                time += h

            if i >= i0:
                central_nodes.append(self._centers[0]._closest_node())

        return Counter(central_nodes).most_common(1)[0][0]

    def run_for_kmeans(self, robust_prop: float = 0.0) -> list:
        """Run simulated annealing and return most frequent closest nodes for each center.

        This method is specific to quantum graphs and returns the node IDs
        that appear most frequently as closest nodes for each center during
        the robustification period.

        Args:
            robust_prop: Proportion of final iterations for robustification.

        Returns:
            List of k node IDs (most frequent for each center).

        Raises:
            ValueError: If robust_prop not in [0, 1].
        """
        if robust_prop < 0 or robust_prop > 1:
            raise ValueError("The proportion must be in [0,1]")

        i0 = int(np.floor((self.n - 1) * (1 - robust_prop)))
        self._centers = self._initialize_kpp_centers()

        # Use object dtype to support both int and string node IDs
        central_nodes = np.empty((self.n - i0, self._k), dtype=object)
        times = self._initialize_times(self.n)
        time = 0.0

        for i, point in enumerate(self._observations):
            T = times[i]

            while time <= T - self._step_size:
                h = min(time + self._step_size, T) - time
                prop = min(h * self._beta * np.log(1 + time), 1)

                closest_center = None
                min_distance = float("inf")

                for center in self._centers:
                    center.brownian_motion(h)
                    dist = self.space.distance(center, point)
                    if dist < min_distance:
                        closest_center, min_distance = center, dist

                if closest_center is not None:
                    closest_center.drift(point, prop)

                time += h

            if i >= i0:
                for m, center in enumerate(self._centers):
                    central_nodes[i - i0, m] = center._closest_node()

        robust_nodes = []
        for m in range(central_nodes.shape[1]):
            nodes_list = list(central_nodes[:, m])
            robust_nodes.append(Counter(nodes_list).most_common(1)[0][0])

        return robust_nodes
