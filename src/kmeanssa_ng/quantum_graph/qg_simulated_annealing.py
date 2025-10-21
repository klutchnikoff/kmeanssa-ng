"""Quantum graph specific simulated annealing implementation."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import numpy as np

from ..core import SimulatedAnnealing
from .strategies import MostFrequentNodeStrategy

if TYPE_CHECKING:
    pass


class QGSimulatedAnnealing(SimulatedAnnealing):
    """Simulated annealing specialized for quantum graphs.

    This class can be used with quantum graph specific strategies, such as
    finding the most frequent node visited by centers during the simulation.

    Example:
        ```python
        from kmeanssa_ng import generate_sbm, QGSimulatedAnnealing
        from kmeanssa_ng.quantum_graph.strategies import MostFrequentNodeStrategy

        graph = generate_sbm(sizes=[50, 50], p=[[0.7, 0.1], [0.1, 0.7]])
        points = graph.sample_points(100)

        # To get the most frequent nodes (for clustering)
        sa_kmeans = QGSimulatedAnnealing(points, k=2)
        node_ids = sa_kmeans.run_interleaved(
            robust_prop=0.1, strategy=MostFrequentNodeStrategy()
        )

        # To get the most frequent node (for mean computation)
        sa_mean = QGSimulatedAnnealing(points, k=1)
        node_id = sa_mean.run_interleaved(
            robust_prop=0.1, strategy=MostFrequentNodeStrategy()
        )
        ```
    """

    pass  # All specific logic is now handled by strategies
