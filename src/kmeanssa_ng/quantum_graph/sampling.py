"""Sampling strategies for quantum graphs.

This module provides sampling strategies specific to quantum graphs,
allowing different probability distributions for selecting points on the graph.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.strategies.sampling import SamplingStrategy

if TYPE_CHECKING:
    from ..core.abstract import Point, Space


class UniformNodeSampling(SamplingStrategy):
    """Uniform sampling over graph nodes (discrete uniform distribution).

    This strategy samples points uniformly at random from the graph nodes,
    where each node has equal probability of being selected.

    This is the natural discrete uniform distribution for quantum graphs.

    Example:
        ```python
        from kmeanssa_ng import QuantumGraph
        from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        # Sample 100 points uniformly on nodes
        strategy = UniformNodeSampling()
        points = graph.sample_points(100, strategy=strategy)
        ```

    Note:
        For continuous uniform sampling along edges, see EdgeBasedSampling
        (to be implemented in a future version).
    """

    def sample(self, space: Space, n: int) -> list[Point]:
        """Sample n points uniformly from graph nodes.

        Args:
            space: The quantum graph to sample from.
            n: Number of points to sample.

        Returns:
            List of n points sampled uniformly from nodes.
        """
        return space._sample_uniform(n)
