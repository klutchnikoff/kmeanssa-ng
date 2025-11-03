"""Sampling strategies for point generation on metric spaces.

This module defines the abstract base class for sampling strategies.
Concrete implementations are provided in space-specific modules:
- quantum_graph.sampling: Strategies for quantum graphs
- riemannian_manifold.sampling: Strategies for Riemannian manifolds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..abstract import Point, Space


class SamplingStrategy(ABC):
    """Abstract base class for point sampling strategies.

    Sampling strategies define probability distributions for generating
    points on metric spaces. Each strategy must implement the sample()
    method to generate points according to its distribution.

    Concrete implementations are space-specific and located in:
    - kmeanssa_ng.quantum_graph.sampling
    - kmeanssa_ng.riemannian_manifold.sampling

    Example:
        ```python
        from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

        strategy = UniformNodeSampling()
        points = graph.sample_points(100, strategy=strategy)
        ```
    """

    @abstractmethod
    def sample(self, space: Space, n: int) -> list[Point]:
        """Sample n points from the space according to this strategy.

        Args:
            space: The metric space to sample from
            n: Number of points to sample

        Returns:
            List of n sampled points
        """
        raise NotImplementedError
