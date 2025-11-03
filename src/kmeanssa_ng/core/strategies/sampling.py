"""Sampling strategies for point generation on metric spaces."""

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

    Example:
        ```python
        strategy = UniformSampling()
        points = space.sample_points(100, strategy=strategy)
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


class UniformSampling(SamplingStrategy):
    """Uniform sampling strategy (natural measure on the space).

    Uses the space's intrinsic uniform distribution method.
    For graphs: uniform over nodes.
    For manifolds: natural volume measure.
    """

    def sample(self, space: Space, n: int) -> list[Point]:
        """Sample n points uniformly."""
        return space._sample_uniform(n)
