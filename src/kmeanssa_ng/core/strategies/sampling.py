"""Sampling strategies for point generation on metric spaces.

This module defines the abstract base class for sampling strategies.
Concrete implementations are provided in space-specific modules:
- quantum_graph.sampling: Strategies for quantum graphs
- riemannian_manifold.sampling: Strategies for Riemannian manifolds
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import numpy as np
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

    def __init__(self, random_state: int | np.random.Generator | None = None):
        """Initialize the sampling strategy.

        Args:
            random_state: Controls randomness for reproducibility. Normalized
                once to a numpy Generator (see the ``random_state`` property),
                so the strategy owns a single stream that advances across
                calls — successive ``sample`` calls on one instance draw
                *different* points, as expected. Passing an int seeds it
                reproducibly; a Generator is adopted as-is.
        """
        self.random_state = random_state

    @property
    def random_state(self) -> np.random.Generator:
        """The strategy's numpy Generator (always normalized)."""
        return self._rng

    @random_state.setter
    def random_state(self, value: int | np.random.Generator | None) -> None:
        # Normalize on every assignment: an int seed used to be re-wrapped in a
        # fresh default_rng on *each* draw, so two sample() calls on one
        # instance returned identical points. Wrapping once, here, gives one
        # advancing stream. Assigning a Generator (as run_parallel does per
        # worker) adopts it directly.
        self._rng = (
            value
            if isinstance(value, np.random.Generator)
            else np.random.default_rng(value)
        )

    def _get_rng(self) -> np.random.Generator:
        """The strategy's Generator (kept for subclasses that call it)."""
        return self._rng

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
