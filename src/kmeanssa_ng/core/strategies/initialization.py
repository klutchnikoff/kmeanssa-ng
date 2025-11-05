"""Center initialization strategies for the simulated annealing algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..abstract import Center
    from ..simulated_annealing import SimulatedAnnealing


class InitializationStrategy(ABC):
    """Abstract base class for center initialization strategies."""

    @abstractmethod
    def initialize_centers(self, sa: SimulatedAnnealing) -> list[Center]:
        """Initialize and return k centers.

        Args:
            sa: The SimulatedAnnealing instance.

        Returns:
            A list of k initial centers.
        """
        raise NotImplementedError


class RandomInit(InitializationStrategy):
    """Initializes centers by sampling them randomly from the space."""

    def initialize_centers(self, sa: SimulatedAnnealing) -> list[Center]:
        """Sample k centers randomly."""
        return sa.space.sample_centers(sa.k)


class KMeansPlusPlus(InitializationStrategy):
    """Initializes centers using the k-means++ algorithm."""

    def initialize_centers(self, sa: SimulatedAnnealing) -> list[Center]:
        """Sample k centers using k-means++."""
        return sa.space.sample_kpp_centers(sa.k)
