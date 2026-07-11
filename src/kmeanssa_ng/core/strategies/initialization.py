"""Center initialization strategies for the simulated annealing algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

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
    """Initializes centers by sampling them randomly from the observations."""

    def initialize_centers(self, sa: SimulatedAnnealing) -> list[Center]:
        """Sample k distinct observations as the initial centers."""
        if sa.k <= 0:
            raise ValueError(f"k must be positive, got {sa.k}")

        if not sa.observations:
            raise ValueError(
                "No observations available. Call sample_points() first or provide observations."
            )

        # Sample without replacement so k distinct observations become k
        # distinct centers. Drawing with replacement could pick one point
        # several times, starting centers on top of each other and shrinking k
        # from the outset. Coincident observations (several points at one node)
        # can still yield coincident centers; KMeansPlusPlus spreads centers by
        # construction when that matters.
        if sa.k > len(sa.observations):
            raise ValueError(
                f"cannot initialize k={sa.k} distinct centers from "
                f"{len(sa.observations)} observations; use fewer clusters or "
                "more observations"
            )
        indices = sa._rng.choice(len(sa.observations), size=sa.k, replace=False)
        points = [sa.observations[idx] for idx in indices]
        return [sa.space.center_from_point(p) for p in points]


class KMeansPlusPlus(InitializationStrategy):
    """Initializes centers using the k-means++ algorithm."""

    def initialize_centers(self, sa: "SimulatedAnnealing") -> list[Center]:
        """Sample k centers using k-means++."""
        if sa.k <= 0:
            raise ValueError(f"k must be positive, got {sa.k}")

        if not sa.observations:
            raise ValueError(
                "No observations available. Call sample_points() first or provide observations."
            )

        centers = []

        # Step 1: Choose first center uniformly at random from observations
        first_idx = sa._rng.integers(len(sa.observations))
        first_point = sa.observations[first_idx]
        centers.append(sa.space.center_from_point(first_point))

        # Step 2-3: Choose remaining centers
        for _ in range(sa.k - 1):
            squared_distances = np.zeros(len(sa.observations))

            for i, obs_point in enumerate(sa.observations):
                min_dist_sq = min(
                    sa.space.distance(center, obs_point) ** 2 for center in centers
                )
                squared_distances[i] = min_dist_sq

            total = squared_distances.sum()
            if total > 0:
                probabilities = squared_distances / total
            else:
                probabilities = np.ones(len(sa.observations)) / len(sa.observations)

            next_idx = sa._rng.choice(len(sa.observations), p=probabilities)
            next_point = sa.observations[next_idx]
            centers.append(sa.space.center_from_point(next_point))

        return centers
