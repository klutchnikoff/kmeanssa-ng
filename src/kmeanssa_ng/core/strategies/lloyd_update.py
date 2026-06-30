"""Abstract base class for center update strategies in Lloyd's algorithm."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from ..abstract import Center, Point, Space
    from .initialization import InitializationStrategy


class LloydUpdateStrategy(ABC):
    """Abstract base class for strategies that update a cluster center.

    This is used in Lloyd's algorithm to compute the new center of a cluster
    based on the points assigned to it.
    """

    @abstractmethod
    def update(self, points: list[Point], space: Space) -> Center:
        """Compute the new center for a given cluster of points.

        Args:
            points: A list of points belonging to a single cluster.
            space: The metric space in which the points and center exist.

        Returns:
            The new center for the cluster.
        """
        raise NotImplementedError


class SimulatedAnnealingFrechetMean(LloydUpdateStrategy):
    """Approximates the Fréchet mean by running Simulated Annealing with k=1.

    This strategy provides a powerful, general-purpose way to find the
    center of a cluster in any metric space. It leverages the existing
    SimulatedAnnealing algorithm to find the point that minimizes the
    sum of squared distances to the points in the cluster.

    Args:
        n_samples: The number of points to sample (with replacement) from the
            cluster to use as observations for the inner SA run. If None, all
            points in the cluster are used.
        sa_initialization_strategy: The initialization strategy to use for the
            inner SA run. Defaults to RandomInit.
        **sa_kwargs: Additional keyword arguments to pass to the
            SimulatedAnnealing `run` method (e.g., T_max, T_min, n_iter).
    """

    def __init__(
        self,
        n_samples: int | None = None,
        sa_initialization_strategy: "InitializationStrategy" | None = None,
        random_state: int | np.random.Generator | None = None,
        **sa_kwargs,
    ):
        self.n_samples = n_samples
        self.sa_kwargs = sa_kwargs
        self.random_state = random_state

        if sa_initialization_strategy is None:
            from .initialization import RandomInit
            self.sa_initialization_strategy = RandomInit()
        else:
            self.sa_initialization_strategy = sa_initialization_strategy


    def update(self, points: list[Point], space: Space) -> Center:
        """Computes the Fréchet mean of the points using SA."""
        if not points:
            return None

        # Import locally to avoid circular dependency at module level
        from ..simulated_annealing import SimulatedAnnealing
        import numpy as np

        if self.n_samples is not None and len(points) > self.n_samples:
            if isinstance(self.random_state, np.random.Generator):
                rng = self.random_state
            else:
                rng = np.random.default_rng(self.random_state)
            indices = rng.choice(len(points), size=self.n_samples, replace=True)
            observations = [points[idx] for idx in indices]
        else:
            observations = points

        # Use SA to find the point that minimizes the energy (the Fréchet mean)
        sa = SimulatedAnnealing(
            observations=observations,
            k=1,
            random_state=self.random_state,
            **self.sa_kwargs
        )
        
        # The run method requires a robustification strategy.
        # MinimizeEnergy is a sensible default to find the best center.
        from .robustification import MinimizeEnergy

        # run() returns a list of centers, we want the single one
        centers = sa.run(
            initialization_strategy=self.sa_initialization_strategy,
            robustification_strategy=MinimizeEnergy()
        )

        if not centers:
            # This could happen if the SA run fails for some reason
            return None
            
        return centers[0]
