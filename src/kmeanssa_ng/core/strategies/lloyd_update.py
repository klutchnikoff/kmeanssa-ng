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

    This strategy provides a general-purpose way to find the center of a
    cluster in **any** metric space of the package: it only uses the abstract
    Space/Center contracts (Brownian motion, drift, distances), so it runs
    unchanged on quantum graphs and on Riemannian manifolds — including
    quotient surfaces such as Bolza, whose dynamics are quotient-aware.

    Being a stochastic global optimizer, it can escape bad critical points of
    the Fréchet functional, at the cost of one annealing run per cluster per
    Lloyd iteration and of stochastic error. On manifolds with ``exp``/``log``
    maps, the deterministic ``KarcherFrechetMean`` (Karcher iteration) is the
    faster, locally-exact default.

    Args:
        n_samples: Number of observations fed to the inner SA run, resampled
            from the cluster **with replacement** (up or down). The annealing
            horizon grows like sqrt(n_samples), so resampling a small cluster
            up to n_samples gives it a full schedule while leaving the
            empirical measure — hence the Fréchet functional — unchanged.
            If None, the cluster points are used as they are (small clusters
            then get almost no annealing time).
        sa_initialization_strategy: The initialization strategy to use for the
            inner SA run. Defaults to RandomInit.
        robust_prop: Memory window of the inner run: the returned center is
            the lowest-energy state visited over the trailing ``robust_prop``
            fraction of the trajectory. With 0.0 only the final (still hot)
            state would be returned.
        **sa_kwargs: Additional keyword arguments passed to the
            SimulatedAnnealing constructor (e.g., lambda0, beta0, step_size).
    """

    def __init__(
        self,
        n_samples: int | None = None,
        sa_initialization_strategy: "InitializationStrategy" | None = None,
        random_state: int | np.random.Generator | None = None,
        robust_prop: float = 0.1,
        **sa_kwargs,
    ):
        self.n_samples = n_samples
        self.sa_kwargs = sa_kwargs
        # Normalize once: with an int seed, the previous code rebuilt
        # default_rng(seed) on *every* update() call, so every cluster and
        # every Lloyd iteration reused an identical stream (the resample drew
        # the same indices, the inner SA followed the same trajectory). One
        # owned Generator advances across calls, decorrelating them.
        self._rng = (
            random_state
            if isinstance(random_state, np.random.Generator)
            else np.random.default_rng(random_state)
        )
        self.robust_prop = robust_prop

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

        if self.n_samples is not None and len(points) != self.n_samples:
            # Resample with replacement, up or down: the empirical measure is
            # unchanged, but the inner annealing horizon (~ sqrt(#obs)) is
            # decoupled from the cluster size.
            indices = self._rng.choice(len(points), size=self.n_samples, replace=True)
            observations = [points[idx] for idx in indices]
        else:
            observations = points

        # Use SA to find the point that minimizes the energy (the Fréchet
        # mean). The Fréchet functional is by definition the empirical energy
        # of the cluster points: with any other mode, MinimizeEnergy would
        # select the best visited state against a foreign objective (on a
        # graph, the old "obs"/uniform default measured the distance to *all*
        # nodes, biasing the returned mean toward the graph's global center).
        if self.sa_kwargs.get("energy_mode", "empirical") != "empirical":
            raise ValueError(
                "SimulatedAnnealingFrechetMean pins energy_mode='empirical': "
                "the Fréchet functional is the empirical energy of the "
                "cluster points, any other selection objective returns a "
                "biased mean"
            )
        sa = SimulatedAnnealing(
            observations=observations,
            k=1,
            random_state=self._rng,
            **{**self.sa_kwargs, "energy_mode": "empirical"},
        )

        # The run method requires a robustification strategy.
        # MinimizeEnergy is a sensible default to find the best center.
        from .robustification import MinimizeEnergy

        # run() returns a list of centers, we want the single one
        centers = sa.run(
            initialization_strategy=self.sa_initialization_strategy,
            robustification_strategy=MinimizeEnergy(),
            robust_prop=self.robust_prop,
        )

        if not centers:
            # This could happen if the SA run fails for some reason
            return None

        return centers[0]
