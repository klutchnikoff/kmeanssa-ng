"""Generalized Lloyd's algorithm for k-means clustering on metric spaces."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

from .abstract import Center, Point, Space
from .strategies.initialization import InitializationStrategy

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class Lloyd:
    """Generalized Lloyd's algorithm for k-means clustering.

    This algorithm solves the k-means problem on arbitrary metric spaces using
    the iterative Lloyd's algorithm. It assigns each observation to the closest
    center and then updates the centers by computing the Fréchet mean of the
    assigned observations.

    Attributes:
        space: The metric space containing the observations.
        k: Number of clusters.
        observations: List of points to cluster.
        centers: Current cluster centers.
    """

    def __init__(
        self,
        observations: list[Point],
        k: int,
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the Lloyd's algorithm.

        Args:
            observations: List of points to cluster, all in the same metric space.
            k: Number of clusters.
            random_state: Controls randomness for reproducibility.
        """
        if not observations:
            raise ValueError("Observations must be a non-empty list of points.")
        if k <= 0:
            raise ValueError("Number of clusters 'k' must be greater than zero.")
        if any(obs.space != observations[0].space for obs in observations):
            raise ValueError("All observations must belong to the same metric space.")

        self._space = observations[0].space
        self._observations = observations.copy()
        self._k = k
        self._rng = np.random.default_rng(random_state)
        self._centers: list[Center] = []

    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self._observations)

    @property
    def observations(self) -> list[Point]:
        """List of observation points."""
        return self._observations

    @property
    def centers(self) -> list[Center]:
        """Current cluster centers."""
        return self._centers

    @property
    def space(self) -> Space:
        """Metric space containing the observations."""
        return self._space

    @property
    def k(self) -> int:
        """Number of clusters."""
        return self._k

    def run(
        self,
        initialization_strategy: InitializationStrategy,
        max_iter: int = 100,
        tol: float = 1e-4,
    ) -> list[Center]:
        """Run the generalized Lloyd's algorithm.

        Args:
            initialization_strategy: Strategy for initializing centers.
            max_iter: Maximum number of iterations.
            tol: Tolerance for convergence.

        Returns:
            List of final cluster centers.
        """
        logger.info(
            "Starting Lloyd's algorithm: k=%d, n_obs=%d, max_iter=%d, tol=%.4f",
            self._k,
            self.n,
            max_iter,
            tol,
        )

        self._centers = initialization_strategy.initialize_centers(self)

        logger.info("Lloyd's algorithm completed successfully.")
        return self._centers
