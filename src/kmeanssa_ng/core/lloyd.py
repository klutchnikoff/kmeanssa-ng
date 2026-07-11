"""Lloyd's algorithm for k-means clustering."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from .abstract import Center, Point
    from .strategies.initialization import InitializationStrategy
    from .strategies.lloyd_update import LloydUpdateStrategy

logger = logging.getLogger(__name__)


class Lloyd:
    """Implementation of Lloyd's algorithm for k-means clustering.

    This class provides a classic iterative implementation of k-means.
    It is strategy-based, allowing for custom initialization and center
    update logic.

    Attributes:
        points: The list of points to cluster.
        k: The number of clusters.
        space: The metric space in which the points reside.
        update_strategy: The strategy for computing new cluster centers.
    """

    def __init__(
        self,
        points: list[Point],
        k: int,
        update_strategy: LloydUpdateStrategy,
        random_state: int | np.random.Generator | None = None,
    ):
        """Initialize Lloyd's algorithm.

        Args:
            points: A list of points to be clustered.
            k: The number of clusters.
            update_strategy: The strategy for updating cluster centers.
            random_state: Controls randomness for reproducibility.
        """
        if not points:
            raise ValueError("Input points list cannot be empty.")
        if k <= 0:
            raise ValueError("Number of clusters k must be positive.")

        self.points = points
        self.k = k
        self.space = points[0].space
        self.update_strategy = update_strategy

        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(random_state)

    @property
    def observations(self) -> list[Point]:
        """Return the points to be clustered.
        This is for compatibility with initialization strategies.
        """
        return self.points

    def run(
        self,
        initialization_strategy: InitializationStrategy | None = None,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> list[Center]:
        """Run Lloyd's algorithm.

        Args:
            initialization_strategy: The strategy for initializing centers.
                Defaults to :class:`KMeansPlusPlus`. (Unlike the initialization,
                the ``update_strategy`` is required at construction: its
                canonical choice depends on the space — a graph node update, a
                Karcher mean on a manifold — so there is no universal default.)
            max_iterations: The maximum number of iterations to run.
            tolerance: The tolerance for convergence. If the change in
                energy is less than this value, the algorithm stops.

        Returns:
            A list of the final cluster centers.
        """
        if initialization_strategy is None:
            from .strategies.initialization import KMeansPlusPlus

            initialization_strategy = KMeansPlusPlus()

        # 1. Initialization
        centers = initialization_strategy.initialize_centers(self)

        last_energy = float("inf")

        for i in range(max_iterations):
            # 2. Assignment step
            labels = self.space.assign_clusters(self.points, centers)

            # 3. Update step: always produce exactly k centers, reseeding any
            # cluster that came out empty (or whose update failed), so k never
            # silently shrinks.
            new_centers = []
            for cluster_idx in range(self.k):
                cluster_points = [
                    p for j, p in enumerate(self.points) if labels[j] == cluster_idx
                ]
                new_center = (
                    self.update_strategy.update(cluster_points, self.space)
                    if cluster_points
                    else None
                )
                if new_center is None:
                    # Reseed against the configuration as it stands *now* —
                    # the centers already updated (including earlier reseeds
                    # of this same iteration) plus the not-yet-updated ones.
                    # Reseeding against the pre-iteration centers would hand
                    # every simultaneously-empty cluster the same farthest
                    # point, leaving k shrunk despite the reseeding.
                    reference = new_centers + centers[cluster_idx + 1 :]
                    new_center = self._reseed_center(reference or centers)
                    logger.warning(
                        "Cluster %d is empty; reseeding its center on the point "
                        "farthest from the current centers.",
                        cluster_idx,
                    )
                new_centers.append(new_center)

            centers = new_centers

            # Check for convergence on the empirical objective of this
            # algorithm's own points (the space may be shared with other
            # running algorithms, and may carry unrelated node measures).
            current_energy = self.space.calculate_energy(
                centers, how="empirical", observations=self.points
            )
            if abs(last_energy - current_energy) < tolerance:
                break
            last_energy = current_energy

        return centers

    def _reseed_center(self, centers: list[Center]) -> Center:
        """Center on the point farthest from the current centers.

        The farthest point is the one worst served by the current
        configuration, so seeding there maximally reduces the energy a lone
        empty cluster can recover.
        """
        farthest = max(
            self.points,
            key=lambda p: min(self.space.distance(p, c) for c in centers),
        )
        return self.space.center_from_point(farthest)
