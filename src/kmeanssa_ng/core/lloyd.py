"""Lloyd's algorithm for k-means clustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .abstract import Center, Point
    from .strategies.initialization import InitializationStrategy
    from .strategies.lloyd_update import LloydUpdateStrategy


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
    ):
        """Initialize Lloyd's algorithm.

        Args:
            points: A list of points to be clustered.
            k: The number of clusters.
            update_strategy: The strategy for updating cluster centers.
        """
        if not points:
            raise ValueError("Input points list cannot be empty.")
        if k <= 0:
            raise ValueError("Number of clusters k must be positive.")

        self.points = points
        self.k = k
        self.space = points[0].space
        self.update_strategy = update_strategy

    @property
    def observations(self) -> list[Point]:
        """Return the points to be clustered.
        This is for compatibility with initialization strategies.
        """
        return self.points

    def run(
        self,
        initialization_strategy: InitializationStrategy,
        max_iterations: int = 100,
        tolerance: float = 1e-4,
    ) -> list[Center]:
        """Run Lloyd's algorithm.

        Args:
            initialization_strategy: The strategy for initializing centers.
            max_iterations: The maximum number of iterations to run.
            tolerance: The tolerance for convergence. If the change in
                energy is less than this value, the algorithm stops.

        Returns:
            A list of the final cluster centers.
        """
        # 1. Initialization
        centers = initialization_strategy.initialize_centers(self)

        last_energy = float("inf")

        for i in range(max_iterations):
            # 2. Assignment step
            labels = self.space.assign_clusters(self.points, centers)

            # 3. Update step
            new_centers = []
            for cluster_idx in range(self.k):
                cluster_points = [
                    p for j, p in enumerate(self.points) if labels[j] == cluster_idx
                ]
                if cluster_points:
                    new_center = self.update_strategy.update(cluster_points, self.space)
                    if new_center:
                        new_centers.append(new_center)

            if not new_centers:
                # This can happen if all points are in one cluster and the update fails
                # Or if all clusters are empty
                break

            centers = new_centers

            # Check for convergence
            current_energy = self.space.calculate_energy(centers)
            if abs(last_energy - current_energy) < tolerance:
                break
            last_energy = current_energy

        return centers
