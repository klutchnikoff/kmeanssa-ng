"""Implements the Space abstract base class for Riemannian Manifolds using geomstats."""

from __future__ import annotations

import numpy as np

from kmeanssa_ng.core.abstract import Space

from .center import RiemannianCenter
from .point import RiemannianPoint


class RiemannianManifold(Space):
    """A Riemannian manifold space using geomstats.

    This class wraps a geomstats manifold object and implements the Space
    interface for k-means clustering on Riemannian manifolds.

    Attributes:
        manifold: The geomstats manifold object.
        observations: List of sampled point coordinates for energy calculation.

    Note:
        On manifolds with non-unique geodesics (e.g., antipodal points on spheres),
        the drift operation may exhibit degenerate behavior where centers do not
        move toward their targets. This is a known limitation of geodesic
        computation. The Brownian motion in the simulated annealing algorithm
        provides thermal agitation to escape such configurations.

    Example:
        ```python
        from geomstats.geometry.hypersphere import Hypersphere
        sphere = Hypersphere(dim=2)
        space = RiemannianManifold(sphere)
        points = space.sample_points(100)
        centers = space.sample_kpp_centers(3)
        energy = space.calculate_energy(centers)
        ```
    """

    def __init__(self, manifold) -> None:
        """Initialize a Riemannian manifold space.

        Args:
            manifold: A geomstats manifold object (e.g., Hypersphere, Hyperboloid).
        """
        self.manifold = manifold
        self.observations = []  # Store sampled points for energy calculation

    def distance(self, point1: RiemannianPoint, point2: RiemannianPoint) -> float:
        """Compute the geodesic distance between two points.

        Uses the manifold's Riemannian metric to compute the distance.

        Args:
            point1: First point.
            point2: Second point.

        Returns:
            The geodesic distance between point1 and point2.
        """
        dist = self.manifold.metric.dist(point1.coordinates, point2.coordinates)
        # Handle both scalar and 0-d array results
        return float(np.asarray(dist).item())

    def sample_points(self, n: int) -> list[RiemannianPoint]:
        """Sample n points uniformly from the manifold.

        Uses geomstats' random_uniform or random_point method to sample points
        according to the natural measure on the manifold.

        Args:
            n: Number of points to sample.

        Returns:
            List of n uniformly sampled RiemannianPoint objects.
        """
        # Sample coordinates from the manifold
        # Use random_uniform if available, otherwise random_point
        if hasattr(self.manifold, "random_uniform"):
            coords = self.manifold.random_uniform(n_samples=n)
        else:
            coords = self.manifold.random_point(n_samples=n)

        # Store observations for energy calculation
        if n > 0:
            self.observations = coords if coords.ndim > 1 else coords.reshape(1, -1)

        # Create RiemannianPoint objects
        points = []
        for i in range(n):
            point_coords = coords[i] if coords.ndim > 1 else coords
            points.append(RiemannianPoint(self, point_coords))

        return points

    def sample_centers(self, k: int) -> list[RiemannianCenter]:
        """Sample k centers uniformly from the manifold.

        Args:
            k: Number of centers to sample.

        Returns:
            List of k uniformly sampled RiemannianCenter objects.
        """
        # Use random_uniform if available, otherwise random_point
        if hasattr(self.manifold, "random_uniform"):
            coords = self.manifold.random_uniform(n_samples=k)
        else:
            coords = self.manifold.random_point(n_samples=k)

        centers = []
        for i in range(k):
            point_coords = coords[i] if coords.ndim > 1 else coords
            point = RiemannianPoint(self, point_coords)
            centers.append(RiemannianCenter(point))

        return centers

    def sample_kpp_centers(self, k: int) -> list[RiemannianCenter]:
        """Sample k centers using k-means++ initialization.

        The k-means++ algorithm:
        1. Choose first center uniformly at random
        2. For each subsequent center, choose with probability proportional
           to squared distance to nearest existing center
        3. Repeat until k centers are chosen

        Args:
            k: Number of centers to sample.

        Returns:
            List of k centers sampled using k-means++.

        Raises:
            ValueError: If k <= 0 or if no observations have been sampled.
        """
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        if len(self.observations) == 0:
            raise ValueError(
                "No observations available. Call sample_points() first or provide observations."
            )

        centers = []

        # Step 1: Choose first center uniformly at random from observations
        first_idx = np.random.randint(len(self.observations))
        first_point = RiemannianPoint(self, self.observations[first_idx])
        centers.append(RiemannianCenter(first_point))

        # Step 2-3: Choose remaining centers using k-means++
        for _ in range(k - 1):
            # Compute squared distances from each observation to nearest center
            squared_distances = np.zeros(len(self.observations))

            for i, obs_coords in enumerate(self.observations):
                obs_point = RiemannianPoint(self, obs_coords)
                min_dist_sq = min(
                    self.distance(center, obs_point) ** 2 for center in centers
                )
                squared_distances[i] = min_dist_sq

            # Choose next center with probability proportional to squared distance
            # Normalize to get probabilities
            total = squared_distances.sum()
            if total > 0:
                probabilities = squared_distances / total
            else:
                # All observations are at existing centers, choose uniformly
                probabilities = np.ones(len(self.observations)) / len(self.observations)

            next_idx = np.random.choice(len(self.observations), p=probabilities)
            next_point = RiemannianPoint(self, self.observations[next_idx])
            centers.append(RiemannianCenter(next_point))

        return centers

    def compute_clusters(self, centers: list[RiemannianCenter]) -> None:
        """Assign observations to their nearest center.

        For continuous manifolds, this is primarily for compatibility with
        the Space interface. The actual clustering is implicit in calculate_energy.

        Args:
            centers: List of cluster centers.
        """
        # For continuous manifolds, clustering is implicit
        # This could be extended to track cluster assignments if needed
        pass

    def calculate_energy(self, centers: list[RiemannianCenter]) -> float:
        """Calculate the k-means energy for the given centers.

        The energy is the sum of squared distances from each observation
        to its nearest center, divided by the number of observations.

        Args:
            centers: List of cluster centers.

        Returns:
            The k-means energy (average squared distance to nearest center).

        Raises:
            ValueError: If no observations are available or centers list is empty.
        """
        if len(self.observations) == 0:
            raise ValueError("No observations available for energy calculation")

        if len(centers) == 0:
            raise ValueError("Centers list cannot be empty")

        total_energy = 0.0

        # For each observation, find squared distance to nearest center
        for obs_coords in self.observations:
            obs_point = RiemannianPoint(self, obs_coords)
            min_dist_sq = min(
                self.distance(center, obs_point) ** 2 for center in centers
            )
            total_energy += min_dist_sq

        return total_energy / len(self.observations)
