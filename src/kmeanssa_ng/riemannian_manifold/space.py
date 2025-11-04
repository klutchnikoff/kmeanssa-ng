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

    def center_from_point(self, point: RiemannianPoint) -> RiemannianCenter:
        """Create a RiemannianCenter object from a RiemannianPoint object."""
        return RiemannianCenter(point)

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

    def distances_from_centers(
        self, centers: list[RiemannianCenter], target: RiemannianPoint
    ) -> np.ndarray:
        """Compute distances from multiple centers to a single target point.

        Args:
            centers: List of k centers to compute distances from.
            target: The target point.

        Returns:
            Array of shape (k,) with distances from each center to target.

        Example:
            ```python
            centers = space.sample_centers(5)
            target = space.sample_points(1)[0]
            distances = space.distances_from_centers(centers, target)
            closest_idx = np.argmin(distances)
            closest_center = centers[closest_idx]
            ```
        """
        distances = np.empty(len(centers))
        for i, center in enumerate(centers):
            distances[i] = self.distance(center, target)
        return distances

    def calculate_energy(
        self, centers: list[RiemannianCenter], how: str = "obs"
    ) -> float:
        """Calculate the k-means energy for the given centers.

        The energy is the sum of squared distances from each observation
        to its nearest center, divided by the number of observations.

        Args:
            centers: List of cluster centers.
            how: Energy calculation mode. For Riemannian manifolds, only "obs"
                mode is supported (uses sampled observations). The "uniform"
                mode is not applicable as there's no natural notion of uniform
                distribution over all points of a continuous manifold.
                This parameter is kept for API compatibility but ignored.

        Returns:
            The k-means energy (average squared distance to nearest center).

        Raises:
            ValueError: If no observations are available or centers list is empty.

        Note:
            Unlike discrete spaces (e.g., quantum graphs), continuous manifolds
            don't support a "uniform" distribution over all possible points.
            Energy is always computed using the sampled observations.
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
