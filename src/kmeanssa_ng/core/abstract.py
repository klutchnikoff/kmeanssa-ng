"""Abstract base classes for metric spaces and k-means clustering.

This module defines the core abstractions for implementing k-means clustering
on arbitrary metric spaces. The design follows a clear separation of concerns:
- Point: Represents a point in a metric space
- Center: A movable point used as cluster center
- Space: The metric space containing points and centers
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class Point(ABC):
    """Abstract base class for points in a metric space.

    A point is an element of a metric space with a fixed location.
    Concrete implementations must define which space the point belongs to.
    """

    @property
    @abstractmethod
    def space(self) -> Space:
        """The metric space this point belongs to.

        Returns:
            The Space instance containing this point.
        """
        raise NotImplementedError


class Center(Point):
    """Abstract base class for cluster centers.

    A center is a special type of point that can move through the space
    using two mechanisms:
    - Brownian motion: Random exploration
    - Drift: Directed movement toward a target point

    This class is used in simulated annealing for k-means clustering.
    """

    @abstractmethod
    def brownian_motion(self, time_to_travel: float) -> None:
        """Perform random Brownian motion in the space.

        Args:
            time_to_travel: Time parameter controlling the magnitude of motion.
                Typical distance traveled is proportional to sqrt(time_to_travel).
        """
        raise NotImplementedError

    @abstractmethod
    def drift(self, target_point: Point, prop_to_travel: float) -> None:
        """Move toward a target point.

        Args:
            target_point: The point to move toward.
            prop_to_travel: Proportion of the distance to travel (between 0 and 1).
                0 means no movement, 1 means move all the way to target.
        """
        raise NotImplementedError


class Space(ABC):
    """Abstract base class for metric spaces.

    A metric space provides:
    - Distance computation between points
    - Sampling of random points and centers
    - Cluster computation and energy calculation
    """

    @abstractmethod
    def distance(self, p1: Point, p2: Point) -> float:
        """Compute the distance between two points.

        Args:
            p1: First point.
            p2: Second point.

        Returns:
            The distance between p1 and p2.
        """
        raise NotImplementedError

    def sample_points(self, n: int, strategy) -> list[Point]:
        """Sample n points using the specified sampling strategy.

        Args:
            n: Number of points to sample
            strategy: Sampling strategy defining the probability distribution.
                     Must be a SamplingStrategy instance specific to the space type.

        Returns:
            List of n sampled points

        Example:
            ```python
            # For quantum graphs
            from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
            points = graph.sample_points(100, strategy=UniformNodeSampling())

            # For Riemannian manifolds
            from kmeanssa_ng.riemannian_manifold.sampling import UniformManifoldSampling
            points = manifold.sample_points(100, strategy=UniformManifoldSampling())
            ```

        Note:
            The strategy parameter is REQUIRED to avoid ambiguity about which
            probability distribution to use. Each space type has its own
            specific sampling strategies in space-specific modules.
        """
        return strategy.sample(self, n)

    @abstractmethod
    def compute_clusters(self, centers: list[Center]) -> None:
        """Assign points to their nearest center.

        This method typically updates internal state or annotations
        indicating which cluster each point belongs to.

        Args:
            centers: List of cluster centers.
        """
        raise NotImplementedError

    @abstractmethod
    def calculate_energy(self, centers: list[Center]) -> float:
        """Calculate the k-means energy (distortion) for given centers.

        The energy is the sum of squared distances from each point
        to its nearest center.

        Args:
            centers: List of cluster centers.

        Returns:
            The total energy (sum of squared distances).
        """
        raise NotImplementedError

    @abstractmethod
    def distances_from_centers(self, centers: list[Center], target: Point):
        """Compute distances from multiple centers to a single target point.

        This method is used by the simulated annealing algorithm to efficiently
        find the nearest center to a given observation point.

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
            ```
        """
        raise NotImplementedError

    @abstractmethod
    def center_from_point(self, point: Point) -> Center:
        """Create a Center object from a Point object."""
        raise NotImplementedError
