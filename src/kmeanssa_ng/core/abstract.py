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

    @abstractmethod
    def sample_points(self, n: int) -> list[Point]:
        """Sample random points from the space.

        Args:
            n: Number of points to sample.

        Returns:
            List of n randomly sampled points.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_centers(self, k: int) -> list[Center]:
        """Sample random centers from the space.

        Args:
            k: Number of centers to sample.

        Returns:
            List of k randomly sampled centers.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_kpp_centers(self, k: int) -> list[Center]:
        """Sample centers using k-means++ initialization.
        
        The k-means++ algorithm chooses initial centers to be spread out,
        improving convergence compared to random initialization.

        Args:
            k: Number of centers to sample.

        Returns:
            List of k centers sampled using k-means++ procedure.
        """
        raise NotImplementedError

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
    def calculate_energy_graph(self, centers: list[Center]) -> float:
        """Calculate the k-means energy (distortion) for given centers.

        The energy is the sum of squared distances from each point
        to its nearest center.

        Args:
            centers: List of cluster centers.

        Returns:
            The total energy (sum of squared distances).
        """
        raise NotImplementedError
