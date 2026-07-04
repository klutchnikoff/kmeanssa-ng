"""Implements the Space abstract base class for Riemannian Manifolds using geomstats."""

from __future__ import annotations

import numpy as np
from geomstats.geometry.hypersphere import Hypersphere

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

    # ------------------------------------------------------------------
    # Geodesic operations on ambient coordinate arrays.
    # These let geodesic strategies (e.g. epsilon-net construction) drive the
    # manifold without touching geomstats or the RiemannianPoint wrapper.
    # ------------------------------------------------------------------
    @property
    def dim(self) -> int:
        """Intrinsic dimension of the manifold."""
        return self.manifold.dim

    @property
    def is_sphere(self) -> bool:
        """Whether the underlying manifold is a hypersphere."""
        return isinstance(self.manifold, Hypersphere)

    def exp(self, base_point: np.ndarray, tangent_vec: np.ndarray) -> np.ndarray:
        """Riemannian exponential: retract ``tangent_vec`` from ``base_point``.

        Batched over leading axes; arrays are ambient (extrinsic) coordinates.
        """
        return np.asarray(self.manifold.metric.exp(tangent_vec, base_point))

    def log(self, base_point: np.ndarray, point: np.ndarray) -> np.ndarray:
        """Riemannian logarithm: tangent at ``base_point`` pointing to ``point``.

        Inverse of :meth:`exp`; its Riemannian norm is the geodesic distance.
        """
        return np.asarray(self.manifold.metric.log(point, base_point))

    def norm(self, base_point: np.ndarray, tangent_vec: np.ndarray) -> np.ndarray:
        """Riemannian norm of ``tangent_vec`` in the tangent space at ``base_point``."""
        return np.asarray(self.manifold.metric.norm(tangent_vec, base_point))

    def to_tangent(self, base_point: np.ndarray, ambient_vec: np.ndarray) -> np.ndarray:
        """Project an ambient vector onto the tangent space at ``base_point``."""
        return np.asarray(self.manifold.to_tangent(ambient_vec, base_point))

    @property
    def shape(self) -> tuple:
        """Ambient shape of a single point on the manifold."""
        return self.manifold.shape

    def embed(self, points: np.ndarray) -> np.ndarray:
        """Ambient coordinates of ``points`` (for neighbour search).

        Points are already extrinsic here, so this is the identity; it exists so
        geodesic strategies can stay agnostic to the coordinate representation.
        """
        return np.asarray(points)

    def random_uniform(
        self, n: int, random_state: int | np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample ``n`` points uniformly, reproducibly from ``random_state``.

        Returns an ``(n, dim + 1)`` array of ambient coordinates.
        """
        rng = np.random.default_rng(random_state)
        if self.is_sphere:
            # Gaussian in the ambient space, normalised -> uniform on the sphere.
            x = rng.standard_normal((n, self.dim + 1))
            return x / np.linalg.norm(x, axis=1, keepdims=True)
        # TODO: area-uniform sampling for other manifolds. Non-compact domains
        # (e.g. hyperbolic) also need a bounded region + soft confinement; this
        # is handled at the epsilon-net strategy level.
        raise NotImplementedError(
            "random_uniform is currently implemented for hyperspheres only, not "
            f"{type(self.manifold).__name__}."
        )

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
        for obs in self.observations:
            if isinstance(obs, RiemannianPoint):
                obs_point = obs
            else:  # It's a numpy array (coordinates)
                obs_point = RiemannianPoint(self, obs)

            min_dist_sq = min(
                self.distance(center, obs_point) ** 2 for center in centers
            )
            total_energy += min_dist_sq

        return total_energy / len(self.observations)

    def get_point_type(self) -> type[RiemannianPoint]:
        """Return the type of points in this space."""
        return RiemannianPoint


class Sphere(RiemannianManifold):
    """Hypersphere with closed-form geodesic operations.

    The unit sphere's exponential, logarithm, distance and tangent projection
    have simple closed forms. Overriding them here avoids geomstats' generic
    per-call overhead (~7x on a single point), which dominates the manifold
    annealing loop; the results match geomstats to machine precision. Every
    override is vectorised over leading axes, matching the base class.
    """

    def distance(self, point1: RiemannianPoint, point2: RiemannianPoint) -> float:
        inner = np.clip(np.dot(point1.coordinates, point2.coordinates), -1.0, 1.0)
        return float(np.arccos(inner))

    def exp(self, base_point: np.ndarray, tangent_vec: np.ndarray) -> np.ndarray:
        base_point = np.asarray(base_point, dtype=float)
        tangent_vec = np.asarray(tangent_vec, dtype=float)
        # Project onto the tangent space first (as geomstats' exp does), so any
        # radial component does not push the result off the sphere.
        radial = np.sum(tangent_vec * base_point, axis=-1, keepdims=True)
        tangent_vec = tangent_vec - radial * base_point
        norm = np.linalg.norm(tangent_vec, axis=-1, keepdims=True)
        # cos|v| * p + sin|v| * v/|v|, with the |v| -> 0 limit equal to p.
        direction = np.divide(
            tangent_vec, norm, out=np.zeros_like(tangent_vec), where=norm > 1e-12
        )
        return np.cos(norm) * base_point + np.sin(norm) * direction

    def log(self, base_point: np.ndarray, point: np.ndarray) -> np.ndarray:
        base_point = np.asarray(base_point, dtype=float)
        point = np.asarray(point, dtype=float)
        inner = np.clip(np.sum(base_point * point, axis=-1, keepdims=True), -1.0, 1.0)
        proj = point - inner * base_point  # tangent component of `point` at base
        proj_norm = np.linalg.norm(proj, axis=-1, keepdims=True)
        direction = np.divide(
            proj, proj_norm, out=np.zeros_like(proj), where=proj_norm > 1e-12
        )
        return np.arccos(inner) * direction  # scaled by the geodesic distance

    def norm(self, base_point: np.ndarray, tangent_vec: np.ndarray) -> np.ndarray:
        # A tangent vector's Riemannian norm on the unit sphere is its ambient norm.
        return np.linalg.norm(np.asarray(tangent_vec, dtype=float), axis=-1)

    def to_tangent(self, base_point: np.ndarray, ambient_vec: np.ndarray) -> np.ndarray:
        base_point = np.asarray(base_point, dtype=float)
        ambient_vec = np.asarray(ambient_vec, dtype=float)
        inner = np.sum(ambient_vec * base_point, axis=-1, keepdims=True)
        return ambient_vec - inner * base_point
