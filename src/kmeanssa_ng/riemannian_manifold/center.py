"""Center class for Riemannian manifolds with Brownian motion and drift."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng

from kmeanssa_ng.core.abstract import Center as AbstractCenter

from .point import RiemannianPoint


class RiemannianCenter(RiemannianPoint, AbstractCenter):
    """A movable cluster center on a Riemannian manifold.

    Centers can perform:
    - Brownian motion: Random walk using Geomstats BrownianMotion
    - Drift: Directed movement toward target points along geodesics

    Attributes:
        space: The Riemannian manifold this center belongs to.
        coordinates: The coordinates of the center on the manifold.

    Example:
        ```python
        from geomstats.geometry.hypersphere import Hypersphere
        manifold = Hypersphere(dim=2)
        space = RiemannianManifold(manifold)
        point = RiemannianPoint(space, coordinates=np.array([1.0, 0.0, 0.0]))
        center = RiemannianCenter(point)
        center.brownian_motion(0.1)
        center.drift(target_point, 0.5)
        ```
    """

    def __init__(
        self,
        point: RiemannianPoint,
        rng: Generator | None = None,
    ) -> None:
        """Initialize a center from a point.

        Args:
            point: The initial point location.
            rng: Random number generator. If None, creates a new default_rng().
        """
        super().__init__(point.space, point.coordinates)
        self._rng = rng if rng is not None else default_rng()

    def brownian_motion(self, time_to_travel: float) -> None:
        """Perform Brownian motion on the Riemannian manifold.

        Implements a simple Brownian motion by:
        1. Generating a random tangent vector
        2. Scaling it by sqrt(time_to_travel)
        3. Moving along the geodesic in that direction

        Args:
            time_to_travel: Time parameter (distance ~ sqrt(time)).

        Raises:
            ValueError: If time_to_travel is negative or not numeric.
        """
        # Validate time_to_travel
        try:
            time_float = float(time_to_travel)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"time_to_travel must be a number, got {type(time_to_travel).__name__}"
            ) from e

        if time_float < 0:
            raise ValueError(f"time_to_travel must be non-negative, got {time_float}")

        if time_float == 0:
            return  # No movement

        # The Brownian increment is sqrt(time) * V, where V is a standard
        # Gaussian tangent vector (iid N(0,1) components in a metric-orthonormal
        # frame). random_tangent already returns that full Gaussian vector --
        # random in both direction *and* length -- so it is the whole
        # increment; multiplying it by an extra scalar N(0,1) would double-count
        # the noise (a product of two normals: kurtosis 9 instead of 3, wrong
        # radial law at finite step_size) and diverge from the 1-D graph step,
        # which draws a single Gaussian. It comes from the space (spaces without
        # a metric-orthonormal frame refuse) and from this center's own
        # generator, not geomstats' global RNG, so runs stay reproducible.
        tangent_vec = self.space.random_tangent(self.coordinates, self._rng)

        # Move along the exponential map (self.space dispatches to a closed form
        # on known manifolds, e.g. the sphere, and to geomstats otherwise).
        self.coordinates = self.space.exp(
            self.coordinates, np.sqrt(time_float) * tangent_vec
        )

    def drift(self, target_point: RiemannianPoint, prop_to_travel: float) -> None:
        """Move toward a target point along the geodesic.

        Moves a proportion of the geodesic distance to the target point.

        Args:
            target_point: The point to move toward.
            prop_to_travel: Proportion of distance to travel (0 to 1).

        Raises:
            ValueError: If target_point is None, prop_to_travel is not numeric,
                or prop_to_travel is not in [0, 1].

        Note:
            On manifolds where geodesics are not unique (e.g., antipodal points
            on a sphere), the drift may not move the center. This is a known
            limitation of Geomstats' geodesic computation. In practice, this
            rarely occurs due to the measure-zero probability of exact antipodal
            configurations, and the Brownian motion component of the simulated
            annealing algorithm provides thermal agitation to escape such
            degenerate configurations.
        """
        # Validate target_point
        if target_point is None:
            raise ValueError("target_point cannot be None")

        # Validate prop_to_travel
        try:
            prop_float = float(prop_to_travel)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"prop_to_travel must be a number, got {type(prop_to_travel).__name__}"
            ) from e

        if prop_float < 0 or prop_float > 1:
            raise ValueError(f"prop_to_travel must be in [0, 1], got {prop_float}")

        if prop_float == 0:
            return  # No movement

        # Move a proportion prop_float along the geodesic toward the target:
        # exp_p(prop * log_p(target)). This equals geomstats' geodesic evaluated at
        # prop, without building a geodesic callable per call, and dispatches to a
        # closed form on known manifolds (e.g. the sphere) via self.space.
        log_vec = self.space.log(self.coordinates, target_point.coordinates)
        new_coords = self.space.exp(self.coordinates, prop_float * log_vec)

        self.coordinates = new_coords

    def clone(self) -> RiemannianCenter:
        """Create an independent copy of this center.

        The cloned center shares the same manifold space but has
        independent coordinates. This is much faster than deepcopy.

        Returns:
            A new RiemannianCenter with the same location but independent state.

        Example:
            ```python
            original = RiemannianCenter(...)
            copy = original.clone()
            original.brownian_motion(0.1)  # Doesn't affect copy
            ```
        """
        # Create new center bypassing validation for speed
        new_center = object.__new__(RiemannianCenter)
        new_center._space = self._space
        new_center.coordinates = self.coordinates.copy()
        new_center._rng = self._rng
        return new_center

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"RiemannianCenter(coordinates={self.coordinates})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        manifold_name = (
            self._space.manifold.__class__.__name__
            if hasattr(self._space, "manifold")
            else "Unknown"
        )
        return f"Center on {manifold_name} at {self.coordinates}"
