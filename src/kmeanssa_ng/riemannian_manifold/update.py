"""Update strategies for RiemannianManifold."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ..core.strategies.lloyd_update import LloydUpdateStrategy

if TYPE_CHECKING:
    from .center import RiemannianCenter
    from .point import RiemannianPoint
    from .space import RiemannianManifold


class KarcherFrechetMean(LloydUpdateStrategy):
    """Update strategy that computes the new center as the Fréchet mean
    (Karcher mean) of the points in the cluster.

    The mean is computed by the intrinsic Karcher iteration
    ``mean <- exp_mean(average of log_mean(x_i))``, driven entirely by the
    space's own ``exp``/``log`` maps. This works on every space of the
    package -- including quotient surfaces like Bolza, whose ``log`` picks
    the nearest copy under the group action, which a chart-level estimator
    (e.g. geomstats' ``FrechetMean`` on the underlying manifold) would
    silently ignore.

    The iteration is deterministic, fast and locally exact, but it needs the
    space to provide ``exp``/``log`` and it descends to the nearest critical
    point. For a stochastic, globally-minded alternative that works on any
    metric space of the package (graphs included), see
    ``SimulatedAnnealingFrechetMean``.

    Args:
        max_iter: Maximum number of Karcher iterations.
        tol: Stop when the Riemannian norm of the mean update step falls
            below this threshold.
    """

    def __init__(self, max_iter: int = 64, tol: float = 1e-9):
        self.max_iter = max_iter
        self.tol = tol

    def update(
        self, points: list[RiemannianPoint], space: "RiemannianManifold"
    ) -> "RiemannianCenter":
        """Compute the new center for a given cluster of points.

        Args:
            points: A list of points belonging to a single cluster.
            space: The Riemannian manifold in which the points and center exist.

        Returns:
            The new center for the cluster.
        """
        if not points:
            return None

        # Extract coordinates from RiemannianPoint objects
        points_coords = []
        for p in points:
            if not isinstance(p, space.get_point_type()):
                raise TypeError("All points must be RiemannianPoint instances.")
            points_coords.append(p.coordinates)

        coords = np.asarray(points_coords, dtype=float)

        # Karcher iteration: follow the mean of the log directions until the
        # step vanishes (exp/log are batched over leading axes).
        mean = coords[0].copy()
        for _ in range(self.max_iter):
            logs = np.asarray(space.log(mean, coords))
            step = logs.mean(axis=0)
            if float(np.asarray(space.norm(mean, step))) < self.tol:
                break
            mean = np.asarray(space.exp(mean, step))

        # Create a RiemannianCenter from the mean coordinates
        return space.center_from_point(space.get_point_type()(space, mean))
