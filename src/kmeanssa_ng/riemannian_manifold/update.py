"""Update strategies for RiemannianManifold."""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

from ..core.strategies.update import UpdateStrategy
from geomstats.learning.frechet_mean import FrechetMean

if TYPE_CHECKING:
    from ..core.abstract import Center, Point, Space
    from .point import RiemannianPoint
    from .center import RiemannianCenter


class FrechetMeanUpdate(UpdateStrategy):
    """Update strategy that computes the new center as the Fréchet mean
    (Karcher mean) of the points in the cluster.
    """

    def update(self, points: list[RiemannianPoint], space: "RiemannianManifold") -> "RiemannianCenter":
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

        # Convert to numpy array for geomstats
        points_coords_array = np.array(points_coords)

        # Compute the Fréchet mean (Karcher mean) using geomstats
        mean_estimator = FrechetMean(space=space.manifold)
        mean_estimator.fit(points_coords_array)
        mean_coords = mean_estimator.estimate_

        # Create a RiemannianCenter from the mean coordinates
        return space.center_from_point(space.get_point_type()(space, mean_coords))

# I will also need to add get_point_type to RiemannianManifold
