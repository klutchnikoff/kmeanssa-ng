"""Sampling strategies for Riemannian manifolds.

This module provides sampling strategies specific to Riemannian manifolds,
allowing different probability distributions for selecting points on the manifold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.strategies.sampling import SamplingStrategy
from .point import RiemannianPoint

if TYPE_CHECKING:
    from .space import RiemannianManifold


class UniformManifoldSampling(SamplingStrategy):
    """Uniform sampling with respect to the Riemannian volume measure.

    This strategy samples points uniformly from the manifold using the
    natural volume measure induced by the Riemannian metric.

    For standard manifolds (sphere, hyperbolic space, etc.), this uses
    the geomstats library's built-in uniform sampling methods.

    Example:
        ```python
        from kmeanssa_ng.riemannian_manifold import create_sphere
        from kmeanssa_ng.riemannian_manifold.sampling import UniformManifoldSampling

        # Create a 2-sphere
        sphere = create_sphere(dim=2)

        # Sample 100 points uniformly on the sphere
        strategy = UniformManifoldSampling()
        points = sphere.sample_points(100, strategy=strategy)
        ```

    Note:
        The actual sampling implementation depends on the geomstats library
        and the specific manifold being used. Most standard manifolds provide
        efficient uniform sampling methods.
    """

    def sample(self, space: "RiemannianManifold", n: int) -> list["RiemannianPoint"]:
        """Sample n points uniformly from the manifold.

        Args:
            space: The Riemannian manifold to sample from.
            n: Number of points to sample.

        Returns:
            List of n points sampled uniformly using the volume measure.
        """
        rng = self._get_rng()
        # Draw through the space's own generator-driven sampler so that
        # ``random_state`` is honoured. geomstats' manifold.random_uniform /
        # random_point draw from the global numpy RNG and take no seed.
        try:
            coords = space.random_uniform(n, rng)
        except NotImplementedError:
            # No generator-driven uniform sampler for this manifold yet: fall back
            # to geomstats, which draws from the global RNG (not reproducible).
            if hasattr(space.manifold, "random_uniform"):
                coords = space.manifold.random_uniform(n_samples=n)
            else:
                coords = space.manifold.random_point(n_samples=n)

        # Create RiemannianPoint objects
        points = []
        for i in range(n):
            point_coords = coords[i] if coords.ndim > 1 else coords
            points.append(RiemannianPoint(space, point_coords))

        return points
