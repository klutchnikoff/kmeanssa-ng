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
    natural volume measure induced by the Riemannian metric, through the
    space's own generator-driven sampler (``random_uniform``), so results
    are reproducible from ``random_state``.

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
        Only spaces implementing ``random_uniform`` support this strategy
        (hyperspheres, the Bolza surface). Others raise
        ``NotImplementedError``: uniform-in-volume does not exist on
        non-compact manifolds such as hyperbolic space, and geomstats'
        samplers draw from the global RNG, which would break
        reproducibility.
    """

    def sample(self, space: "RiemannianManifold", n: int) -> list["RiemannianPoint"]:
        """Sample n points uniformly from the manifold.

        Args:
            space: The Riemannian manifold to sample from.
            n: Number of points to sample.

        Returns:
            List of n points sampled uniformly using the volume measure.

        Raises:
            NotImplementedError: If the space has no generator-driven
                uniform sampler.
        """
        rng = self._get_rng()
        # Draw through the space's own generator-driven sampler so that
        # ``random_state`` is honoured. There is deliberately no geomstats
        # fallback: manifold.random_uniform/random_point draw from the global
        # numpy RNG (unseedable per run, hence non-reproducible), and uniform-
        # in-volume does not even exist on non-compact manifolds such as
        # hyperbolic space. Spaces without a generator-driven uniform sampler
        # raise NotImplementedError here.
        coords = space.random_uniform(n, rng)

        # Create RiemannianPoint objects
        points = []
        for i in range(n):
            point_coords = coords[i] if coords.ndim > 1 else coords
            points.append(RiemannianPoint(space, point_coords))

        return points
