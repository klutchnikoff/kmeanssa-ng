"""Sampling strategies for Riemannian manifolds.

This module provides sampling strategies specific to Riemannian manifolds,
allowing different probability distributions for selecting points on the manifold.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core.strategies.sampling import SamplingStrategy

if TYPE_CHECKING:
    from ..core.abstract import Point, Space


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

    def sample(self, space: Space, n: int) -> list[Point]:
        """Sample n points uniformly from the manifold.

        Args:
            space: The Riemannian manifold to sample from.
            n: Number of points to sample.

        Returns:
            List of n points sampled uniformly using the volume measure.
        """
        return space._sample_uniform(n)
