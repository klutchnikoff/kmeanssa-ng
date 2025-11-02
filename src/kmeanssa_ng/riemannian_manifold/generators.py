"""Factory functions to generate common Riemannian Manifold spaces."""

from __future__ import annotations

from geomstats.geometry.hypersphere import Hypersphere

from .space import RiemannianManifold


def create_sphere(dim: int = 2, **kwargs) -> RiemannianManifold:
    """Create a hypersphere space wrapped in a RiemannianManifold.

    Creates a hypersphere S^dim embedded in R^(dim+1). The sphere is equipped
    with the standard round metric inherited from the Euclidean ambient space.

    Args:
        dim: Dimension of the sphere (default: 2 for the standard 2-sphere S^2).
            - dim=1: Circle S^1 in R^2
            - dim=2: Standard sphere S^2 in R^3
            - dim=3: 3-sphere S^3 in R^4
            etc.
        **kwargs: Additional arguments passed to Hypersphere constructor.

    Returns:
        A RiemannianManifold wrapping the hypersphere.

    Raises:
        ValueError: If dim is not a positive integer (raised by Geomstats).
        TypeError: If dim is not a valid type (raised by Geomstats).

    Example:
        ```python
        # Create a 2-sphere (surface of a ball in 3D)
        sphere = create_sphere(dim=2)
        points = sphere.sample_points(100)

        # Create a circle
        circle = create_sphere(dim=1)
        ```

    Note:
        The sphere S^dim is the set of unit vectors in R^(dim+1):
        S^dim = {x ∈ R^(dim+1) : ||x|| = 1}
    """
    # Geomstats handles validation
    #  Note: intrinsic coordinates cause issues with belongs() and BrownianMotion
    # For now, use extrinsic (default) coordinates
    sphere_manifold = Hypersphere(dim=dim, **kwargs)
    return RiemannianManifold(sphere_manifold)


def create_hyperbolic_space(dim: int = 2, **kwargs) -> RiemannianManifold:
    """Create a hyperbolic space wrapped in a RiemannianManifold.

    Creates the hyperboloid model of hyperbolic space H^dim.

    Args:
        dim: Dimension of the hyperbolic space (default: 2).
        **kwargs: Additional arguments passed to Hyperboloid constructor.

    Returns:
        A RiemannianManifold wrapping the hyperbolic space.

    Raises:
        ValueError: If dim is not a positive integer (raised by Geomstats).
        TypeError: If dim is not a valid type (raised by Geomstats).

    Example:
        ```python
        # Create 2D hyperbolic space (Poincaré disk)
        hyperbolic = create_hyperbolic_space(dim=2)
        points = hyperbolic.sample_points(100)
        ```

    Note:
        This uses the hyperboloid model of hyperbolic geometry.
    """
    from geomstats.geometry.hyperboloid import Hyperboloid

    # Geomstats handles validation
    hyperbolic_manifold = Hyperboloid(dim=dim, **kwargs)
    return RiemannianManifold(hyperbolic_manifold)
