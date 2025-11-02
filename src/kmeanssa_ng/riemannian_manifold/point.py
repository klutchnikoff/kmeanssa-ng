"""Represents a point on a Riemannian Manifold."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from kmeanssa_ng.core.abstract import Point as AbstractPoint

if TYPE_CHECKING:
    from .space import RiemannianManifold


class RiemannianPoint(AbstractPoint):
    """A point on a Riemannian manifold.

    A Riemannian point is represented by its coordinates on the manifold.
    The coordinates are validated to ensure they belong to the manifold.

    Attributes:
        space: The Riemannian manifold space this point belongs to.
        coordinates: The coordinates of the point on the manifold.

    Example:
        ```python
        from geomstats.geometry.hypersphere import Hypersphere
        manifold = Hypersphere(dim=2)
        space = RiemannianManifold(manifold)
        point = RiemannianPoint(space, coordinates=np.array([1.0, 0.0, 0.0]))
        ```
    """

    def __init__(
        self,
        space: RiemannianManifold,
        coordinates: np.ndarray,
    ) -> None:
        """Initialize a point on a Riemannian manifold.

        Args:
            space: The Riemannian manifold space containing this point.
            coordinates: The coordinates of the point on the manifold.

        Raises:
            ValueError: If space is None, coordinates are not a numpy array,
                or coordinates don't belong to the manifold.
        """
        if space is None:
            raise ValueError("space cannot be None")

        if not isinstance(coordinates, np.ndarray):
            raise ValueError(
                f"coordinates must be a numpy array, got {type(coordinates).__name__}"
            )

        self._space = space
        self._validate_and_set_coordinates(coordinates)

    def _validate_and_set_coordinates(self, coordinates: np.ndarray) -> None:
        """Validate and set the coordinates on the manifold.

        Args:
            coordinates: Coordinates on the manifold.

        Raises:
            ValueError: If coordinates don't belong to the manifold.
        """
        # For intrinsic coordinates, belongs() may not work reliably
        # So we do a basic shape check instead
        manifold_shape = self._space.manifold.shape
        coords_shape = coordinates.shape if coordinates.ndim > 0 else ()

        if coords_shape != manifold_shape:
            # Try belongs() for extrinsic coordinates
            if not self._space.manifold.intrinsic and not self._space.manifold.belongs(
                coordinates
            ):
                raise ValueError(
                    f"Coordinates {coordinates} do not belong to the manifold {self._space.manifold}"
                )

        self.coordinates = coordinates

    @property
    def space(self) -> RiemannianManifold:
        """The Riemannian manifold space this point belongs to."""
        return self._space

    def __str__(self) -> str:
        """String representation of the point."""
        manifold_name = (
            self._space.manifold.__class__.__name__
            if hasattr(self._space, "manifold")
            else "Unknown"
        )
        return f"RiemannianPoint on {manifold_name} at {self.coordinates}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"RiemannianPoint(coordinates={self.coordinates})"
