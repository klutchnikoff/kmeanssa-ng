"""Abstract base class for center update strategies in Lloyd's algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..abstract import Center, Point, Space


class LloydUpdateStrategy(ABC):
    """Abstract base class for strategies that update a cluster center.

    This is used in Lloyd's algorithm to compute the new center of a cluster
    based on the points assigned to it.
    """

    @abstractmethod
    def update(self, points: list[Point], space: Space) -> Center:
        """Compute the new center for a given cluster of points.

        Args:
            points: A list of points belonging to a single cluster.
            space: The metric space in which the points and center exist.

        Returns:
            The new center for the cluster.
        """
        raise NotImplementedError
