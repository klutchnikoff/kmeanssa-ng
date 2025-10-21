"""Tests for abstract base classes."""

import pytest

from kmeanssa_ng.core.abstract import Center, Point, Space


class DummySpace(Space):
    """A concrete implementation of Space for testing purposes."""

    def distance(self, p1: Point, p2: Point) -> float:
        return super().distance(p1, p2)

    def sample_points(self, n: int) -> list[Point]:
        return super().sample_points(n)

    def sample_centers(self, k: int) -> list[Center]:
        return super().sample_centers(k)

    def sample_kpp_centers(self, k: int) -> list[Center]:
        return super().sample_kpp_centers(k)

    def compute_clusters(self, centers: list[Center]) -> None:
        return super().compute_clusters(centers)

    def calculate_energy_graph(self, centers: list[Center]) -> float:
        return super().calculate_energy_graph(centers)


class DummyPoint(Point):
    """A concrete implementation of Point for testing."""

    @property
    def space(self) -> Space:
        return super().space


class DummyCenter(Center):
    """A concrete implementation of Center for testing."""

    @property
    def space(self) -> Space:
        return DummySpace()

    def brownian_motion(self, time_to_travel: float) -> None:
        return super().brownian_motion(time_to_travel)

    def drift(self, target_point: Point, prop_to_travel: float) -> None:
        return super().drift(target_point, prop_to_travel)


def test_abstract_methods_raise_not_implemented():
    """Verify that calling abstract methods directly raises NotImplementedError."""
    space = DummySpace()
    point = DummyPoint()
    center = DummyCenter()

    with pytest.raises(NotImplementedError):
        point.space

    with pytest.raises(NotImplementedError):
        center.brownian_motion(1.0)

    with pytest.raises(NotImplementedError):
        center.drift(point, 0.5)

    with pytest.raises(NotImplementedError):
        space.distance(point, point)

    with pytest.raises(NotImplementedError):
        space.sample_points(1)

    with pytest.raises(NotImplementedError):
        space.sample_centers(1)

    with pytest.raises(NotImplementedError):
        space.compute_clusters([center])

    with pytest.raises(NotImplementedError):
        space.calculate_energy_graph([center])
