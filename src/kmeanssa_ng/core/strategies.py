"""Robustification strategies for the simulated annealing algorithm."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from .abstract import Center
    from .simulated_annealing import SimulatedAnnealing


T_Result = TypeVar("T_Result")


class RobustificationStrategy(ABC, Generic[T_Result]):
    """Abstract base class for robustification strategies."""

    @abstractmethod
    def initialize(self, sa: SimulatedAnnealing) -> None:
        """Initialize the strategy before the simulation loop starts.

        Args:
            sa: The SimulatedAnnealing instance.
        """
        raise NotImplementedError

    @abstractmethod
    def collect(self, sa: SimulatedAnnealing) -> None:
        """Collect data during the robustification phase of the simulation.

        Args:
            sa: The SimulatedAnnealing instance.
        """
        raise NotImplementedError

    @abstractmethod
    def get_result(self) -> T_Result:
        """Compute and return the final result after the simulation.

        Returns:
            The final result of the strategy.
        """
        raise NotImplementedError


class MinimizeEnergyStrategy(RobustificationStrategy[list["Center"]]):
    """Strategy to find centers that minimize the k-means energy."""

    def initialize(self, sa: "SimulatedAnnealing") -> None:
        """Initialize with current centers and their energy."""
        self._best_centers = sa._clone_centers(sa.centers)
        self._best_energy = sa.space.calculate_energy_graph(self._best_centers)

    def collect(self, sa: "SimulatedAnnealing") -> None:
        """If current centers have lower energy, save them."""
        new_energy = sa.space.calculate_energy_graph(sa.centers)
        if new_energy < self._best_energy:
            self._best_centers = sa._clone_centers(sa.centers)
            self._best_energy = new_energy

    def get_result(self) -> list["Center"]:
        """Return the list of centers with the minimum energy found."""
        return self._best_centers
