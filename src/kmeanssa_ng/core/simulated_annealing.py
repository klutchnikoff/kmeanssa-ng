"""Simulated annealing algorithm for k-means clustering on metric spaces."""

from __future__ import annotations

import random as rd
from copy import deepcopy
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .abstract import Center, Point, Space


class SimulatedAnnealing:
    """Simulated annealing for offline k-means clustering.

    This algorithm solves the k-means problem on arbitrary metric spaces using
    simulated annealing. Centers perform Brownian motion (exploration) and drift
    toward observations (exploitation), with temperature controlled by an
    inhomogeneous Poisson process.

    Attributes:
        space: The metric space containing the observations.
        k: Number of clusters.
        observations: List of points to cluster.
        centers: Current cluster centers.

    Example:
        ```python
        # Create observations and space
        space = QuantumGraph(...)
        points = space.sample_points(100)

        # Run simulated annealing
        sa = SimulatedAnnealing(points, k=5)
        centers = sa.run(robust_prop=0.1, initialization="kpp")
        ```
    """

    def __init__(
        self,
        observations: list[Point],
        k: int,
        lambda_param: int = 1,
        beta: float = 1.0,
        step_size: float = 0.1,
    ) -> None:
        """Initialize the simulated annealing algorithm.

        Args:
            observations: List of points to cluster, all in the same metric space.
            k: Number of clusters.
            lambda_param: Intensity parameter for Poisson process (must be > 0).
            beta: Inverse temperature parameter (must be > 0, higher = faster convergence).
            step_size: Time step for updating centers (must be > 0).

        Raises:
            ValueError: If observations is empty, k <= 0, points are in different spaces,
                or hyperparameters are invalid.
        """
        if not observations:
            raise ValueError("Observations must be a non-empty list of points.")
        if k <= 0:
            raise ValueError("Number of clusters 'k' must be greater than zero.")
        if any(obs.space != observations[0].space for obs in observations):
            raise ValueError("All observations must belong to the same metric space.")

        # Validate lambda_param
        try:
            lambda_float = float(lambda_param)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"lambda_param must be a number, got {type(lambda_param).__name__}"
            ) from e
        if lambda_float <= 0:
            raise ValueError(f"lambda_param must be positive, got {lambda_float}")

        # Validate beta
        try:
            beta_float = float(beta)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"beta must be a number, got {type(beta).__name__}"
            ) from e
        if beta_float <= 0:
            raise ValueError(f"beta must be positive, got {beta_float}")

        # Validate step_size
        try:
            step_size_float = float(step_size)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"step_size must be a number, got {type(step_size).__name__}"
            ) from e
        if step_size_float <= 0:
            raise ValueError(f"step_size must be positive, got {step_size_float}")

        self._space = observations[0].space
        self._observations = observations.copy()
        self._k = k
        self._lambda = lambda_float
        self._beta = beta_float
        self._step_size = step_size_float

        rd.shuffle(self._observations)
        self._centers: list[Center] = []

    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self._observations)

    @property
    def centers(self) -> list[Center]:
        """Current cluster centers."""
        return self._centers

    @property
    def space(self) -> Space:
        """Metric space containing the observations."""
        return self._space

    def _initialize_centers(self) -> list[Center]:
        """Initialize centers randomly in the metric space."""
        return self.space.sample_centers(self._k)

    def _initialize_kpp_centers(self) -> list[Center]:
        """Initialize centers using k-means++ procedure."""
        return self.space.sample_kpp_centers(self._k)

    def _initialize_times(self, n: int) -> np.ndarray:
        """Generate inhomogeneous Poisson times.

        Args:
            n: Number of time points to generate.

        Returns:
            Array of n+1 time points.
        """
        T = np.zeros(n + 1)
        poiss_sum = 0.0
        for i in range(n):
            poiss_sum += -1 / self._lambda * np.log(rd.random())
            T[i + 1] = np.sqrt(poiss_sum + 1) - 1
        return T

    def calculate_energy(self, centers: list[Center], points: list[Point]) -> float:
        """Calculate k-means energy for given centers.

        Args:
            centers: List of cluster centers.
            points: List of points.

        Returns:
            Average squared distance to nearest center.
        """
        energy = sum(
            min(self.space.distance(center, point) ** 2 for center in centers) for point in points
        )
        return energy / len(points)

    def run(
        self,
        robust_prop: float = 0.0,
        robust_points: list[Point] | None = None,
        initialization: str = "kpp",
        algorithm_version: str = "v1",
    ) -> list[Center]:
        """Run the simulated annealing algorithm.

        Args:
            robust_prop: Proportion of final iterations for robustification (0 to 1).
                The best centers from this period are returned.
            robust_points: Optional dataset for computing energy during robustification.
                Defaults to the observations.
            initialization: Initialization method ("kpp" for k-means++, "random" otherwise).
            algorithm_version: Algorithm variant ("v1" or "v2"). v1 interleaves drift
                and brownian motion, v2 performs all brownian motion first.

        Returns:
            List of k robust cluster centers.

        Raises:
            ValueError: If robust_prop not in [0, 1] or algorithm_version invalid.
        """
        if robust_prop < 0 or robust_prop > 1:
            raise ValueError("The proportion must be in [0,1]")
        if algorithm_version not in ["v1", "v2"]:
            raise ValueError("algorithm_version must be 'v1' or 'v2'")

        if robust_points is None:
            robust_points = self._observations

        i0 = int(np.floor((self.n - 1) * (1 - robust_prop)))

        # Initialize centers
        if initialization == "kpp":
            self._centers = self._initialize_kpp_centers()
        else:
            self._centers = self._initialize_centers()

        best_centers = deepcopy(self._centers)
        best_energy = self.space.calculate_energy_graph(best_centers)

        times = self._initialize_times(self.n)

        if algorithm_version == "v1":
            return self._run_v1(times, i0, best_centers, best_energy)
        else:
            return self._run_v2(times, i0, best_centers, best_energy)

    def _run_v1(
        self,
        times: np.ndarray,
        i0: int,
        best_centers: list[Center],
        best_energy: float,
    ) -> list[Center]:
        """Run algorithm version 1 (interleaved drift and brownian motion)."""
        time = 0.0

        for i, point in enumerate(self._observations):
            T = times[i]

            while time <= T - self._step_size:
                h = min(time + self._step_size, T) - time
                prop = min(h * self._beta * np.log(1 + time), 1)

                closest_center = None
                min_distance = float("inf")

                for center in self._centers:
                    center.brownian_motion(h)
                    dist = self.space.distance(center, point)
                    if dist < min_distance:
                        closest_center, min_distance = center, dist

                if closest_center is not None:
                    closest_center.drift(point, prop)

                time += h

            if i >= i0:
                new_energy = self.space.calculate_energy_graph(self._centers)
                if new_energy < best_energy:
                    best_centers = deepcopy(self._centers)
                    best_energy = new_energy

        return best_centers

    def _run_v2(
        self,
        times: np.ndarray,
        i0: int,
        best_centers: list[Center],
        best_energy: float,
    ) -> list[Center]:
        """Run algorithm version 2 (brownian motion then drift)."""
        time = 0.0

        for i, point in enumerate(self._observations, start=1):
            T = times[i]

            # Brownian motion phase
            while time <= T - self._step_size:
                h = min(time + self._step_size, T) - time
                for center in self._centers:
                    center.brownian_motion(h)
                time += h

            # Drift phase
            closest_center = None
            min_distance = float("inf")

            for center in self._centers:
                dist = self.space.distance(center, point)
                if dist < min_distance:
                    closest_center, min_distance = center, dist

            prop = min((times[i] - times[i - 1]) * self._beta * np.log(1 + time), 1)
            if closest_center is not None:
                closest_center.drift(point, prop)

            time = T

            if i >= i0:
                new_energy = self.space.calculate_energy_graph(self._centers)
                if new_energy < best_energy:
                    best_centers = deepcopy(self._centers)
                    best_energy = new_energy

        return best_centers
