"""Simulated annealing algorithm for k-means clustering on metric spaces."""

from __future__ import annotations

import random as rd
from typing import TYPE_CHECKING

import numpy as np

from .strategies import MinimizeEnergyStrategy

if TYPE_CHECKING:
    from .abstract import Center, Point, Space
    from .strategies import RobustificationStrategy


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

        # Run simulated annealing with the interleaved algorithm
        sa = SimulatedAnnealing(points, k=5)
        centers = sa.run_interleaved(robust_prop=0.1, initialization="kpp")
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
            raise ValueError(f"beta must be a number, got {type(beta).__name__}") from e
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

    def _clone_centers(self, centers: list[Center]) -> list[Center]:
        """Create independent copies of centers.

        Uses the clone() method if available (much faster than deepcopy),
        otherwise falls back to deepcopy for compatibility.

        Args:
            centers: List of centers to clone.

        Returns:
            List of cloned centers with independent state.
        """
        if hasattr(centers[0], "clone"):
            return [center.clone() for center in centers]
        else:
            # Fallback for custom Center implementations without clone()
            from copy import deepcopy

            return deepcopy(centers)

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
            min(self.space.distance(center, point) ** 2 for center in centers)
            for point in points
        )
        return energy / len(points)

    def _prepare_run(
        self,
        robust_prop: float,
        initialization: str,
        strategy: RobustificationStrategy | None,
    ) -> tuple[int, RobustificationStrategy]:
        """Prepare the simulation by initializing centers and strategy."""
        if robust_prop < 0 or robust_prop > 1:
            raise ValueError("The proportion must be in [0,1]")

        if strategy is None:
            strategy = MinimizeEnergyStrategy()

        i0 = int(np.floor((self.n - 1) * (1 - robust_prop)))

        if initialization == "kpp":
            self._centers = self._initialize_kpp_centers()
        else:
            self._centers = self._initialize_centers()

        strategy.initialize(self)
        return i0, strategy

    def run_interleaved(
        self,
        robust_prop: float = 0.0,
        initialization: str = "kpp",
        strategy: RobustificationStrategy | None = None,
    ):
        """Run SA with interleaved drift and brownian motion."""
        i0, strategy = self._prepare_run(robust_prop, initialization, strategy)
        times = self._initialize_times(self.n)
        time = 0.0

        for i, point in enumerate(self._observations):
            T = times[i]

            while time <= T - self._step_size:
                h = min(time + self._step_size, T) - time
                prop = min(h * self._beta * np.log(1 + time), 1)

                for center in self._centers:
                    center.brownian_motion(h)

                distances = self.space.batch_distances_from_centers(
                    self._centers, point
                )
                closest_idx = np.argmin(distances)
                closest_center = self._centers[closest_idx]

                closest_center.drift(point, prop)
                time += h

            if i >= i0:
                strategy.collect(self)

        return strategy.get_result()

    def run_sequential(
        self,
        robust_prop: float = 0.0,
        initialization: str = "kpp",
        strategy: RobustificationStrategy | None = None,
    ):
        """Run SA with sequential brownian motion then drift."""
        i0, strategy = self._prepare_run(robust_prop, initialization, strategy)
        times = self._initialize_times(self.n)
        time = 0.0

        for i, point in enumerate(self._observations, start=1):
            T = times[i]

            while time <= T - self._step_size:
                h = min(time + self._step_size, T) - time
                for center in self._centers:
                    center.brownian_motion(h)
                time += h

            distances = self.space.batch_distances_from_centers(self._centers, point)
            closest_idx = np.argmin(distances)
            closest_center = self._centers[closest_idx]

            prop = min((times[i] - times[i - 1]) * self._beta * np.log(1 + time), 1)
            closest_center.drift(point, prop)

            time = T

            if i >= i0:
                strategy.collect(self)

        return strategy.get_result()
