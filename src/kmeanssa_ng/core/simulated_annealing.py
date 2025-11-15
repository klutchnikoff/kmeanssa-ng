"""Simulated annealing algorithm for k-means clustering on metric spaces."""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

import numpy as np

from .abstract import Center, Point, Space
from .strategies.initialization import (
    InitializationStrategy,
)

if TYPE_CHECKING:
    from .strategies.robustification import RobustificationStrategy

logger = logging.getLogger(__name__)


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
        centers = sa.run_interleaved(robust_prop=0.1)
        ```
    """

    def __init__(
        self,
        observations: list[Point],
        k: int,
        lambda0: float = 1.0,
        beta0: float = 1.0,
        step_size: float = 0.1,
        energy_mode: str = "uniform",
        random_state: int | np.random.Generator | None = None,
    ) -> None:
        """Initialize the simulated annealing algorithm.

        Args:
            observations: List of points to cluster, all in the same metric space.
            k: Number of clusters.
            lambda0: Initial Brownian motion intensity parameter (must be > 0).
                Controls the magnitude of random exploration.

                Mathematical role: Scales the diffusion coefficient in the
                Brownian motion component. The standard deviation of each
                Brownian step is proportional to lambda0 * sqrt(step_size).

                Practical effect:
                - Higher values (1.5-3.0): More exploration, slower convergence,
                  better escape from local minima
                - Lower values (0.3-0.8): Less exploration, faster convergence,
                  higher risk of local minima
                - Recommended default: 1.0 for balanced exploration/exploitation

                TODO: Add reference to article on HAL/ArXiv when available.

            beta0: Initial drift intensity parameter (must be > 0).
                Controls how strongly centers are pulled toward observations.

                Mathematical role: The drift proportion at time t is computed as
                alpha(t) = min(h * beta0 * log(1 + t), 1) where h is the time
                interval. This controls the strength of attraction toward the
                nearest observation.

                Practical effect:
                - Higher values (2.0-5.0): Stronger drift, faster convergence,
                  more exploitation of current best positions
                - Lower values (0.3-0.8): Weaker drift, more exploration,
                  slower convergence
                - Recommended default: 1.0-2.0 for most cases

                TODO: Add reference to article on HAL/ArXiv when available.

            step_size: Time discretization step for the SDE solver (must be > 0).
                Controls the temporal resolution of the stochastic process.

                Mathematical role: Euler discretization step Δt for solving the
                stochastic differential equation. Smaller values give more
                accurate simulation at the cost of more computation.

                Practical effect:
                - Smaller values (0.001-0.01): More accurate simulation, slower
                - Larger values (0.05-0.1): Faster but less accurate
                - Recommended default: 0.01 for good accuracy/speed tradeoff
                - Rule of thumb: Use step_size much smaller than the typical
                  time scale of the Poisson process (~ 1/lambda0)

            energy_mode: Energy calculation mode, either "uniform" or "obs".
                TODO: Document the difference between these modes.

            random_state: Controls randomness for reproducibility.
                Determines random number generation for all random operations:
                - Shuffling observations
                - Poisson process time generation
                - Brownian motion (via centers)
                - Initialization strategies (KMeansPlusPlus, RandomInit)
                - Space-specific random operations

                Pass an int for reproducible results across multiple function calls.
                When an int is provided, both random.seed() and np.random.seed()
                are set globally to ensure full reproducibility across all components.

                Pass a Generator instance for fine-grained control without affecting
                global state (advanced usage).

                Pass None for non-deterministic behavior (default).

                Example:
                    >>> # Reproducible with seed (recommended)
                    >>> sa1 = SimulatedAnnealing(points, k=3, random_state=42)
                    >>> sa2 = SimulatedAnnealing(points, k=3, random_state=42)
                    >>> # sa1 and sa2 will produce identical results
                    >>>
                    >>> # Advanced: use Generator without global state
                    >>> rng = np.random.default_rng(42)
                    >>> sa = SimulatedAnnealing(points, k=3, random_state=rng)
                    >>> # Note: This only controls operations using self._rng
                    >>> # Other components may still use global random state

        Raises:
            ValueError: If observations is empty, k <= 0, points are in different spaces,
                or hyperparameters are invalid.

        References:
            TODO: Add reference to your article:
            [1] Your Name. "Title of your paper". HAL/ArXiv, 2025.
                URL: https://...

        Example:
            >>> # Quick convergence setup
            >>> sa = SimulatedAnnealing(
            ...     points, k=5,
            ...     lambda0=0.5,  # Less exploration
            ...     beta0=3.0,     # Stronger drift
            ...     step_size=0.01
            ... )
            >>>
            >>> # Thorough search setup (avoid local minima)
            >>> sa = SimulatedAnnealing(
            ...     points, k=5,
            ...     lambda0=2.0,   # More exploration
            ...     beta0=1.0,     # Gentler drift
            ...     step_size=0.01
            ... )
        """
        self._validate_constructor_parameters(
            observations, k, lambda0, beta0, step_size
        )
        self._initialize_random_generator(random_state)

        self._space = observations[0].space
        self._observations = observations.copy()
        self._k = k
        self._lambda = float(lambda0)
        self._beta = float(beta0)
        self._step_size = float(step_size)
        self._energy_mode = energy_mode

        # Use random.shuffle (global state) for consistency with rest of codebase
        # If we used self._rng.shuffle, it would desynchronize from global state
        random.shuffle(self._observations)
        self._centers: list[Center] = []

    def _validate_constructor_parameters(
        self,
        observations: list[Point],
        k: int,
        lambda0: float,
        beta0: float,
        step_size: float,
    ) -> None:
        """Validate parameters for the constructor."""
        if not observations:
            raise ValueError("Observations must be a non-empty list of points.")
        if k <= 0:
            raise ValueError("Number of clusters 'k' must be greater than zero.")
        if any(obs.space != observations[0].space for obs in observations):
            raise ValueError("All observations must belong to the same metric space.")

        self._validate_positive_float(lambda0, "lambda0")
        self._validate_positive_float(beta0, "beta0")
        self._validate_positive_float(step_size, "step_size")

    def _validate_positive_float(self, value: float, name: str) -> None:
        """Validate that a value is a positive float."""
        try:
            float_value = float(value)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"{name} must be a number, got {type(value).__name__}"
            ) from e
        if float_value <= 0:
            raise ValueError(f"{name} must be positive, got {float_value}")

    def _initialize_random_generator(
        self, random_state: int | np.random.Generator | None
    ) -> None:
        """Initialize the random number generator."""
        if isinstance(random_state, np.random.Generator):
            self._rng = random_state
        else:
            self._rng = np.random.default_rng(random_state)
            # Also set global random state for full reproducibility
            # This affects: initialization strategies, centers' brownian motion, space operations
            if isinstance(random_state, int):
                random.seed(random_state)
                np.random.seed(random_state)

    @property
    def n(self) -> int:
        """Number of observations."""
        return len(self._observations)

    @property
    def observations(self) -> list[Point]:
        """List of observation points."""
        return self._observations

    @property
    def centers(self) -> list[Center]:
        """Current cluster centers."""
        return self._centers

    @property
    def space(self) -> Space:
        """Metric space containing the observations."""
        return self._space

    @property
    def k(self) -> int:
        """Number of clusters."""
        return self._k

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
            # Use random.random() for consistency with global state
            poiss_sum += -1 / self._lambda * np.log(random.random())
            T[i + 1] = np.sqrt(poiss_sum + 1) - 1
        return T

    def calculate_energy_fallback(
        self, centers: list[Center], points: list[Point]
    ) -> float:
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

    def calculate_energy(self, centers: list[Center]) -> float:
        """Calculate k-means energy for given centers based on the energy mode."""
        if (
            hasattr(self.space, "calculate_energy_numba")
            and self.space.calculate_energy_numba is not None
        ):
            return self.space.calculate_energy_numba(centers, how=self._energy_mode)
        return self.space.calculate_energy(centers, how=self._energy_mode)

    def _prepare_run(
        self,
        robust_prop: float,
        initialization_strategy: InitializationStrategy,
        robustification_strategy: RobustificationStrategy,
    ) -> tuple[int, RobustificationStrategy]:
        """Prepare the simulation by initializing centers and strategy."""
        if robust_prop < 0 or robust_prop > 1:
            raise ValueError("The proportion must be in [0,1]")

        i0 = int(np.floor((self.n - 1) * (1 - robust_prop)))

        self._centers = initialization_strategy.initialize_centers(self)

        robustification_strategy.initialize(self)
        return i0, robustification_strategy

    def run_interleaved(
        self,
        initialization_strategy: InitializationStrategy,
        robustification_strategy: RobustificationStrategy,
        robust_prop: float = 0.0,
    ):
        """Run SA with interleaved drift and brownian motion."""
        return self._run_algorithm(
            "interleaved",
            initialization_strategy,
            robustification_strategy,
            robust_prop,
        )

    def run_sequential(
        self,
        initialization_strategy: InitializationStrategy,
        robustification_strategy: RobustificationStrategy,
        robust_prop: float = 0.0,
    ):
        """Run SA with sequential brownian motion then drift."""
        return self._run_algorithm(
            "sequential",
            initialization_strategy,
            robustification_strategy,
            robust_prop,
        )

    def _run_algorithm(
        self,
        mode: str,
        initialization_strategy: InitializationStrategy,
        robustification_strategy: RobustificationStrategy,
        robust_prop: float = 0.0,
    ):
        """Core SA algorithm, supporting interleaved and sequential modes."""
        logger.info(
            "Starting %s SA: k=%d, n_obs=%d, lambda0=%.3f, beta0=%.3f, "
            "step_size=%.4f, robust_prop=%.2f",
            mode,
            self._k,
            self.n,
            self._lambda,
            self._beta,
            self._step_size,
            robust_prop,
        )

        i0, strategy = self._prepare_run(
            robust_prop, initialization_strategy, robustification_strategy
        )
        times = self._initialize_times(self.n)
        time = 0.0
        progress_interval = max(1, self.n // 10)

        for i, point in enumerate(self._observations):
            T = times[i] if mode == "interleaved" else times[i + 1]

            if i % progress_interval == 0 and i > 0:
                progress = 100 * i / self.n
                logger.info(
                    "Progress: %.1f%% (%d/%d observations processed)",
                    progress,
                    i,
                    self.n,
                )

            logger.debug("Processing observation %d, target time T=%.4f", i, T)

            if mode == "interleaved":
                while time <= T - self._step_size:
                    h = min(time + self._step_size, T) - time
                    prop = min(h * self._beta * np.log(1 + time), 1)
                    logger.debug(
                        "Time step: time=%.4f, h=%.4f, drift_prop=%.4f", time, h, prop
                    )
                    for center in self._centers:
                        center.brownian_motion(h)
                    distances = self.space.distances_from_centers(self._centers, point)
                    closest_idx = np.argmin(distances)
                    self._centers[closest_idx].drift(point, prop)
                    time += h
            else:  # sequential
                while time <= T - self._step_size:
                    h = min(time + self._step_size, T) - time
                    logger.debug("Brownian motion: time=%.4f, h=%.4f", time, h)
                    for center in self._centers:
                        center.brownian_motion(h)
                    time += h
                distances = self.space.distances_from_centers(self._centers, point)
                closest_idx = np.argmin(distances)
                prop = min((times[i + 1] - times[i]) * self._beta * np.log(1 + time), 1)
                logger.debug(
                    "Drift: closest_center=%d, drift_prop=%.4f", closest_idx, prop
                )
                self._centers[closest_idx].drift(point, prop)
                time = T

            if i >= i0:
                strategy.collect(self)
                logger.debug(
                    "Collected centers for robustification at observation %d", i
                )

        result = strategy.get_result()
        logger.info("%s SA completed successfully", mode.capitalize())
        return result
