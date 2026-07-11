"""Simulated annealing algorithm for k-means clustering on metric spaces."""

from __future__ import annotations

import logging
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
        from kmeanssa_ng import (
            KMeansPlusPlus,
            MinimizeEnergy,
            SimulatedAnnealing,
            generate_simple_graph,
        )
        from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

        # Create a space and sample observations
        graph = generate_simple_graph()
        points = graph.sample_points(100, strategy=UniformNodeSampling(random_state=0))

        # Run simulated annealing with the interleaved algorithm
        sa = SimulatedAnnealing(points, k=5, random_state=0)
        centers = sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1)
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
            lambda0: Intensity scale of the Poisson observation clock (must be > 0).

                Mathematical role: the annealing processes one observation per
                arrival of an inhomogeneous Poisson process of intensity
                lambda(t) = lambda0 * (1 + t) (the paper's schedule). It does
                **not** scale the Brownian steps themselves (each micro-step
                has standard deviation sqrt(step_size), independent of
                lambda0).

                Practical effect:
                - Higher values: arrivals come faster, so the same number of
                  observations spans a shorter annealing horizon (less
                  Brownian exploration per observation)
                - Lower values: longer horizon, more exploration between
                  observation events
                - Recommended default: 1.0

                See the companion paper (References) for the derivation of the
                time schedule.

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

                See the companion paper (References) for the derivation of the
                drift schedule.

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

            energy_mode: Which reference measure the k-means energy (mean
                squared distance to the nearest center) is averaged under —
                used by ``MinimizeEnergy`` to select the best visited state
                and by ``record_energy`` diagnostics:
                - "uniform": average over all graph nodes with equal weight,
                  measuring how well the centers cover the whole geometry
                  irrespective of where the observations lie.
                - "empirical": average over this algorithm's own observation
                  points, exactly where they lie (the empirical k-means
                  objective). The only mode supported on Riemannian
                  manifolds.
                - "node_measure": average under the per-node ``obs_weight``
                  measure registered on the graph by the caller (e.g. a
                  population measure); graph spaces only.
                The former "obs" mode was split into "empirical" and
                "node_measure" and now raises.

            random_state: Controls randomness for reproducibility.
                Determines random number generation for all random operations:
                - Shuffling observations
                - Poisson process time generation
                - Brownian motion (via centers)
                - Initialization strategies (KMeansPlusPlus, RandomInit)
                - Space-specific random operations

                All randomness flows through a single numpy Generator
                (``self._rng``), which is propagated to the centers so that
                every stochastic component is driven by the same stream. No
                global random state (``random.seed``/``np.random.seed``) is
                touched, so runs are isolated and fully reproducible.

                Pass an int for a reproducible seed, a Generator instance for
                fine-grained control, or None for non-deterministic behavior
                (default).

                Example:
                    >>> # Reproducible with seed (recommended)
                    >>> sa1 = SimulatedAnnealing(points, k=3, random_state=42)
                    >>> sa2 = SimulatedAnnealing(points, k=3, random_state=42)
                    >>> # sa1 and sa2 produce identical results
                    >>>
                    >>> # Or pass an explicit Generator
                    >>> rng = np.random.default_rng(42)
                    >>> sa = SimulatedAnnealing(points, k=3, random_state=rng)

        Raises:
            ValueError: If observations is empty, k <= 0, points are in different spaces,
                or hyperparameters are invalid.

        References:
            C. Brécheteau, I. Gavra, N. Klutchnikoff. "Online k-means Clustering
            on Metric Graphs and Geodesic Spaces" (preprint). Derives the
            annealing dynamics and its convergence analysis.

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
        if energy_mode == "obs":
            raise ValueError(
                "energy_mode 'obs' was split into two explicit modes: use "
                "'empirical' to average over this algorithm's observation "
                "points, or 'node_measure' to average under the per-node "
                "'obs_weight' measure registered on the graph."
            )
        if energy_mode not in ("uniform", "empirical", "node_measure"):
            raise ValueError(
                "energy_mode must be 'uniform', 'empirical' or "
                f"'node_measure', got {energy_mode!r}"
            )
        self._initialize_random_generator(random_state)

        self._space = observations[0].space
        self._observations = observations.copy()
        self._k = k
        self._lambda = float(lambda0)
        self._beta = float(beta0)
        self._step_size = float(step_size)
        self._energy_mode = energy_mode

        # Shuffle through the instance Generator so ordering is reproducible
        # from random_state without touching global random state.
        self._rng.shuffle(self._observations)
        self._centers: list[Center] = []
        self._energy_history: list[float] = []
        self._time_history: list[float] = []

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

    @property
    def energy_history(self) -> np.ndarray:
        """Energy after each observation from the last ``run(record_energy=True)``.

        Empty until such a run; the first entry is the energy of the initial
        centers (time 0), aligned with :attr:`time_history`.
        """
        return np.asarray(self._energy_history)

    @property
    def time_history(self) -> np.ndarray:
        """Annealing time at each recorded energy (see :attr:`energy_history`)."""
        return np.asarray(self._time_history)

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
        # Arrival times of an inhomogeneous Poisson process of intensity
        # lambda(t) = lambda0 * (1 + t) (the paper's schedule). Its cumulative
        # intensity is Lambda(t) = lambda0 * (t + t^2/2), and the i-th arrival
        # is T_i = Lambda^{-1}(S_i) where S_i is a sum of unit-rate
        # exponentials. Here poiss_sum accumulates S_i / lambda0 (draws of mean
        # 1/lambda0), so inverting Lambda gives sqrt(2 * poiss_sum + 1) - 1.
        # The factor 2 was previously missing, which realised the schedule
        # lambda(t) = 2 * lambda0 * (1 + t) instead -- a silent doubling of
        # lambda0 relative to the paper.
        T = np.zeros(n + 1)
        poiss_sum = 0.0
        for i in range(n):
            poiss_sum += self._rng.exponential(1.0 / self._lambda)
            T[i + 1] = np.sqrt(2.0 * poiss_sum + 1.0) - 1.0
        return T

    def calculate_energy(self, centers: list[Center]) -> float:
        """Calculate k-means energy for given centers based on the energy mode.

        Delegates to the space. The algorithm's own observations are the data
        of the "empirical" mode only: "uniform" and "node_measure" define
        their reference measure without them (and reject them, so no mode can
        silently shadow another). Acceleration (e.g. the quantum graph's
        numba kernels) is the space's concern, dispatched inside
        ``Space.calculate_energy``.
        """
        observations = self._observations if self._energy_mode == "empirical" else None
        return self.space.calculate_energy(
            centers, how=self._energy_mode, observations=observations
        )

    def run(
        self,
        initialization_strategy: InitializationStrategy,
        robustification_strategy: RobustificationStrategy,
        robust_prop: float = 0.0,
        record_energy: bool = False,
    ):
        """Run the simulated annealing algorithm.

        This is the primary method to execute the simulated annealing algorithm.
        It performs an interleaved sequence of Brownian motion (exploration)
        and drift (exploitation) for the cluster centers.

        Args:
            record_energy: If True, record the energy and annealing time after
                each observation into :attr:`energy_history` and
                :attr:`time_history` (for convergence diagnostics). Off by
                default so the energy is not recomputed when not needed.
        """
        logger.info(
            "Starting SA: k=%d, n_obs=%d, lambda0=%.3f, beta0=%.3f, "
            "step_size=%.4f, robust_prop=%.2f",
            self._k,
            self.n,
            self._lambda,
            self._beta,
            self._step_size,
            robust_prop,
        )

        if robust_prop < 0 or robust_prop > 1:
            raise ValueError("The proportion must be in [0,1]")

        i0 = int(np.floor((self.n - 1) * (1 - robust_prop)))

        self._centers = initialization_strategy.initialize_centers(self)
        # Seed each center's RNG from the SA's generator so that all stochastic moves
        # (Brownian step size and vertex routing) are reproducible from random_state.
        for _center in self._centers:
            _center.seed_rng(self._rng)

        robustification_strategy.initialize(self)
        strategy = robustification_strategy

        times = self._initialize_times(self.n)
        time = 0.0
        progress_interval = max(1, self.n // 10)

        if record_energy:
            self._energy_history = [self.calculate_energy(self._centers)]
            self._time_history = [time]

        for i, point in enumerate(self._observations):
            # times[0] is the origin of the clock: observation i is processed
            # over the interval (times[i], times[i + 1]].
            T = times[i + 1]

            if i % progress_interval == 0 and i > 0:
                progress = 100 * i / self.n
                logger.info(
                    "Progress: %.1f%% (%d/%d observations processed)",
                    progress,
                    i,
                    self.n,
                )

            logger.debug("Processing observation %d, target time T=%.4f", i, T)

            while time < T:
                h = min(self._step_size, T - time)
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

            if i >= i0:
                strategy.collect(self)
                logger.debug(
                    "Collected centers for robustification at observation %d", i
                )

            if record_energy:
                self._energy_history.append(self.calculate_energy(self._centers))
                self._time_history.append(time)

        result = strategy.get_result()
        logger.info("SA completed successfully")
        return result
