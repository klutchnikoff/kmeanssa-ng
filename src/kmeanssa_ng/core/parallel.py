"""Parallel execution utilities for simulated annealing runs.

This module provides functions to run multiple simulated annealing executions
in parallel with different random seeds, useful for robust clustering and
statistical analysis.
"""

from __future__ import annotations

import multiprocessing as mp
import random as rd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

if TYPE_CHECKING:
    from .abstract import Center, Space
    from .strategies.initialization import InitializationStrategy
    from .strategies.robustification import RobustificationStrategy
    from .strategies.sampling import SamplingStrategy


def _run_with_seed(
    space: "Space",
    n_points: int,
    k: int,
    seed: int,
    algorithm: Literal["interleaved", "sequential"],
    lambda0: float,
    beta0: float,
    step_size: float,
    energy_mode: str,
    robust_prop: float,
    sampling_strategy: "SamplingStrategy",
    initialization_strategy: "InitializationStrategy",
    robustification_strategy: "RobustificationStrategy",
) -> tuple[list[Center], float, int]:
    """Run simulated annealing with a specific seed.

    This is a worker function designed to be pickled for multiprocessing.

    Args:
        space: The metric space to sample points from.
        n_points: Number of points to sample.
        k: Number of clusters.
        seed: Random seed for reproducibility.
        algorithm: Which algorithm to use ("interleaved" or "sequential").
        lambda0: Poisson process intensity parameter.
        beta0: Inverse temperature parameter.
        step_size: Time step size.
        energy_mode: Energy calculation mode ('uniform' or 'obs').
        robust_prop: Proportion of observations for robustification.
        sampling_strategy: Strategy for sampling points from the space.
        initialization_strategy: Strategy instance for initializing centers.
        robustification_strategy: Strategy instance for robustifying results.

    Returns:
        Tuple of (centers, energy, seed) for this run.
    """
    # Import here to avoid circular dependencies
    from .simulated_annealing import SimulatedAnnealing

    # Set random seed for reproducibility (affects sampling, Poisson process, etc.)
    rd.seed(seed)
    np.random.seed(seed)

    # Sample observations with this seed
    observations = space.sample_points(n_points, strategy=sampling_strategy)

    # Create algorithm instance
    sa = SimulatedAnnealing(
        observations=observations,
        k=k,
        lambda0=lambda0,
        beta0=beta0,
        step_size=step_size,
        energy_mode=energy_mode,
    )

    # Run the algorithm
    if algorithm == "interleaved":
        centers = sa.run_interleaved(
            initialization_strategy=initialization_strategy,
            robustification_strategy=robustification_strategy,
            robust_prop=robust_prop,
        )
    else:  # sequential
        centers = sa.run_sequential(
            initialization_strategy=initialization_strategy,
            robustification_strategy=robustification_strategy,
            robust_prop=robust_prop,
        )

    # Calculate final energy
    energy = sa.calculate_energy_fallback(centers, observations)

    return centers, energy, seed


def run_parallel(
    space: "Space",
    n_points: int,
    k: int,
    sampling_strategy: SamplingStrategy,
    initialization_strategy: InitializationStrategy,
    robustification_strategy: RobustificationStrategy,
    n_runs: int = 10,
    algorithm: Literal["interleaved", "sequential"] = "interleaved",
    lambda0: float = 1,
    beta0: float = 1.0,
    step_size: float = 0.1,
    energy_mode: str = "uniform",
    robust_prop: float = 0.0,
    n_jobs: int = -1,
    seeds: list[int] | None = None,
    return_all: bool = False,
    mp_context: Literal["fork", "spawn", "forkserver"] | None = None,
) -> list[Center] | tuple[list[Center], list[tuple[list[Center], float, int]]]:
    """Run simulated annealing multiple times in parallel with different seeds.

    This function executes n_runs independent simulated annealing runs in parallel,
    each with a different random seed. Each run samples its own observations,
    generates its own Poisson process, and initializes differently, ensuring
    complete independence between runs.

    Args:
        space: The metric space to sample points from.
        n_points: Number of points to sample for each run.
        k: Number of clusters.
        n_runs: Number of parallel runs to execute.
        algorithm: Which algorithm variant to use ("interleaved" or "sequential").
        lambda_param: Poisson process intensity parameter (must be > 0).
        beta: Inverse temperature parameter (must be > 0).
        step_size: Time step for updating centers (must be > 0).
        sampling_strategy: Strategy for sampling points from the space (required).
        initialization_strategy: Strategy for initializing centers (required).
        robustification_strategy: Strategy for robustifying results (required).
        robust_prop: Proportion of final observations to use for robustification (0-1).
        n_jobs: Number of parallel jobs. -1 uses all available cores.
        seeds: Optional list of specific seeds to use. If None, generates random seeds.
        return_all: If True, return all results; if False, return only the best.
        mp_context: Multiprocessing context to use ('fork', 'spawn', 'forkserver').
            If None, uses the system default. Use 'fork' for Jupyter/Quarto compatibility.

    Returns:
        If return_all is False: List of best centers (lowest energy).
        If return_all is True: Tuple of (best_centers, all_results) where all_results
            is a list of (centers, energy, seed) tuples sorted by energy.

    Raises:
        ValueError: If n_runs <= 0 or other parameters are invalid.

    Example:
        ```python
        from kmeanssa_ng import run_parallel

        # Generate a graph
        graph = QuantumGraph(...)

        # Run 10 parallel executions, each sampling its own 100 points
        best_centers = run_parallel(graph, n_points=100, k=5, n_runs=10)

        # Get all results for analysis
        best, all_results = run_parallel(graph, n_points=100, k=5, n_runs=10, return_all=True)
        for centers, energy, seed in all_results:
            print(f"Seed {seed}: energy = {energy:.4f}")
        ```
    """
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")

    # Check n_jobs and issue a warning if it's too high
    import os
    import warnings

    cpu_count = os.cpu_count() or 1
    if n_jobs > cpu_count:
        warnings.warn(
            f"n_jobs={n_jobs} is greater than the number of available CPUs ({cpu_count}). "
            "This may lead to performance degradation.",
            UserWarning,
        )

    # Determine number of workers
    if n_jobs == -1:
        n_jobs = cpu_count
    elif n_jobs == -2:
        n_jobs = max(1, cpu_count - 1)

    # Generate seeds if not provided
    if seeds is None:
        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31, size=n_runs).tolist()
    elif len(seeds) != n_runs:
        raise ValueError(f"Length of seeds ({len(seeds)}) must match n_runs ({n_runs})")

    # Run all jobs in parallel
    results: list[tuple[list[Center], float, int]] = []

    # Set up multiprocessing context if specified
    executor_kwargs = {"max_workers": n_jobs}
    if mp_context is not None:
        if mp_context not in mp.get_all_start_methods():
            raise ValueError(
                f"Multiprocessing context '{mp_context}' not available on this platform. "
                f"Available: {mp.get_all_start_methods()}"
            )
        executor_kwargs["mp_context"] = mp.get_context(mp_context)

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        # Submit all jobs
        futures = [
            executor.submit(
                _run_with_seed,
                space,
                n_points,
                k,
                seed,
                algorithm,
                lambda0,
                beta0,
                step_size,
                energy_mode,
                robust_prop,
                sampling_strategy,
                initialization_strategy,
                robustification_strategy,
            )
            for seed in seeds
        ]

        # Collect results as they complete
        for future in as_completed(futures):
            centers, energy, seed = future.result()
            results.append((centers, energy, seed))

    # Sort by energy (best first)
    results.sort(key=lambda x: x[1])

    # Return results
    if return_all:
        return results[0][0], results
    else:
        return results[0][0]


def run_parallel_with_callback(
    space: "Space",
    n_points: int,
    k: int,
    sampling_strategy: SamplingStrategy,
    initialization_strategy: InitializationStrategy,
    robustification_strategy: RobustificationStrategy,
    n_runs: int = 10,
    algorithm: Literal["interleaved", "sequential"] = "interleaved",
    lambda0: float = 1.0,
    beta0: float = 1.0,
    step_size: float = 0.1,
    energy_mode: str = "uniform",
    robust_prop: float = 0.0,
    n_jobs: int = -1,
    seeds: list[int] | None = None,
    callback: Callable[[int, int, float], None] | None = None,
    mp_context: Literal["fork", "spawn", "forkserver"] | None = None,
) -> list[Center]:
    """Run parallel simulated annealing with progress callback.

    Similar to run_parallel but calls a callback function after each run completes,
    useful for progress tracking and real-time monitoring. Each run samples its own
    observations with its specific seed.

    Args:
        space: The metric space to sample points from.
        n_points: Number of points to sample for each run.
        k: Number of clusters.
        n_runs: Number of parallel runs to execute.
        algorithm: Which algorithm variant to use.
        lambda_param: Poisson process intensity parameter.
        beta: Inverse temperature parameter.
        step_size: Time step for updating centers.
        robust_prop: Proportion for robustification.
        n_jobs: Number of parallel jobs (-1 = all cores).
        seeds: Optional list of specific seeds.
        callback: Optional function(run_index, seed, energy) called after each run.
        mp_context: Multiprocessing context to use ('fork', 'spawn', 'forkserver').
            If None, uses the system default. Use 'fork' for Jupyter/Quarto compatibility.

    Returns:
        List of best centers (lowest energy).

    Example:
        ```python
        def progress_callback(run_idx, seed, energy):
            print(f"Run {run_idx+1}/{n_runs}: energy = {energy:.4f} (seed={seed})")

        graph = QuantumGraph(...)
        centers = run_parallel_with_callback(
            graph, n_points=100, k=5, n_runs=10, callback=progress_callback
        )
        ```
    """
    if n_runs <= 0:
        raise ValueError(f"n_runs must be positive, got {n_runs}")

    # Check n_jobs and issue a warning if it's too high
    import os
    import warnings

    cpu_count = os.cpu_count() or 1
    if n_jobs > cpu_count:
        warnings.warn(
            f"n_jobs={n_jobs} is greater than the number of available CPUs ({cpu_count}). "
            "This may lead to performance degradation.",
            UserWarning,
        )

    # Determine number of workers
    if n_jobs == -1:
        n_jobs = cpu_count
    elif n_jobs == -2:
        n_jobs = max(1, cpu_count - 1)

    # Generate seeds if not provided
    if seeds is None:
        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31, size=n_runs).tolist()
    elif len(seeds) != n_runs:
        raise ValueError(f"Length of seeds ({len(seeds)}) must match n_runs ({n_runs})")

    # Run all jobs in parallel with progress tracking
    results: list[tuple[list[Center], float, int]] = []
    completed_count = 0

    # Set up multiprocessing context if specified
    executor_kwargs = {"max_workers": n_jobs}
    if mp_context is not None:
        if mp_context not in mp.get_all_start_methods():
            raise ValueError(
                f"Multiprocessing context '{mp_context}' not available on this platform. "
                f"Available: {mp.get_all_start_methods()}"
            )
        executor_kwargs["mp_context"] = mp.get_context(mp_context)

    with ProcessPoolExecutor(**executor_kwargs) as executor:
        # Submit all jobs
        future_to_index = {
            executor.submit(
                _run_with_seed,
                space,
                n_points,
                k,
                seed,
                algorithm,
                lambda0,
                beta0,
                step_size,
                energy_mode,
                robust_prop,
                sampling_strategy,
                initialization_strategy,
                robustification_strategy,
            ): (idx, seed)
            for idx, seed in enumerate(seeds)
        }

        # Collect results as they complete
        for future in as_completed(future_to_index):
            idx, seed = future_to_index[future]
            centers, energy, result_seed = future.result()
            results.append((centers, energy, result_seed))
            completed_count += 1

            # Call callback if provided
            if callback is not None:
                callback(idx, result_seed, energy)

    # Sort by energy and return best
    results.sort(key=lambda x: x[1])
    return results[0][0]
