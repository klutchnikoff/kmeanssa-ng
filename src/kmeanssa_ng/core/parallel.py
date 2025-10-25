"""Parallel execution utilities for simulated annealing runs.

This module provides functions to run multiple simulated annealing executions
in parallel with different random seeds, useful for robust clustering and
statistical analysis.
"""

from __future__ import annotations

import random as rd
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import TYPE_CHECKING, Callable, Literal

import numpy as np

if TYPE_CHECKING:
    from .abstract import Center, Space
    from .strategies.initialization import InitializationStrategy
    from .strategies.robustification import RobustificationStrategy


def _run_with_seed(
    space: "Space",
    n_points: int,
    k: int,
    seed: int,
    algorithm: Literal["interleaved", "sequential"],
    lambda_param: int,
    beta: float,
    step_size: float,
    robust_prop: float,
    initialization: str | None,
    robustification: str | None,
) -> tuple[list[Center], float, int]:
    """Run simulated annealing with a specific seed.

    This is a worker function designed to be pickled for multiprocessing.

    Args:
        space: The metric space to sample points from.
        n_points: Number of points to sample.
        k: Number of clusters.
        seed: Random seed for reproducibility.
        algorithm: Which algorithm to use ("interleaved" or "sequential").
        lambda_param: Poisson process intensity parameter.
        beta: Inverse temperature parameter.
        step_size: Time step size.
        robust_prop: Proportion of observations for robustification.
        initialization: Name of initialization strategy (None for default).
        robustification: Name of robustification strategy (None for default).

    Returns:
        Tuple of (centers, energy, seed) for this run.
    """
    # Import here to avoid circular dependencies
    from .simulated_annealing import SimulatedAnnealing

    # Set random seed for reproducibility (affects sampling, Poisson process, etc.)
    rd.seed(seed)
    np.random.seed(seed)

    # Sample observations with this seed
    observations = space.sample_points(n_points)

    # Create algorithm instance
    sa = SimulatedAnnealing(
        observations=observations,
        k=k,
        lambda_param=lambda_param,
        beta=beta,
        step_size=step_size,
    )

    # Parse strategies (could be extended to support more strategies)
    init_strategy = None  # Use default
    robust_strategy = None  # Use default

    # Run the algorithm
    if algorithm == "interleaved":
        centers = sa.run_interleaved(
            robust_prop=robust_prop,
            initialization_strategy=init_strategy,
            robustification_strategy=robust_strategy,
        )
    else:  # sequential
        centers = sa.run_sequential(
            robust_prop=robust_prop,
            initialization_strategy=init_strategy,
            robustification_strategy=robust_strategy,
        )

    # Calculate final energy
    energy = sa.calculate_energy(centers, observations)

    return centers, energy, seed


def run_parallel(
    space: "Space",
    n_points: int,
    k: int,
    n_runs: int = 10,
    algorithm: Literal["interleaved", "sequential"] = "interleaved",
    lambda_param: int = 1,
    beta: float = 1.0,
    step_size: float = 0.1,
    robust_prop: float = 0.0,
    initialization_strategy: InitializationStrategy | None = None,
    robustification_strategy: RobustificationStrategy | None = None,
    n_jobs: int = -1,
    seeds: list[int] | None = None,
    return_all: bool = False,
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
        robust_prop: Proportion of final observations to use for robustification (0-1).
        initialization_strategy: Strategy for initializing centers (None = k-means++).
        robustification_strategy: Strategy for robustifying results (None = minimize energy).
        n_jobs: Number of parallel jobs. -1 uses all available cores.
        seeds: Optional list of specific seeds to use. If None, generates random seeds.
        return_all: If True, return all results; if False, return only the best.

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

    # Generate seeds if not provided
    if seeds is None:
        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31, size=n_runs).tolist()
    elif len(seeds) != n_runs:
        raise ValueError(f"Length of seeds ({len(seeds)}) must match n_runs ({n_runs})")

    # Determine number of workers
    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count() or 1

    # Run all jobs in parallel
    results: list[tuple[list[Center], float, int]] = []

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs
        futures = [
            executor.submit(
                _run_with_seed,
                space,
                n_points,
                k,
                seed,
                algorithm,
                lambda_param,
                beta,
                step_size,
                robust_prop,
                None,  # initialization
                None,  # robustification
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
    n_runs: int = 10,
    algorithm: Literal["interleaved", "sequential"] = "interleaved",
    lambda_param: int = 1,
    beta: float = 1.0,
    step_size: float = 0.1,
    robust_prop: float = 0.0,
    n_jobs: int = -1,
    seeds: list[int] | None = None,
    callback: Callable[[int, int, float], None] | None = None,
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

    # Generate seeds if not provided
    if seeds is None:
        rng = np.random.default_rng()
        seeds = rng.integers(0, 2**31, size=n_runs).tolist()
    elif len(seeds) != n_runs:
        raise ValueError(f"Length of seeds ({len(seeds)}) must match n_runs ({n_runs})")

    # Determine number of workers
    if n_jobs == -1:
        import os

        n_jobs = os.cpu_count() or 1

    # Run all jobs in parallel with progress tracking
    results: list[tuple[list[Center], float, int]] = []
    completed_count = 0

    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        # Submit all jobs
        future_to_index = {
            executor.submit(
                _run_with_seed,
                space,
                n_points,
                k,
                seed,
                algorithm,
                lambda_param,
                beta,
                step_size,
                robust_prop,
                None,
                None,
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
