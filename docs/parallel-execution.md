---
title: Parallel Execution
---


# Parallel Execution with Multiple Seeds

`kmeanssa-ng` provides utilities for running multiple simulated
annealing instances in parallel with different random seeds. This is
useful for:

- **Finding better solutions**: Running multiple times increases the
  chance of finding the global optimum
- **Statistical analysis**: Analyzing the variability of clustering
  results across different initializations
- **Reproducible experiments**: Using fixed seeds for consistent results
  across runs

## Principle

The parallel execution uses Python’s `ProcessPoolExecutor` to achieve
true parallelism across multiple CPU cores. Each run:

1.  Executes in a separate process with its own random seed
2.  Runs the complete simulated annealing algorithm independently
3.  Returns the final centers and energy

Results are collected as they complete, sorted by energy (ascending),
and the best result is returned by default.

## Basic Usage

The simplest way to run multiple executions in parallel:

``` python
from kmeanssa_ng import run_parallel, generate_sbm
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.core.strategies.initialization import KMeansPlusPlus
from kmeanssa_ng.core.strategies.robustification import MinimizeEnergy

# Generate a graph with two communities
graph = generate_sbm(sizes=[50, 50], p=[[0.7, 0.1], [0.1, 0.7]])

# Run 10 times in parallel (uses all available CPU cores)
# Each run samples its own 100 points with a different seed
# Note: mp_context='fork' is needed for Jupyter/Quarto compatibility
best_centers = run_parallel(
    graph,
    n_points=100,
    k=2,
    sampling_strategy=UniformNodeSampling(),
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    n_runs=10,
    mp_context='fork'
)
print(f"Best result has {len(best_centers)} centers")
```

    Best result has 2 centers

**Parameters**:

- `space`: The metric space (e.g., QuantumGraph) to sample points from
- `n_points`: Number of points to sample for each run
- `k`: Number of clusters
- `n_runs`: Number of parallel executions (default: 10)
- `n_jobs`: Number of worker processes (default: -1 = all cores)
- Other parameters are passed to `SimulatedAnnealing` (lambda0, beta0,
  step_size, energy_mode, robust_prop)

**Note**: Each run samples its own observations, ensuring complete
independence between runs.

## Retrieving All Results

To analyze the distribution of results across different seeds:

``` python
best_centers, all_results = run_parallel(
    graph,
    n_points=100,
    k=2,
    sampling_strategy=UniformNodeSampling(),
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    n_runs=20,
    return_all=True,
    mp_context='fork'
)

# all_results is a list of (centers, energy, seed) sorted by energy
print(f"Best energy: {all_results[0][1]:.4f}")
print(f"Worst energy: {all_results[-1][1]:.4f}")

# Analyze energy distribution
import numpy as np
energies = [energy for _, energy, _ in all_results]
print(f"Mean energy: {np.mean(energies):.4f}")
print(f"Std energy: {np.std(energies):.4f}")
```

    Best energy: 1.6800
    Worst energy: 2.4400
    Mean energy: 1.8987
    Std energy: 0.1766

## Progress Monitoring

For long-running experiments, use `run_parallel_with_callback` to
monitor progress:

``` python
from kmeanssa_ng import run_parallel_with_callback

def show_progress(run_idx, seed, energy):
    print(f"✓ Run {run_idx+1}/20 completed - Energy: {energy:.4f} (seed={seed})")

centers = run_parallel_with_callback(
    graph,
    n_points=100,
    k=2,
    sampling_strategy=UniformNodeSampling(),
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    n_runs=20,
    callback=show_progress,
    mp_context='fork'
)
```

    ✓ Run 8/20 completed - Energy: 1.8100 (seed=1880415768)
    ✓ Run 1/20 completed - Energy: 1.8000 (seed=1495694099)
    ✓ Run 7/20 completed - Energy: 1.7000 (seed=589881679)
    ✓ Run 2/20 completed - Energy: 1.5100 (seed=183384938)
    ✓ Run 5/20 completed - Energy: 1.8009 (seed=1494222352)
    ✓ Run 3/20 completed - Energy: 2.0131 (seed=1680864592)
    ✓ Run 4/20 completed - Energy: 2.2200 (seed=1690508300)
    ✓ Run 11/20 completed - Energy: 1.9400 (seed=1754662378)
    ✓ Run 6/20 completed - Energy: 1.2800 (seed=83158909)
    ✓ Run 9/20 completed - Energy: 1.8509 (seed=1529584231)
    ✓ Run 13/20 completed - Energy: 1.8000 (seed=1118367909)
    ✓ Run 12/20 completed - Energy: 1.8905 (seed=1327791181)
    ✓ Run 14/20 completed - Energy: 1.8800 (seed=526868701)
    ✓ Run 15/20 completed - Energy: 2.0903 (seed=420427166)
    ✓ Run 16/20 completed - Energy: 1.9100 (seed=96177084)
    ✓ Run 10/20 completed - Energy: 1.7615 (seed=899972701)
    ✓ Run 17/20 completed - Energy: 2.0200 (seed=121421773)
    ✓ Run 18/20 completed - Energy: 2.1323 (seed=2047946061)
    ✓ Run 19/20 completed - Energy: 1.5400 (seed=1455078363)
    ✓ Run 20/20 completed - Energy: 1.8600 (seed=1842492945)

The callback function receives three arguments:

- `run_idx`: Index of the completed run (0 to n_runs-1)
- `seed`: Random seed used for this run
- `energy`: Final energy (k-means objective) for this run

## Reproducible Experiments

To ensure reproducibility, provide a fixed list of seeds:

``` python
# Define specific seeds
seeds = [42, 123, 456, 789, 101112]

# These runs will produce consistent results
result1 = run_parallel(graph, n_points=100, k=2, sampling_strategy=UniformNodeSampling(), initialization_strategy=KMeansPlusPlus(), robustification_strategy=MinimizeEnergy(), n_runs=5, seeds=seeds, mp_context='fork')
result2 = run_parallel(graph, n_points=100, k=2, sampling_strategy=UniformNodeSampling(), initialization_strategy=KMeansPlusPlus(), robustification_strategy=MinimizeEnergy(), n_runs=5, seeds=seeds, mp_context='fork')

# Results will be highly similar (same data, same initialization)
```

**Note**: While individual runs are deterministic given a seed, the
final result may vary slightly due to the order in which parallel
processes complete. However, the energy distribution should be
consistent.

## Advanced Parameters

You can pass all `SimulatedAnnealing` parameters:

``` python
centers = run_parallel(
    graph,
    n_points=150,
    k=3,
    sampling_strategy=UniformNodeSampling(),
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    n_runs=50,
    # SimulatedAnnealing parameters
    lambda0=2,             # Poisson-clock intensity
    beta0=0.5,             # drift strength
    step_size=0.05,        # SDE time step
    robust_prop=0.1,       # Use 10% of observations for robustification
    # Parallel execution parameters
    n_jobs=4,              # Use only 4 cores
    seeds=list(range(100, 150)),  # Specific seeds
    mp_context='fork'      # For Jupyter/Quarto compatibility
)
```

## Multiprocessing Context

The `mp_context` parameter controls how Python creates worker processes:

- **`None` (default)**: Uses the system default (`spawn` on
  macOS/Windows, `fork` on Linux)
- **`'fork'`**: Fast process creation, shares memory (Unix only).
  **Required for Jupyter/Quarto**.
- **`'spawn'`**: Safer isolation, slower startup. Recommended for
  production code.
- **`'forkserver'`**: Hybrid approach (Unix only)

For **production scripts**, omit `mp_context` to use the safer default.
For **documentation/notebooks**, use `mp_context='fork'` to avoid
serialization issues.

## Performance Considerations

- **Speedup**: With `n` cores and `n_runs >= n`, expect near-linear
  speedup (e.g., 4x faster with 4 cores)
- **Memory**: Each process has its own copy of the data. For large
  graphs, this may consume significant memory
- **Overhead**: Process creation has overhead. For very short runs,
  parallel execution may not be faster
- **Optimal n_runs**: Use at least as many runs as you have cores for
  good CPU utilization

## Example: Statistical Analysis

Here’s a complete example analyzing clustering stability:

``` python
from kmeanssa_ng import run_parallel, generate_sbm
import numpy as np

# Generate graph
graph = generate_sbm(sizes=[50, 50], p=[[0.8, 0.1], [0.1, 0.8]])

# Run 100 times, each sampling 150 points
best, all_results = run_parallel(
    graph,
    n_points=150,
    k=2,
    sampling_strategy=UniformNodeSampling(),
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    n_runs=100,
    return_all=True,
    mp_context='fork'
)

# Analyze results
energies = [energy for _, energy, _ in all_results]
print(f"Energy statistics:")
print(f"  Min:    {np.min(energies):.4f}")
print(f"  Median: {np.median(energies):.4f}")
print(f"  Max:    {np.max(energies):.4f}")
print(f"  Std:    {np.std(energies):.4f}")

# How often do we get the best result?
best_energy = energies[0]
n_optimal = sum(1 for e in energies if abs(e - best_energy) < 0.01)
print(f"\nFound optimal solution in {n_optimal}/100 runs ({n_optimal}%)")
```

    Energy statistics:
      Min:    1.0933
      Median: 1.4679
      Max:    2.2933
      Std:    0.2243

    Found optimal solution in 1/100 runs (1%)

## API Reference

For detailed API documentation, see:

- [`run_parallel()`](api/core.md#run_parallel)
- [`run_parallel_with_callback()`](api/core.md#run_parallel_with_callback)
