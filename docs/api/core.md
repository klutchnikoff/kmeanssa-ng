# Core API

The core module provides the fundamental abstractions and algorithms for k-means clustering on arbitrary metric spaces.

## Abstract Base Classes

These classes define the interface for extending kmeanssa-ng to new metric spaces.

::: kmeanssa_ng.core.abstract.Point
    options:
        members:
            - __init__
            - __eq__
            - __hash__

::: kmeanssa_ng.core.abstract.Center
    options:
        members:
            - __init__
            - brownian_motion
            - drift_step
            - copy

::: kmeanssa_ng.core.abstract.Space
    options:
        members:
            - distance
            - sample_points
            - sample_centers
            - compute_clusters

## Simulated Annealing Algorithm

The main clustering algorithm that works with any metric space implementation.

::: kmeanssa_ng.core.simulated_annealing.SimulatedAnnealing
    options:
        members:
            - __init__
            - run
            - _initialize_centers
            - _brownian_motion_step
            - _drift_step
            - _compute_energy