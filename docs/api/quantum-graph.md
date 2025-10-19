# Quantum Graph API

The quantum graph module provides a complete implementation of k-means clustering on metric graphs where points can exist anywhere on edges.

## Quantum Graph Space

The main class representing a quantum graph as a metric space.

::: kmeanssa_ng.quantum_graph.space.QuantumGraph
    options:
        members:
            - __init__
            - distance
            - sample_points
            - sample_centers
            - precomputing
            - compute_clusters
            - validate
            - _distance_point_to_point
            - _distance_node_to_point
            - _distance_point_to_node

## Points and Centers

Classes representing points and centers on quantum graphs.

::: kmeanssa_ng.quantum_graph.point.QGPoint
    options:
        members:
            - __init__
            - __eq__
            - __hash__
            - __str__
            - __repr__

::: kmeanssa_ng.quantum_graph.center.QGCenter
    options:
        members:
            - __init__
            - brownian_motion
            - drift_step
            - copy
            - to_node
            - __str__
            - __repr__

## Quantum Graph Simulated Annealing

Specialized simulated annealing implementation with quantum graph-specific features.

::: kmeanssa_ng.quantum_graph.qg_simulated_annealing.QGSimulatedAnnealing
    options:
        members:
            - __init__
            - run_for_kmeans
            - run_for_mean
            - _get_most_frequent_nodes