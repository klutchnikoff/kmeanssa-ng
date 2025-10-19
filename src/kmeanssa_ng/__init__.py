"""kmeanssa-ng: K-means clustering on quantum graphs and metric spaces.

This package provides tools for k-means clustering on arbitrary metric spaces,
with a focus on quantum graphs (metric graphs where points can lie on edges).

The main algorithm is simulated annealing, which combines:
- Brownian motion for exploration
- Drift toward observations for exploitation
- Inhomogeneous Poisson process for temperature control

Example:
    ```python
    from kmeanssa_ng import generate_sbm, SimulatedAnnealing

    # Create a quantum graph
    graph = generate_sbm(sizes=[50, 50], p=[[0.7, 0.1], [0.1, 0.7]])

    # Sample points
    points = graph.sample_points(100)

    # Run simulated annealing
    sa = SimulatedAnnealing(points, k=2)
    centers = sa.run(robust_prop=0.1, initialization="kpp")
    ```
"""

__version__ = "0.1.0"
__author__ = "Nicolas Klutchnikoff"

from .core import Center, Point, SimulatedAnnealing, Space
from .quantum_graph import (
    QGCenter,
    QGPoint,
    QGSimulatedAnnealing,
    QuantumGraph,
    as_quantum_graph,
    complete_quantum_graph,
    generate_random_sbm,
    generate_sbm,
    generate_simple_graph,
    generate_simple_random_graph,
)

__all__ = [
    # Core abstractions
    "Point",
    "Center",
    "Space",
    "SimulatedAnnealing",
    # Quantum graph classes
    "QGPoint",
    "QGCenter",
    "QuantumGraph",
    "QGSimulatedAnnealing",
    # Generators
    "generate_simple_graph",
    "generate_simple_random_graph",
    "generate_sbm",
    "generate_random_sbm",
    "as_quantum_graph",
    "complete_quantum_graph",
]
