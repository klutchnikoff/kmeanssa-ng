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

from importlib import metadata

__version__ = metadata.version("kmeanssa-ng")

from .core import (
    Center,
    Point,
    SimulatedAnnealing,
    Space,
    Lloyd,
    run_parallel,
    run_parallel_with_callback,
)
from .core.metrics import (
    adjusted_rand_index,
    calinski_harabasz,
    compute_labels,
    davies_bouldin,
    evaluate_clustering,
    normalized_mutual_info,
    silhouette,
)
from .core.strategies import (
    InitializationStrategy,
    KMeansPlusPlus,
    MinimizeEnergy,
    RandomInit,
    RobustificationStrategy,
    LloydUpdateStrategy,
    SimulatedAnnealingFrechetMean,
)
from .quantum_graph import (
    QGCenter,
    QGPoint,
    QuantumGraph,
    as_quantum_graph,
    complete_quantum_graph,
    generate_random_sbm,
    generate_sbm,
    generate_simple_graph,
    generate_simple_random_graph,
    MostFrequentNode,
    MostFrequentNodeUpdate,
    MinimizeEnergyNodeUpdate,
)
from .riemannian_manifold import (
    RiemannianManifold,
    RiemannianPoint,
    RiemannianCenter,
    create_sphere,
    create_hyperbolic_space,
    create_bolza_surface,
    FrechetMeanUpdate,
    EpsilonNetStrategy,
    UniformNet,
    RepulsionNet,
    RepulsionNetExtrinsicSpeedup,
    FibonacciNet,
    approximate_geodesic_space,
    build_epsilon_net_graph,
    estimate_covering_radius,
)

__all__ = [
    # Core abstractions
    "Point",
    "Center",
    "Space",
    "SimulatedAnnealing",
    "Lloyd",
    # Parallel execution
    "run_parallel",
    "run_parallel_with_callback",
    # Metrics
    "adjusted_rand_index",
    "calinski_harabasz",
    "compute_labels",
    "davies_bouldin",
    "evaluate_clustering",
    "normalized_mutual_info",
    "silhouette",
    # Strategies
    "InitializationStrategy",
    "KMeansPlusPlus",
    "RandomInit",
    "RobustificationStrategy",
    "MinimizeEnergy",
    "LloydUpdateStrategy",
    "SimulatedAnnealingFrechetMean",
    "MostFrequentNode",
    "MostFrequentNodeUpdate",
    "MinimizeEnergyNodeUpdate",
    "FrechetMeanUpdate",
    # Quantum graph classes
    "QGPoint",
    "QGCenter",
    "QuantumGraph",
    # Riemannian manifold classes
    "RiemannianManifold",
    "RiemannianPoint",
    "RiemannianCenter",
    "create_sphere",
    "create_hyperbolic_space",
    "create_bolza_surface",
    # Epsilon-net construction (manifold -> quantum graph)
    "EpsilonNetStrategy",
    "UniformNet",
    "RepulsionNet",
    "RepulsionNetExtrinsicSpeedup",
    "FibonacciNet",
    "approximate_geodesic_space",
    "build_epsilon_net_graph",
    "estimate_covering_radius",
    # Generators
    "generate_simple_graph",
    "generate_simple_random_graph",
    "generate_sbm",
    "generate_random_sbm",
    "as_quantum_graph",
    "complete_quantum_graph",
]
