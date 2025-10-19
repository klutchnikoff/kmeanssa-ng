# kmeanssa-ng

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

**K-means clustering on quantum graphs and metric spaces using simulated annealing**

`kmeanssa-ng` is a modern Python package for solving the k-means problem on arbitrary metric spaces, with a focus on **quantum graphs** — metric graphs where points can lie anywhere on edges, not just at nodes.

## Features

- **Quantum Graph Implementation**: Full support for points and clustering on quantum graphs
- **Simulated Annealing**: Robust k-means algorithm combining Brownian motion and drift
- **Flexible Architecture**: Extensible to any metric space via abstract base classes
- **k-means++ Initialization**: Smart center initialization for faster convergence
- **Robustification**: Optional averaging of final iterations for stable results
- **Graph Generators**: Pre-built generators for testing (SBM, random graphs, etc.)
- **Type Hints**: Fully typed codebase for better IDE support
- **Modern Python**: Built with Python 3.9+ features

## Installation

### From source (for now)

```bash
git clone https://github.com/your-username/kmeanssa-ng.git
cd kmeanssa-ng
pdm install
```

### After PyPI publication

```bash
pip install kmeanssa-ng
```

## Quick Start

```python
from kmeanssa_ng import generate_sbm, SimulatedAnnealing

# Create a quantum graph with 2 clusters
graph = generate_sbm(
    sizes=[50, 50],
    p=[[0.7, 0.1], [0.1, 0.7]]  # High intra-cluster, low inter-cluster edges
)

# Precompute pairwise distances (speeds up distance queries)
graph.precomputing()

# Sample 100 random points on the graph
points = graph.sample_points(100)

# Run simulated annealing for k-means with k=2
sa = SimulatedAnnealing(
    points,
    k=2,
    lambda_param=1,
    beta=1.0,
    step_size=0.1
)

# Get cluster centers
centers = sa.run(
    robust_prop=0.1,          # Use last 10% of iterations for robustification
    initialization="kpp",      # k-means++ initialization
    algorithm_version="v1"     # Algorithm variant
)

# Compute cluster assignments
graph.compute_clusters(centers)
```

## Core Concepts

### Quantum Graphs

A **quantum graph** is a metric graph where:
- Points can lie on edges at arbitrary positions
- Distance is measured along the graph (geodesic distance)
- Each edge has a length attribute

```python
from kmeanssa_ng import QuantumGraph, QGPoint

# Create a simple quantum graph
graph = QuantumGraph()
graph.add_edge(0, 1, length=1.0)
graph.add_edge(1, 2, length=2.0)
graph.add_edge(0, 2, length=3.0)

# A point on edge (0, 1) at position 0.5
point = QGPoint(graph, edge=(0, 1), position=0.5)
```

### Simulated Annealing Algorithm

The algorithm alternates between:

1. **Brownian Motion** (exploration): Centers perform random walks
2. **Drift** (exploitation): Centers move toward nearest observations
3. **Robustification**: Averages over the last iterations to avoid local minima

Temperature is controlled by an inhomogeneous Poisson process.

## Graph Generators

### Stochastic Block Model

```python
from kmeanssa_ng import generate_sbm, generate_random_sbm

# Uniform SBM
graph = generate_sbm(
    sizes=[50, 50],
    p=[[0.7, 0.1], [0.1, 0.7]]
)

# SBM with custom edge lengths per block
graph = generate_random_sbm(
    sizes=[50, 50],
    p=[[0.7, 0.1], [0.1, 0.7]],
    lengths=[[1, 4], [4, 1]]  # Longer inter-cluster edges
)
```

### Simple Test Graphs

```python
from kmeanssa_ng import generate_simple_graph, generate_simple_random_graph

# Symmetric two-cluster graph
graph = generate_simple_graph(n_a=5, n_aa=3, bridge_length=2.0)

# Random two-cluster graph with Poisson branching
graph = generate_simple_random_graph(
    n_a=5, n_b=5,
    lam_a=2, lam_b=2,  # Poisson parameters
    bridge_length=10.0
)
```

### From NetworkX Graphs

```python
import networkx as nx
from kmeanssa_ng import as_quantum_graph

G = nx.karate_club_graph()
qg = as_quantum_graph(G, edge_length=1.0)
qg.precomputing()
```

## Quantum Graph Specific Features

For quantum graphs, you can use specialized methods to get node-based results:

```python
from kmeanssa_ng import generate_sbm, QGSimulatedAnnealing

graph = generate_sbm(sizes=[50, 50], p=[[0.7, 0.1], [0.1, 0.7]])
points = graph.sample_points(100)

# Use QGSimulatedAnnealing for quantum-specific methods
sa = QGSimulatedAnnealing(points, k=2)

# Get most frequent closest nodes during robustification
node_ids = sa.run_for_kmeans(robust_prop=0.1)

# For k=1, get the single most frequent node
sa_mean = QGSimulatedAnnealing(points, k=1)
mean_node = sa_mean.run_for_mean(robust_prop=0.1)
```

## Extending to Custom Metric Spaces

You can implement k-means on any metric space by subclassing the core abstractions:

```python
from kmeanssa_ng.core import Space, Point, Center

class MyCustomSpace(Space):
    def distance(self, p1: Point, p2: Point) -> float:
        # Implement your distance function
        ...

    def sample_points(self, n: int) -> list[Point]:
        # Implement point sampling
        ...

    # Implement other required methods...
```

The core `SimulatedAnnealing` class works with any metric space. For space-specific features, create a specialized subclass like `QGSimulatedAnnealing`.

## Development

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/kmeanssa-ng.git
cd kmeanssa-ng

# Install with dev dependencies
pdm install

# Run tests
pdm run pytest

# Run linter
pdm run ruff check src/

# Format code
pdm run ruff format src/
```

### Testing

```bash
# Run all tests
pdm run pytest

# With coverage
pdm run pytest --cov=kmeanssa_ng --cov-report=html

# Specific test file
pdm run pytest tests/test_quantum_graph.py
```

## Project Structure

```
kmeanssa-ng/
├── src/
│   └── kmeanssa_ng/
│       ├── core/                        # Abstract base classes and SA algorithm
│       │   ├── abstract.py              # Point, Center, Space abstractions
│       │   └── simulated_annealing.py   # Core SA algorithm
│       ├── quantum_graph/               # Quantum graph implementation
│       │   ├── point.py                 # QGPoint
│       │   ├── center.py                # QGCenter
│       │   ├── space.py                 # QuantumGraph
│       │   ├── qg_simulated_annealing.py  # Quantum-specific SA methods
│       │   └── generators.py            # Graph generators
│       └── utils/                       # Utilities (empty for now)
├── tests/                               # Test suite
├── docs/                                # Documentation (MkDocs)
└── pyproject.toml                      # PDM configuration
```

## Algorithm Variants

The package provides two algorithm variants:

- **v1**: Interleaves Brownian motion and drift within each time step
- **v2**: Performs all Brownian motion first, then drift

```python
centers = sa.run(algorithm_version="v2")
```

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{kmeanssa_ng,
  author = {Klutchnikoff, Nicolas},
  title = {kmeanssa-ng: K-means clustering on quantum graphs},
  year = {2025},
  url = {https://github.com/your-username/kmeanssa-ng}
}
```

## Author

**Nicolas Klutchnikoff**
Email: nicolas.klutchnikoff@univ-rennes2.fr

## Acknowledgments

This package is a refactored and modernized version of the original `kmeanssa` package.
