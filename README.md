

# kmeanssa-ng

[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI
version](https://img.shields.io/pypi/v/kmeanssa-ng.svg)](https://pypi.org/project/kmeanssa-ng/)
[![Python
3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Documentation
Status](https://readthedocs.org/projects/kmeanssa-ng/badge/?version=latest.png)](https://kmeanssa-ng.readthedocs.io/en/latest/?badge=latest)
[![Pipeline
Status](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/badges/main/pipeline.svg)](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/pipelines)
[![Coverage
Report](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/badges/main/coverage.svg)](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/commits/main)
[![Code style:
Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**K-means clustering on metric graphs and Riemannian manifolds, via
simulated annealing.**

`kmeanssa-ng` clusters data that lives on complex network structures
(**quantum graphs**), on **Riemannian manifolds** such as spheres and
hyperbolic space, or on any metric space where standard Euclidean
distance does not apply. Manifolds can be clustered directly, or
approximated by a graph (an $\varepsilon$-net) and clustered on that. It
uses a simulated annealing approach for robust convergence.

## Installation

Install the latest version directly from PyPi:

``` bash
pip install kmeanssa-ng
```

## Quickstart

Here is a minimal example of clustering points on a quantum graph:

``` python
from kmeanssa_ng import generate_sbm, SimulatedAnnealing, MostFrequentNode, KMeansPlusPlus

# Generate a graph with two distinct communities
# Distances are precomputed automatically (precompute=True by default)
graph = generate_sbm(
    sizes=[40, 40],       # Two communities of 40 nodes each
    p=[[0.8, 0.1],        # High intra-community connectivity  
       [0.1, 0.8]],       # Low inter-community connectivity
)

# Sample points uniformly across the graph
points = graph.sample_points(150)

# Run quantum graph specialized simulated annealing
sa = SimulatedAnnealing(
    observations=points,
    k=2,                  # We know there are 2 clusters
    lambda0=1.0,     # Standard temperature
    beta0=1.0,             # Standard drift strength
    step_size=0.1         # Standard step size
)

# Get cluster centers as node IDs (more interpretable)
node_centers = sa.run(
    robust_prop=0.1,                                  # 10% robustness  
    initialization_strategy=KMeansPlusPlus(),         # K-means++ initialization
    robustification_strategy=MostFrequentNode()       # Choose centers as most frequent nodes in clusters
)
print(f"Cluster centers near nodes: {node_centers}")
```

### Clustering on a manifold

To cluster on a curved space, approximate it by a graph (an
$\varepsilon$-net) and cluster on that:

``` python
from kmeanssa_ng import (
    create_sphere, approximate_geodesic_space, FibonacciNet,
    SimulatedAnnealing, KMeansPlusPlus, MinimizeEnergy,
)

# Approximate the 2-sphere by a graph, then cluster on it
graph = approximate_geodesic_space(create_sphere(2), 500, net=FibonacciNet())
points = graph.sample_points(200)
sa = SimulatedAnnealing(points, k=3, lambda0=1.0, beta0=0.5, step_size=0.05)
centers = sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1)
```

## Documentation

The full documentation, including API reference and tutorials, is
available at
[kmeanssa-ng.readthedocs.io](https://kmeanssa-ng.readthedocs.io/).

## Citation

If you use this package in your research, please cite:

``` bibtex
@software{kmeanssa_ng,
  author       = {Klutchnikoff, Nicolas and Gavra, Ioana},
  title        = {kmeanssa-ng: K-means Clustering on Quantum Graphs and Metric Spaces},
  year         = {2025},
  url          = {https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng},
  note         = {Python package for k-means clustering using simulated annealing}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file
for details.
