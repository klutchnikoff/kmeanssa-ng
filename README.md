# kmeanssa-ng

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pipeline Status](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/badges/main/pipeline.svg)](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/pipelines)
[![Coverage Report](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/badges/main/coverage.svg)](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/commits/main)
[![Code style: Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

**K-means clustering on quantum graphs and metric spaces using simulated annealing.**

`kmeanssa-ng` provides tools for clustering data points that exist on complex network structures (quantum graphs) or other metric spaces where standard Euclidean distance does not apply. It uses a simulated annealing approach for robust convergence.

## Installation

Install the latest version directly from GitLab:
```bash
pip install git+https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng.git
```

## Quickstart

Here is a minimal example of clustering points on a quantum graph:

```python
from kmeanssa_ng import generate_sbm, QGSimulatedAnnealing, SimulatedAnnealing

# Generate a graph with two distinct communities
graph = generate_sbm(
    sizes=[40, 40],       # Two communities of 40 nodes each
    p=[[0.8, 0.1],        # High intra-community connectivity  
       [0.1, 0.8]],       # Low inter-community connectivity
)

# Essential: precompute shortest paths
graph.precomputing()

# Sample points uniformly across the graph
points = graph.sample_points(150)

# Run quantum graph specialized simulated annealing
qg_sa = QGSimulatedAnnealing(
    observations=points,
    k=2,                  # We know there are 2 clusters
    lambda_param=1.0,     # Standard temperature
    beta=1.0,            # Standard drift strength
    step_size=0.1        # Standard step size
)

# Get cluster centers as node IDs (more interpretable)
node_centers = qg_sa.run_for_kmeans(robust_prop=0.1)
print(f"Cluster centers near nodes: {node_centers}")
```

## Documentation

The full documentation, including API reference and tutorials, is under construction and will be available soon.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.