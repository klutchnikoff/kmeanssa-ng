# kmeanssa-ng

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Pipeline Status](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/badges/main/pipeline.svg)](https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/pipelines)
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
from kmeanssa_ng import KMeansSA, generate_sbm

# 1. Generate a sample Stochastic Block Model graph
graph = generate_sbm(sizes=[50, 50], p=[[0.8, 0.1], [0.1, 0.8]])
graph.precomputing()

# 2. Sample points on the graph
points = graph.sample_points(100)

# 3. Initialize and run the clustering algorithm
kmeans = KMeansSA(n_clusters=2, space=graph)
kmeans.fit(points)

# 4. Access the results
print(f"Cluster centers found at: {kmeans.cluster_centers_}")
print(f"Point labels: {kmeans.labels_}")
```

## Documentation

The full documentation, including API reference and tutorials, is under construction and will be available soon.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.