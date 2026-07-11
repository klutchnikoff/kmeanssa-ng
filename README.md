

# kmeanssa-ng

[![License:
MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI
version](https://img.shields.io/pypi/v/kmeanssa-ng.svg)](https://pypi.org/project/kmeanssa-ng/)
[![Python
3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
<!-- Raw HTML so pandoc's default-image-extension does not append ".png" to
     the extensionless RTD badge URL (which would break the badge image). -->
<a href="https://kmeanssa-ng.readthedocs.io/en/latest/?badge=latest"><img src="https://readthedocs.org/projects/kmeanssa-ng/badge/?version=latest" alt="Documentation Status"></a>
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
hyperbolic space, or on any metric space where the standard Euclidean
distance does not apply. Manifolds can be clustered directly, or
approximated by a graph (an $\varepsilon$-net) and clustered on that.

**Statement of need.** Most clustering tools assume data in
$\mathbb{R}^d$ with the Euclidean metric. Data on graphs, curved
manifolds, or quotient spaces has no such coordinates, and the $k$-means
centroid — an average — is not even defined there. `kmeanssa-ng` targets
exactly this setting: its main algorithm is an **online
simulated-annealing** scheme (from the companion paper) whose centres
only need a Brownian motion and a drift, so it runs on *any* space that
implements three small abstractions — `Point`, `Center` and `Space`. A
classical **Lloyd** iteration is also provided as a reference method.
Both plug together through interchangeable *strategies* (initialisation,
robustification, sampling, centre update).

## Installation

Install the latest version directly from PyPI:

``` bash
pip install kmeanssa-ng
```

## Quickstart

Cluster points on a quantum graph. The annealer works out of the box —
`k`-means++ initialisation and energy-minimising robustification are the
defaults:

``` python
from kmeanssa_ng import generate_sbm, SimulatedAnnealing
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

# A graph with two communities; pairwise distances are precomputed by default.
graph = generate_sbm(sizes=[40, 40], p=[[0.8, 0.1], [0.1, 0.8]], random_state=0)

# Sample observations to cluster (an explicit sampling strategy is required).
points = graph.sample_points(150, strategy=UniformNodeSampling(random_state=0))

# robust_prop=0.1 selects the best-of-window state (0.0, the default, would
# compare only the first and last states).
centers = SimulatedAnnealing(points, k=2, random_state=0).run(robust_prop=0.1)
print(f"Found {len(centers)} cluster centers")
```

### Clustering on a manifold

To cluster on a curved space, approximate it by a graph (an
$\varepsilon$-net) and cluster on that:

``` python
from kmeanssa_ng import (
    create_sphere, approximate_geodesic_space, FibonacciNet, SimulatedAnnealing,
)
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

# Approximate the 2-sphere by an epsilon-net graph, then cluster on it.
graph = approximate_geodesic_space(create_sphere(2), 500, net=FibonacciNet())
points = graph.sample_points(200, strategy=UniformNodeSampling(random_state=0))
centers = SimulatedAnnealing(
    points, k=3, beta0=0.5, step_size=0.05, random_state=0
).run(robust_prop=0.1)
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
  year         = {2026},
  url          = {https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng},
  note         = {Python package for k-means clustering using simulated annealing}
}
```

## License

This project is licensed under the MIT License. See the `LICENSE` file
for details.
