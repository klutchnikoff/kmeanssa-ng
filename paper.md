---
title: 'kmeanssa-ng: Online k-means clustering on metric graphs and geodesic spaces via simulated annealing'
tags:
  - Python
  - clustering
  - k-means
  - metric graphs
  - geodesic spaces
  - simulated annealing
authors:
  - name: Nicolas Klutchnikoff
    orcid: 0000-0001-9349-0771
    affiliation: 1
  - name: Ioana Gavra
    affiliation: 1  # TODO: add ORCID if available
affiliations:
  - name: "Univ Rennes, CNRS, IRMAR (Institut de Recherche Mathématique de Rennes) - UMR 6625, F-35000 Rennes, France"
    index: 1
date: 5 July 2026  # TODO: set to the submission date
bibliography: paper.bib
---

# Summary

`kmeanssa-ng` is a Python framework for online $k$-means clustering — the
quantisation of a probability distribution — on arbitrary metric and geodesic
spaces. Its guiding design lets users cluster on their *own* space by
implementing three small abstractions, `Point`, `Center` and `Space`, where a
`Center` only has to provide a Brownian motion and a drift. The main algorithm is
an **online simulated-annealing** scheme whose centres alternate Brownian
exploration and drift toward incoming observations so as to escape local optima;
a classical **Lloyd** iteration is also provided. Every ingredient is pluggable
through *strategies* (centre initialisation, robustification, sampling, Lloyd
updates, $\varepsilon$-net placement). Metric graphs (`QuantumGraph`) and
Riemannian manifolds (`RiemannianManifold`, with a closed-form `Sphere` and a
generic `geomstats` fallback) ship built in, and an $\varepsilon$-net approximates
a continuous geodesic space by a graph on which the clustering then runs. All
randomness flows through a single NumPy `Generator`.

# Statement of need

Clustering and quantisation of data on non-Euclidean structures — weighted
graphs, Riemannian manifolds, general geodesic spaces — recur across statistics
and machine learning. The classical Lloyd iteration [@lloyd1982] and its online
counterpart, competitive learning vector quantisation [@pages2015], are simple
and widely used but converge only to *local* optima of a non-convex objective;
general-purpose global heuristics such as simulated annealing [@kirkpatrick1983]
are rarely available in a form that operates directly on non-Euclidean spaces.
Most implementations, moreover, are tied to Euclidean data or to one specific
space. `kmeanssa-ng` is built around two ideas. First, a clean
separation between the clustering algorithm and the geometry: any space that
supplies a distance and, for its centres, a Brownian motion and a drift can be
clustered, so users plug in their own spaces without touching the solver. Second,
an online simulated-annealing dynamics that alternates exploration and
exploitation to escape local minima; the companion theoretical article
[@brecheteau2026online] establishes its convergence and rate. For continuous
geodesic spaces an $\varepsilon$-net turns the space into a metric graph on which
the clustering runs, and a system of interchangeable strategies makes
initialisation, robustification, sampling and updates configurable without
subclassing the algorithm. Randomness is fully reproducible through a single
generator. The package targets researchers who need principled clustering on
structured metric data and a framework to experiment on their own spaces.

# Key features

- **Define your own space** through the `Point`/`Center`/`Space` abstractions —
  a `Center` only needs a Brownian motion and a drift.
- Two algorithms: online **simulated annealing** (with native energy-trajectory
  recording) and the classical **Lloyd** iteration.
- **Interchangeable strategies** for centre initialisation (k-means++, random),
  robustification, sampling, Lloyd updates and $\varepsilon$-net placement.
- Built-in spaces: metric graphs (`QuantumGraph`, with graph generators) and
  Riemannian manifolds (`RiemannianManifold`, closed-form `Sphere`, generic
  `geomstats` fallback).
- $\varepsilon$-net approximation of geodesic spaces
  (`approximate_geodesic_space`, `build_epsilon_net_graph`).
- Fully reproducible randomness through a single NumPy `Generator`, clustering
  metrics (ARI, silhouette, …), and a one-command reproduction of the companion
  paper's experiments (`examples/paper/reproduce.py`).
- Built on NumPy [@harris2020array], networkx [@hagberg2008networkx],
  geomstats [@miolane2020geomstats], scikit-learn [@pedregosa2011scikit] and
  Numba [@lam2015numba].

# Example

Clustering on a weighted stochastic-block-model graph with the online annealer:

```python
from kmeanssa_ng import (
    SimulatedAnnealing, KMeansPlusPlus, MinimizeEnergy, generate_random_sbm,
)
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

graph = generate_random_sbm(
    sizes=[50, 50], p=[[0.8, 0.1], [0.1, 0.8]],
    weights=[1, 1], lengths=[[1, 4], [4, 1]], random_state=0,
)
observations = graph.sample_points(500, strategy=UniformNodeSampling(random_state=0))

sa = SimulatedAnnealing(observations, k=2, random_state=0)
centers = sa.run(KMeansPlusPlus(), MinimizeEnergy())
labels = graph.assign_clusters(observations, centers)
```

The same few lines cluster on a sphere or any user-defined space: replace the
graph by `create_sphere(2)` (or a custom `Space`) and choose space-appropriate
sampling and update strategies.

# Acknowledgements

Funding: ANR GeoDSiC (ANR-22-CE40-0007).

# AI usage disclosure

The design and architecture of this software were conceived entirely by the
authors: the algorithms, the `Point`/`Center`/`Space` abstraction and the
strategy-based configuration of every component are the authors' own ideas. Claude
Code (Anthropic; model Claude Opus 4.8) was used as a development assistant, whose
contribution was limited to implementation-level help under the authors'
direction — refactoring, bug fixing and test scaffolding, the tooling that
reproduces the paper's experiments, and drafting of documentation. All AI-assisted outputs were reviewed, edited and validated by the
human authors, who made every design and scientific decision and take full
responsibility for the software and the paper.

# References
