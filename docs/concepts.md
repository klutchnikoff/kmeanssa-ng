---
title: Core Concepts
---


## Overview: K-means on Arbitrary Metric Spaces

### The Quantization Problem

The key insight of `kmeanssa-ng` is that k-means can be generalized to
**quantization of probability distributions** on almost any metric space
$(\mathcal{M}, d)$, not just Euclidean space. Given a probability
measure $P$ on $(\mathcal{M}, d)$, we seek $k$ centers that minimize:

$$ 
\text{minimize} \quad U(c_1, \ldots, c_k) = \frac{1}{2}\int_{\mathcal{M}} \min_{j=1,\ldots,k} d^2(x, c_j) \, P(dx)
$$

where $d: \mathcal{M} \times \mathcal{M} \to \mathbb{R}_+$ is a distance
function.

**Key distinction**: We are not clustering a fixed dataset, but rather
**quantizing a probability distribution**. The algorithm is **online**:
observations arrive sequentially according to a Poisson process, and
each new observation $Y_k \sim P$ causes a drift that pulls the nearest
center toward it. Between observations, centers explore the space via
Brownian motion. This homogenized version of [Simulated
Annealing](simulated-annealing.md) processes data incrementally without
storing all observations.

**Connection to classical k-means**: When $P$ is a uniform distribution
over a finite set of data points, this reduces to the standard k-means
problem. But the framework is much more general:

- For **graph clustering without weights**: use $P = \text{Uniform}(V)$
  over vertices
- For **weighted graphs**: use $P$ proportional to vertex weights
- For **continuous spaces**: use any continuous probability distribution
  on the metric space

This opens up quantization to:

- **Metric graphs** (quantum graphs): networks where points can exist
  anywhere on edges
- **Riemannian manifolds**: curved spaces with geodesic distances
- **Custom spaces**: any domain with a meaningful distance metric and
  probability measure

### Why This Matters

Many real-world problems involve probability distributions on
non-Euclidean spaces:

- Network data (social networks, transportation, molecules)
- Geographical data on curved surfaces
- Data constrained to manifolds (e.g., directional data on spheres)

Standard k-means fails in these settings because Euclidean distance is
inappropriate. `kmeanssa-ng` provides a principled framework for
quantizing distributions in these spaces, using stochastic observations
to drive the optimization.

## Architecture: A Three-Layer Design

The power of `kmeanssa-ng` comes from its **modular, extensible
architecture** that cleanly separates concerns into three layers. This
design makes it possible to apply the same clustering algorithm to any
user-defined space.

### Layer 1: The Metric Space (`Space`)

This layer defines the geometric context. It consists of: - A `Space`
object that defines the distance metric and knows how to create
points. - `Point` objects, which are the data points to be clustered. -
`Center` objects, which are the mobile cluster centers.

The `Space` is responsible for all geometric calculations: measuring
distances, moving centers, and sampling points.

### Layer 2: The Algorithm (`SimulatedAnnealing`)

This layer contains the core optimization logic. The
`SimulatedAnnealing` class orchestrates the clustering process by
iteratively moving the `Center`s to minimize the k-means energy
function.

Crucially, **the algorithm is completely independent of the space it
operates on**. It interacts with the `Space`, `Point`s, and `Center`s
only through the abstract interfaces they provide.

### Layer 3: Concrete Implementations

This layer provides concrete implementations of the abstract `Space`,
`Point`, and `Center` classes for specific metric spaces. `kmeanssa-ng`
comes with a built-in implementation for [Quantum
Graphs](quantum-graphs.md).

The key benefit of this architecture is its extensibility. Users can add
support for new metric spaces simply by providing their own concrete
implementation of the `Space` layer. See [Custom
Spaces](custom-spaces.md) for a detailed guide.
