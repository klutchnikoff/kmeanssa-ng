---
title: Riemannian Manifolds
---


## Clustering on curved spaces

Not all data lives on a graph or in flat Euclidean space. **Directions**
(wind, magnetic fields, gene expression on the unit sphere),
**geographic positions** on the globe, or **hierarchical structures**
embedded in hyperbolic space all live on curved **Riemannian
manifolds**, where the meaningful notion of distance is the
**geodesic**—the length of the shortest path *along the surface*, not
the straight line through the ambient space.

`kmeanssa-ng` clusters directly on such manifolds. Under the hood it
wraps [geomstats](https://geomstats.github.io/), which provides the
exponential and logarithm maps that let cluster centers move along
geodesics.

## Creating a manifold

Two factory functions return a `RiemannianManifold` ready to use:

``` python
from kmeanssa_ng import create_sphere, create_hyperbolic_space

sphere = create_sphere(2)  # the 2-sphere S^2, embedded in R^3
print(f"dim = {sphere.dim}, is_sphere = {sphere.is_sphere}")
```

    dim = 2, is_sphere = True

Points are stored in **extrinsic** coordinates—for `create_sphere(2)`,
unit vectors in $\mathbb{R}^3$.

## Points and clustering

An observation on a manifold is a `RiemannianPoint`. Sample a few
uniformly, then cluster them with the usual [simulated
annealing](simulated-annealing.md) workflow—the only difference is that
centers now perform Brownian motion and drift **along great-circle
geodesics**, via the exp/log maps, with no ambient projection:

``` python
from kmeanssa_ng import (
    RiemannianPoint, SimulatedAnnealing, KMeansPlusPlus, MinimizeEnergy,
)

coords = sphere.random_uniform(150, random_state=0)
observations = [RiemannianPoint(sphere, x) for x in coords]

sa = SimulatedAnnealing(
    observations, k=3, lambda0=1.0, beta0=0.5, step_size=0.05, energy_mode="obs"
)
centers = sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1)
print(f"Found {len(centers)} cluster centers on the sphere")
```

    Found 3 cluster centers on the sphere

On a manifold, energy is always measured over the sampled observations
(`energy_mode="obs"`), since there is no natural uniform distribution
over a continuous surface to sum against.

## Geodesic operations

A `RiemannianManifold` exposes the geometric primitives that drive the
algorithm—and that the [$\varepsilon$-net construction](epsilon-net.md)
builds on. The logarithm map gives the tangent direction from one point
to another, its norm is the geodesic distance, and the exponential map
is its inverse:

``` python
import numpy as np

a, b = sphere.random_uniform(2, random_state=1)
tangent = sphere.log(a, b)                       # direction from a toward b
print(f"geodesic distance a->b = {float(sphere.norm(a, tangent)):.3f}")
print(f"exp(a, log(a, b)) == b: {np.allclose(sphere.exp(a, tangent), b)}")
```

    geodesic distance a->b = 1.286
    exp(a, log(a, b)) == b: True

## Scaling up: mesh the manifold

Clustering directly on the manifold evaluates geodesics at every step.
For larger problems it is faster—and theoretically grounded—to
approximate the manifold **once** by a quantum graph and cluster on that
graph instead. That is the subject of [Meshing a
Manifold](epsilon-net.md).

## Hyperbolic space

The hyperboloid model of hyperbolic space is available too, and the same
geodesic operations apply:

``` python
hyperbolic = create_hyperbolic_space(2)
base = np.array([1.0, 0.0, 0.0])                 # the apex of the hyperboloid
tangent = np.array([0.0, 0.3, 0.4])              # a tangent direction there
print(f"geodesic step length = {float(hyperbolic.norm(base, tangent)):.3f}")
```

    geodesic step length = 0.500

> [!NOTE]
>
> Hyperbolic space is **non-compact**. Uniform sampling and
> $\varepsilon$-net meshing currently target **compact** manifolds (the
> sphere and the like); support for bounded hyperbolic regions is a
> planned extension.
