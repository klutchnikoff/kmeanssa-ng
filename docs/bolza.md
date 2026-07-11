---
title: The Bolza Surface
---


The **Bolza surface** is a compact genus-2 surface of constant negative
curvature ($-1$) — a closed hyperbolic counterpart to the sphere. It is
the quotient of the Poincaré disk by the Fuchsian group that glues
opposite sides of a regular hyperbolic octagon (interior angle $\pi/4$).
`kmeanssa-ng` ships it as `BolzaSurface` (via `create_bolza_surface`),
and it is the one built-in space with **no geomstats backend**: its
geometry is entirely closed-form and *quotient-aware*.

## Why it is special

Unlike the sphere or the hyperboloid, the Bolza surface is a
**quotient**: points glued across the octagon’s boundary are the *same*
point of the surface, even though their disk coordinates lie far apart.
Its geodesic operations account for this — `log` takes each target to
its nearest copy under the group, and `exp` folds every step back into
the fundamental octagon. This makes it a concrete demonstration that a
manifold can be defined **without** geomstats, and it is the space on
which the intrinsic $\varepsilon$-net strategies are exercised.

Points are `(2,)` real `(Re, Im)` coordinates inside the fundamental
octagon.

## Creating the surface and sampling

``` python
from kmeanssa_ng import create_bolza_surface, RiemannianPoint

surface = create_bolza_surface()
coords = surface.random_uniform(2, random_state=0)   # uniform by hyperbolic area
a, b = RiemannianPoint(surface, coords[0]), RiemannianPoint(surface, coords[1])
print(f"quotient distance a-b = {surface.distance(a, b):.3f}")
```

    quotient distance a-b = 0.910

## Clustering on the surface

Because the quotient has **no faithful Euclidean embedding** — points
glued across the boundary are close on the surface yet far apart in any
disk image — you must mesh it with the **intrinsic** $\varepsilon$-net
strategies, which search by geodesic distance.
`approximate_geodesic_space(..., intrinsic=True)` turns the surface into
a quantum graph you then cluster as usual:

``` python
from kmeanssa_ng import approximate_geodesic_space, SimulatedAnnealing
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

# Intrinsic build: O(n^2) in the net size, so this takes a while for large nets.
graph = approximate_geodesic_space(surface, 800, intrinsic=True, random_state=0)
points = graph.sample_points(300, strategy=UniformNodeSampling(random_state=0))
centers = SimulatedAnnealing(points, k=3, random_state=0).run(robust_prop=0.1)
```

The extrinsic strategies (KD-tree on ambient coordinates) would return
wrong neighbours on a quotient, so they refuse to run there — see
[Meshing a Manifold](epsilon-net.md) for the intrinsic/extrinsic
distinction. The paper’s `examples/paper/bolza.py` runs this end to end
and renders the recovered partition inside the fundamental octagon.
