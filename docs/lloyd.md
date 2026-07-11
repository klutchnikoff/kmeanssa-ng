---
title: Lloyd's Algorithm
---


Lloyd’s algorithm is the classical $k$-means iteration: alternate an
**assignment** step (attach every point to its nearest centre) and an
**update** step (recompute each centre from its assigned points), until
the assignment stops changing. `kmeanssa-ng` provides it as a familiar
**reference method** alongside its main algorithm, [Simulated
Annealing](simulated-annealing.md).

## What Lloyd needs that annealing does not

Simulated Annealing moves centres with only a Brownian motion and a
drift, so it works on any space out of the box. Lloyd’s **update** step
is different: it has to *recompute* a centre from a set of points — a
Fréchet mean — and how you do that depends on the space. `kmeanssa-ng`
supplies this through a `LloydUpdateStrategy`, chosen to match your
space:

| Strategy                        | Space               | The new centre is…                              |
|---------------------------------|---------------------|-------------------------------------------------|
| `MostFrequentNodeUpdate`        | quantum graph       | the most frequent nearest node of the cluster   |
| `MinimizeEnergyNodeUpdate`      | quantum graph       | the node minimising the cluster’s energy        |
| `KarcherFrechetMean`            | Riemannian manifold | the intrinsic (Karcher) mean, via `exp`/`log`   |
| `SimulatedAnnealingFrechetMean` | any space           | the Fréchet mean found by a short annealing run |

The last one makes Lloyd usable on *any* space — at the cost of
delegating the update to Simulated Annealing itself.

## Example

``` python
from kmeanssa_ng import generate_sbm, Lloyd, MostFrequentNodeUpdate
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

graph = generate_sbm(sizes=[25, 25], p=[[0.8, 0.1], [0.1, 0.8]], random_state=0)
points = graph.sample_points(150, strategy=UniformNodeSampling(random_state=0))

lloyd = Lloyd(
    points, k=2,
    update_strategy=MostFrequentNodeUpdate(random_state=0),
    random_state=0,
)
centers = lloyd.run()  # KMeansPlusPlus initialisation by default
print(f"Found {len(centers)} cluster centers")
```

    Found 2 cluster centers

`run` initialises the centres with `KMeansPlusPlus` by default; pass an
`initialization_strategy` to change it. The `update_strategy`, by
contrast, has no universal default — it is required at construction,
since the right choice depends on the space.

## When to use it

Lloyd converges quickly and makes a good **baseline** to compare
against. Because it descends to the nearest critical point of the
energy, it can settle in a local minimum; when that matters, prefer
[Simulated Annealing](simulated-annealing.md), whose cooling schedule is
built to escape them.
