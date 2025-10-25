---
title: Quantum Graphs
---


## What is a Quantum Graph?

A quantum graph is a metric space built on a network structure where
**points can exist anywhere along edges**, not just at vertices. This is
the key distinction from traditional graphs: edges are treated as
continuous one-dimensional spaces, not just connections between nodes.

Consider a road network. In a classical graph, you can only place things
at intersections (vertices). In a quantum graph, you can place a sensor
at “30% of the way down Main Street” or a delivery point at “2.3 km
along Highway 5”. This makes quantum graphs ideal for modeling:

- Transportation networks (GPS coordinates along roads)
- Timeline data (events occurring between discrete time points)
- Molecular structures (atoms positioned along chemical bonds)
- Sensor networks (devices placed along communication lines)

Each edge has a **`length`** attribute representing its physical or
logical distance. A point on the graph is specified by a pair
`(edge, position)`, where `position ∈ [0,1]` indicates how far along
that edge the point sits. The distance between any two points is the
**geodesic distance**—the length of the shortest path through the
graph—making this a genuine metric space suitable for k-means
clustering.

## Structure and Requirements

The `QuantumGraph` class extends NetworkX’s `Graph`, so you can use all
standard NetworkX operations (adding edges, querying neighbors, etc.).
Two attributes are mandatory:

- **`length`** (edge attribute): Must be positive. Represents the
  physical or logical distance of the edge.
- **`weight`** (node attribute): Controls sampling probability.
  Typically set to 1.0 for uniform sampling.

The graph must also be **connected**—all nodes must be reachable from
each other—so that distances between any two points are well-defined.

In practice, you won’t create quantum graphs manually. Instead, use the
[graph generators](generators.md) which handle all attributes and
validation automatically. But if you need to build one from scratch or
convert an existing NetworkX graph, the `as_quantum_graph()` function
takes care of the details.

## Points and Centers

There are two types of quantum graph entities:

**`QGPoint`** represents an **observation**—a fixed data point. It’s
immutable and specified by an `(edge, position)` pair. For example,
`QGPoint(graph, edge=(0,1), position=0.3)` places a point 30% along the
edge from node 0 to node 1. These are the data you want to cluster.

**`QGCenter`** represents a **cluster center**. Unlike points, centers
are mobile: they can perform **Brownian motion** (random walks for
exploration) and **drift** (directed movement toward nearby
observations). These operations are the core of the [simulated annealing
algorithm](simulated-annealing.md). You don’t typically create centers
manually—the algorithm handles that.

## How Distances Work

The quantum graph computes **geodesic distances**—the shortest path
length through the graph structure. The computation strategy depends on
whether two points share an edge:

**Same edge**: The distance is simply
$|\text{pos}_1 - \text{pos}_2| \times \text{edge length}$. If two
delivery points are at 20% and 80% along a 5 km road, they’re
$|0.8 - 0.2| \times 5 = 3$ km apart.

**Different edges**: The algorithm finds the shortest path by: 1.
Computing the distance from each point to its nearest nodes 2. Finding
the shortest path between those nodes (using precomputed data) 3.
Choosing the combination that minimizes total distance

This is where **precomputing** becomes essential. The `precomputing()`
method calculates all-pairs shortest paths between nodes using
Dijkstra’s algorithm. Without this, every distance query would require
running a pathfinding algorithm—extremely slow for large graphs. All
[graph generators](generators.md) call `precomputing()` automatically,
but if you modify a graph after creation, you must call it again.

``` python
from kmeanssa_ng import generate_sbm

graph = generate_sbm(sizes=[30, 30], p=[[0.8, 0.1], [0.1, 0.8]])
print(f"Graph diameter (longest shortest path): {graph.diameter:.2f}")
```

    Graph diameter (longest shortest path): 3.00

## Sampling Points

To cluster data on a quantum graph, you need observations—`QGPoint`
objects. The `sample_points(n)` method generates them uniformly across
the graph:

``` python
points = graph.sample_points(100)
print(f"Sampled {len(points)} points across the graph")
```

    Sampled 100 points across the graph

The sampling is **proportional to edge lengths**: a 5 km road gets more
points than a 1 km road, ensuring true uniformity across the graph’s
total length.

### Weighted Sampling with Node Weights

You can control the sampling distribution by adjusting node `weight`
attributes. Points are sampled proportionally to the combined weight of
their edge’s endpoints. This is useful for modeling non-uniform
distributions—like population density on a road network or traffic
intensity.

Here’s a concrete example:

``` python
from kmeanssa_ng import QuantumGraph
import networkx as nx

# Create a simple triangle graph
weighted_graph = QuantumGraph()
weighted_graph.add_edge('A', 'B', length=1.0)
weighted_graph.add_edge('B', 'C', length=1.0)
weighted_graph.add_edge('C', 'A', length=1.0)

# Set node weights: node B is a "hot spot" with 5x more weight
weighted_graph.nodes['A']['weight'] = 1.0
weighted_graph.nodes['B']['weight'] = 5.0  # High-density area
weighted_graph.nodes['C']['weight'] = 1.0

weighted_# Distances precomputed automatically

# Sample points - more will appear near node B
weighted_points = weighted_graph.sample_points(1000)

# Count points by closest node to see the distribution
node_counts = {'A': 0, 'B': 0, 'C': 0}
for point in weighted_points:
    # Find which node the point is closest to
    u, v = point.edge
    closest_node = u if point.position < 0.5 else v
    node_counts[closest_node] += 1

print("Observations per node (approximate):")
for node in ['A', 'B', 'C']:
    weight = weighted_graph.nodes[node]['weight']
    print(f"  Node {node} (weight={weight}): {node_counts[node]} observations")
```

    Observations per node (approximate):
      Node A (weight=1.0): 142 observations
      Node B (weight=5.0): 716 observations
      Node C (weight=1.0): 142 observations

You should see approximately a 1:5:1 ratio matching the node weights
(A=1.0, B=5.0, C=1.0). Points are sampled proportionally to the combined
weight of their edge’s endpoints, so edges touching high-weight nodes
receive more observations. This allows you to model scenarios where
certain regions of the graph have higher data density.

## Working with Quantum Graphs: Key Points

When using quantum graphs for clustering, remember:

**Use generators**: The [graph generators](generators.md) handle all
requirements automatically. Manual construction is rarely necessary.

**Precomputing is mandatory**: Always call `precomputing()` after
creating or modifying a graph. This caches shortest paths and is
required for distance computation. Generators do this automatically.

**Connectivity matters**: The graph must be connected. For random graphs
(e.g., Erdős-Rényi), you may need to extract the largest connected
component.

**Edge lengths are semantic**: The `length` attribute should reflect
meaningful distances in your domain—travel time for roads, physical
distance for molecules, etc. This is what the clustering algorithm
optimizes over.

With these concepts in place, you can apply the [simulated annealing
algorithm](simulated-annealing.md) to cluster data on quantum graphs.
