---
title: Quickstart
---


This guide provides a basic example of how to use `kmeanssa-ng` to
perform clustering on a “quantum graph” and visualize the results. A
quantum graph is a metric space where points can exist not only on nodes
but also along the edges, allowing for more granular analysis.

## Prerequisites

To run this entire guide, including the visualization step, you need to
install `kmeanssa-ng` with the `plot` extra:

``` bash
pip install "kmeanssa-ng[plot]"
```

## 1. Generate a Sample Graph

First, we generate a stochastic block model (SBM) graph with two
distinct communities. This graph will serve as our metric space.

``` python
from kmeanssa_ng import generate_sbm

# Generate a graph with two distinct communities
graph = generate_sbm(
    sizes=[40, 40],  # Two communities of 40 nodes each
    p=[
        [0.7, 0.01],  # High intra-community connectivity
        [0.01, 0.7],  # Low inter-community connectivity
    ],
)

# Note: Distances are automatically precomputed by default.
# The algorithm relies on distances between points on the graph,
# so generate_sbm() precomputes all-pairs shortest paths automatically.
```

## 2. Define the Data Distribution

The algorithm quantizes a probability distribution, not a fixed dataset.
We need to provide a set of `observations` that act as a representative
sample (a proxy) of this distribution.

In this example, our goal is to find centers for the **uniform
distribution over the nodes of the graph**. We thus generate points
sampled uniformly from the nodes.

``` python
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

# Sample points to serve as a proxy for a uniform data distribution
points = graph.sample_points(500, strategy=UniformNodeSampling())
```

## 3. Run K-means with Simulated Annealing

Now, we run the simulated annealing algorithm to find the cluster
centers. We specify the number of clusters (`k=2`) and other parameters
for the annealing process.

``` python
from kmeanssa_ng import SimulatedAnnealing, MostFrequentNode, KMeansPlusPlus

# Run quantum graph specialized simulated annealing
sa = SimulatedAnnealing(
    observations=points,
    k=2,  # We know there are 2 clusters
    lambda0=1.0,  # Cooling rate: higher values mean slower cooling
    beta0=1.0,  # Drift strength: higher values attract centers to dense areas more strongly
    step_size=0.1,  # Step size for center updates in each iteration
)

# Get cluster centers. The robustification strategy ensures these are node IDs.
centers = sa.run(
    robust_prop=0.1,  # 10% robustness
    initialization_strategy=KMeansPlusPlus(),  # K-means++ initialization
    robustification_strategy=MostFrequentNode(),  # Choose centers as most frequent nodes in clusters
)

print("Cluster centers (position in edge):")
for center in centers:
    print(center)
```

    Cluster centers (position in edge):
    Center near node 17 [edge (17, 2), pos=0.000]
    Center near node 57 [edge (57, 77), pos=0.000]

## 4. Visualize the Results

Finally, we visualize the graph and the resulting cluster centers.

The `draw()` method can color nodes based on a `"cluster"` attribute. To
get the cluster assignments, we use the `assign_clusters()` method. This
is a **stateless** operation that returns a list of cluster labels for a
given list of points. We can then use these labels to set the node
attributes for visualization.

``` python
import matplotlib.pyplot as plt
import networkx as nx

# Get the nodes as a list of points
nodes_as_points = graph.nodes_as_points()

# Assign each node to the nearest center and get the labels
labels = graph.assign_clusters(nodes_as_points, centers)

# Create a dictionary to map node IDs to cluster labels
node_to_cluster = {node.closest_node(): label for node, label in zip(nodes_as_points, labels)}

# Set the 'cluster' attribute on the graph for visualization
nx.set_node_attributes(graph, node_to_cluster, "cluster")

# Visualize the graph and clusters
fig, ax = plt.subplots(figsize=(10, 8))
graph.draw(
    ax=ax,
    color_by="cluster",
    centers=centers,
    node_size_by_obs=True,  # Show which nodes have more sampled points
    edge_color="grey",
)
plt.title("K-means Clustering on a Quantum Graph")
plt.show()
```

![](quickstart_files/figure-commonmark/cell-5-output-1.png)

The resulting plot will show the two communities of the graph, with the
nodes colored according to their assigned cluster and the cluster
centers highlighted.
