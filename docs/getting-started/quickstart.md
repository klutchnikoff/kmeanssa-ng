# Quick Start

Get up and running with kmeanssa-ng in 5 minutes! This guide will walk you through your first clustering example.

## Basic Example

Let's cluster points on a simple quantum graph with two natural clusters:

```python
from kmeanssa_ng import generate_sbm, SimulatedAnnealing

# Step 1: Create a quantum graph with block structure
graph = generate_sbm(
    sizes=[30, 30],  # Two clusters of 30 nodes each
    p=[[0.8, 0.1],   # 80% intra-cluster edges, 10% inter-cluster
       [0.1, 0.8]]   # 10% inter-cluster, 80% intra-cluster  
)

# Step 2: Precompute distances (important for performance!)
graph.precomputing()

# Step 3: Sample random points on the graph
points = graph.sample_points(100)  # 100 points distributed on edges

# Step 4: Run k-means clustering
sa = SimulatedAnnealing(
    points=points,
    k=2,              # Number of clusters
    lambda_param=1,   # Temperature parameter
    beta=1.0,         # Drift strength
    step_size=0.1     # Step size for random walks
)

# Step 5: Get cluster centers
centers = sa.run(
    robust_prop=0.1,        # Use last 10% iterations for averaging
    initialization="kpp",   # k-means++ initialization
    algorithm_version="v1"  # Algorithm variant
)

# Step 6: Assign points to clusters
graph.compute_clusters(centers)

print(f"Found {len(centers)} cluster centers")
print(f"Centers located at: {[str(c) for c in centers]}")
```

## Understanding the Results

The algorithm returns `centers` - a list of `QGCenter` objects representing the optimal cluster centers. Each center has:

- `edge`: The edge where the center is located
- `position`: Position on the edge (between 0 and 1)

```python
for i, center in enumerate(centers):
    print(f"Cluster {i+1}: {center}")
    # Output: Cluster 1: QGCenter(edge=(5, 8), position=0.342)
```

## Customizing the Algorithm

### Algorithm Parameters

```python
# More exploration (higher temperature)
sa = SimulatedAnnealing(points, k=2, lambda_param=2, beta=0.5)

# More exploitation (stronger drift)  
sa = SimulatedAnnealing(points, k=2, lambda_param=0.5, beta=2.0)

# Smaller steps (more precise)
sa = SimulatedAnnealing(points, k=2, step_size=0.05)
```

### Different Graph Types

=== "Stochastic Block Model"

    ```python
    # Custom block probabilities
    graph = generate_sbm(
        sizes=[20, 30, 25], 
        p=[[0.9, 0.1, 0.05],
           [0.1, 0.8, 0.15], 
           [0.05, 0.15, 0.85]]
    )
    ```

=== "Simple Test Graph"

    ```python
    # Symmetric two-cluster graph
    graph = generate_simple_graph(
        n_a=10,          # Nodes in cluster A
        n_aa=5,          # Extra connections in cluster A  
        bridge_length=3.0 # Distance between clusters
    )
    ```

=== "From NetworkX"

    ```python
    import networkx as nx
    from kmeanssa_ng import as_quantum_graph
    
    # Use any NetworkX graph
    G = nx.karate_club_graph()
    graph = as_quantum_graph(G, edge_length=1.0)
    ```

### Quantum Graph Specific Features

For quantum graphs, you can use specialized methods:

```python
from kmeanssa_ng import QGSimulatedAnnealing

# Get node-based results (useful for interpretation)
qg_sa = QGSimulatedAnnealing(points, k=2)
node_ids = qg_sa.run_for_kmeans(robust_prop=0.1)

print(f"Cluster centers near nodes: {node_ids}")

# For k=1 (finding the mean)
mean_sa = QGSimulatedAnnealing(points, k=1) 
mean_node = mean_sa.run_for_mean(robust_prop=0.1)
print(f"Mean center near node: {mean_node}")
```

## Performance Tips

!!! tip "Speed Up Your Clustering"

    1. **Always call `precomputing()`** - This caches shortest paths
    2. **Use appropriate `robust_prop`** - 0.1 (10%) is usually sufficient  
    3. **Start with k-means++** - Set `initialization="kpp"`
    4. **Tune parameters gradually** - Start with defaults, then adjust

```python
# Fast setup for large graphs
graph.precomputing()  # Essential!
sa = SimulatedAnnealing(points, k=3, step_size=0.2)  # Larger steps
centers = sa.run(robust_prop=0.05, initialization="kpp")  # Less robustification
```

## Next Steps

Now that you've run your first example:

- **[Core Concepts](concepts.md)** - Understand quantum graphs and simulated annealing
- **[User Guide](../user-guide/quantum-graphs.md)** - Deep dive into quantum graphs
- **[Examples](../examples/basic-clustering.md)** - More practical examples
- **[API Reference](../api/core.md)** - Complete API documentation

## Common Patterns

### Batch Processing

```python
# Process multiple graphs
results = []
for graph_params in parameter_sets:
    graph = generate_sbm(**graph_params)
    graph.precomputing()
    points = graph.sample_points(100)
    sa = SimulatedAnnealing(points, k=2)
    centers = sa.run(initialization="kpp")
    results.append(centers)
```

### Parameter Optimization

```python
# Test different parameters
best_energy = float('inf')
best_params = None

for lambda_param in [0.5, 1.0, 2.0]:
    for beta in [0.5, 1.0, 2.0]:
        sa = SimulatedAnnealing(points, k=2, 
                              lambda_param=lambda_param, beta=beta)
        centers = sa.run()
        energy = sa._compute_energy(centers)
        
        if energy < best_energy:
            best_energy = energy
            best_params = (lambda_param, beta)

print(f"Best parameters: λ={best_params[0]}, β={best_params[1]}")
```