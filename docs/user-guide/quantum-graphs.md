# Quantum Graphs

Quantum graphs are the foundation of kmeanssa-ng. This guide provides everything you need to know about working with quantum graphs effectively.

## What Makes Quantum Graphs Special?

Unlike traditional graphs where vertices represent discrete entities, quantum graphs treat edges as **first-class citizens**:

- Points can exist **anywhere** on an edge, not just at vertices
- Distance is measured along the graph structure (**geodesic distance**)
- Each edge has a **length** attribute representing physical distance
- The graph must be **connected** for distance computation

```python
from kmeanssa_ng import QuantumGraph, QGPoint

# Create a quantum graph manually
graph = QuantumGraph()
graph.add_edge(0, 1, length=2.0)  # Edge with length 2
graph.add_edge(1, 2, length=1.5)  # Edge with length 1.5
graph.add_edge(2, 0, length=3.0)  # Edge with length 3

# Points can be positioned anywhere on edges
point_a = QGPoint(graph, edge=(0, 1), position=0.25)  # 25% along edge (0,1)
point_b = QGPoint(graph, edge=(1, 2), position=0.80)  # 80% along edge (1,2)

# Distance is computed along the graph
distance = graph.distance(point_a, point_b)
print(f"Distance: {distance}")
```

## Creating Quantum Graphs

### Method 1: From Scratch

```python
from kmeanssa_ng import QuantumGraph

graph = QuantumGraph()

# Add edges with lengths
graph.add_edge('A', 'B', length=1.0)
graph.add_edge('B', 'C', length=2.5)
graph.add_edge('C', 'D', length=1.8)
graph.add_edge('D', 'A', length=2.2)

# Always validate and precompute
graph.validate()  # Checks connectivity and edge lengths
graph.precomputing()  # Caches shortest paths
```

### Method 2: From NetworkX Graphs

```python
import networkx as nx
from kmeanssa_ng import as_quantum_graph

# Start with any NetworkX graph
G = nx.erdos_renyi_graph(20, 0.3)

# Convert with uniform edge lengths
quantum_graph = as_quantum_graph(G, edge_length=1.0)

# Or use existing edge attributes
G = nx.karate_club_graph()
# Add custom lengths based on some property
for (u, v) in G.edges():
    G[u][v]['weight'] = 1.0 + (u + v) * 0.1  # Example weighting

quantum_graph = as_quantum_graph(G, length_attr='weight')
```

### Method 3: Using Generators

```python
from kmeanssa_ng import generate_sbm, generate_simple_graph

# Stochastic Block Model - great for testing clustering
graph = generate_sbm(
    sizes=[25, 25, 30],  # Three clusters
    p=[[0.8, 0.1, 0.05],  # Within and between cluster probabilities
       [0.1, 0.9, 0.1],
       [0.05, 0.1, 0.85]]
)

# Simple symmetric graph - good for development
graph = generate_simple_graph(
    n_a=10,          # Size of each cluster
    n_aa=5,          # Extra edges within clusters
    bridge_length=5.0 # Distance between clusters
)
```

## Working with Points and Centers

### QGPoint - Data Points

Points represent data locations on the quantum graph:

```python
from kmeanssa_ng import QGPoint

# Create points at specific locations
point1 = QGPoint(graph, edge=(0, 1), position=0.3)
point2 = QGPoint(graph, edge=(2, 3), position=0.7)

# Points are immutable and hashable
print(f"Point: {point1}")  # QGPoint(edge=(0, 1), position=0.30)
print(f"Hash: {hash(point1)}")  # Can be used in sets/dicts

# Access properties
print(f"Edge: {point1.edge}")
print(f"Position: {point1.position}")
```

### QGCenter - Cluster Centers

Centers can move during the clustering process:

```python
from kmeanssa_ng import QGCenter

# Create a center
center = QGCenter(graph, edge=(1, 2), position=0.5)

# Centers can perform operations
new_center = center.brownian_motion(step_size=0.1)  # Random walk
drift_center = center.drift_step(points, beta=1.0)  # Move toward points

# Get the closest node (useful for interpretation)
closest_node = center.to_node()
print(f"Center near node: {closest_node}")
```

## Distance Computation

Understanding how distances are computed is crucial for effective use:

### Same Edge Distance

```python
# Two points on the same edge
point1 = QGPoint(graph, edge=(0, 1), position=0.2)  # 20% along edge
point2 = QGPoint(graph, edge=(0, 1), position=0.8)  # 80% along edge

# Distance = |0.8 - 0.2| * edge_length
distance = graph.distance(point1, point2)
```

### Different Edge Distance

For points on different edges, the algorithm:
1. Finds shortest path from each point to its edge endpoints
2. Uses precomputed shortest paths between nodes
3. Combines the distances

```python
# Points on different edges
point1 = QGPoint(graph, edge=(0, 1), position=0.3)
point2 = QGPoint(graph, edge=(2, 3), position=0.7)

# Complex geodesic distance computation
distance = graph.distance(point1, point2)
```

### Performance: Precomputing

Always call `precomputing()` before intensive distance calculations:

```python
# Without precomputing: O(|V|²) for each distance query
graph.precomputing()  # One-time O(|V|³) cost

# Now all distance queries are much faster
for i in range(1000):
    d = graph.distance(random_point1, random_point2)  # Fast!
```

## Sampling Points

### Uniform Sampling

```python
# Sample points uniformly across all edges
points = graph.sample_points(100)

# Each edge gets points proportional to its length
# Longer edges → more points
```

### Custom Sampling (Advanced)

```python
# If you need custom sampling, implement it manually
import random

def custom_sample_points(graph, n_points):
    points = []
    edges = list(graph.edges())
    
    for _ in range(n_points):
        # Choose edge (you can customize this distribution)
        edge = random.choice(edges)
        
        # Choose position on edge
        position = random.random()
        
        points.append(QGPoint(graph, edge=edge, position=position))
    
    return points

custom_points = custom_sample_points(graph, 50)
```

## Validation and Debugging

### Graph Validation

```python
# Check if your graph is valid
try:
    graph.validate()
    print("Graph is valid!")
except ValueError as e:
    print(f"Graph validation failed: {e}")

# Common issues:
# - Graph not connected
# - Missing or negative edge lengths
# - Self-loops or parallel edges (depending on use case)
```

### Connectivity Check

```python
import networkx as nx

# Check connectivity
if nx.is_connected(graph):
    print("Graph is connected")
else:
    print(f"Graph has {nx.number_connected_components(graph)} components")
    # You might need to work with the largest component
    largest_cc = max(nx.connected_components(graph), key=len)
    subgraph = graph.subgraph(largest_cc).copy()
```

### Distance Debugging

```python
# Debug distance computation
point1 = QGPoint(graph, edge=(0, 1), position=0.5)
point2 = QGPoint(graph, edge=(2, 3), position=0.5)

print(f"Point 1: {point1}")
print(f"Point 2: {point2}")
print(f"Distance: {graph.distance(point1, point2)}")

# Check if precomputing is working
print(f"Shortest path (0→2): {graph.shortest_distance.get((0, 2), 'Not computed')}")
```

## Real-World Applications

### Transportation Networks

```python
# Model a city road network
city_graph = QuantumGraph()

# Add roads with travel times as lengths
city_graph.add_edge('MainSt_1st', 'MainSt_2nd', length=2.5)  # 2.5 minutes
city_graph.add_edge('MainSt_2nd', 'MainSt_3rd', length=3.0)  # 3.0 minutes

# Points represent locations (GPS coordinates mapped to roads)
restaurant = QGPoint(city_graph, edge=('MainSt_1st', 'MainSt_2nd'), position=0.3)
school = QGPoint(city_graph, edge=('MainSt_2nd', 'MainSt_3rd'), position=0.8)

# Clustering can find optimal service locations
```

### Neural Networks

```python
# Model neural connections
brain_graph = QuantumGraph()

# Edges represent axon connections with signal delays
brain_graph.add_edge('neuron_1', 'neuron_2', length=0.5)  # 0.5ms delay
brain_graph.add_edge('neuron_2', 'neuron_3', length=1.2)  # 1.2ms delay

# Points represent signal locations along axons
signal_location = QGPoint(brain_graph, edge=('neuron_1', 'neuron_2'), position=0.6)
```

### River Networks

```python
# Model watershed systems
river_graph = QuantumGraph()

# Edges are river segments with lengths in kilometers
river_graph.add_edge('source', 'junction1', length=15.0)
river_graph.add_edge('junction1', 'junction2', length=22.0)

# Points represent measurement stations or pollution sources
station = QGPoint(river_graph, edge=('source', 'junction1'), position=0.4)
```

## Performance Optimization

### For Large Graphs

```python
# Optimize for large graphs (>1000 nodes)
graph.precomputing()  # Essential!

# Limit point sampling
n_points = min(5000, len(graph.edges()) * 20)
points = graph.sample_points(n_points)

# Use larger step sizes in clustering
from kmeanssa_ng import SimulatedAnnealing
sa = SimulatedAnnealing(points, k=5, step_size=0.2)
```

### Memory Considerations

```python
# For very large graphs, consider:
# 1. Working with subgraphs
largest_component = max(nx.connected_components(graph), key=len)
working_graph = graph.subgraph(largest_component).copy()

# 2. Reducing robustification
centers = sa.run(robust_prop=0.05)  # Use only 5% for averaging

# 3. Iterative processing
# Process clusters in batches rather than all at once
```

## Common Patterns

### Graph Comparison

```python
# Compare clustering quality across different graphs
graphs = [
    generate_sbm(sizes=[30, 30], p=[[0.8, 0.1], [0.1, 0.8]]),
    generate_simple_graph(n_a=30, n_aa=10, bridge_length=2.0)
]

for i, graph in enumerate(graphs):
    graph.precomputing()
    points = graph.sample_points(100)
    sa = SimulatedAnnealing(points, k=2)
    centers = sa.run(initialization="kpp")
    energy = sa._compute_energy(centers)
    print(f"Graph {i+1} energy: {energy:.3f}")
```

### Multi-Scale Analysis

```python
# Analyze clustering at different scales
for k in [2, 3, 4, 5]:
    sa = SimulatedAnnealing(points, k=k)
    centers = sa.run(initialization="kpp")
    energy = sa._compute_energy(centers)
    print(f"k={k}: energy={energy:.3f}")
```

## Next Steps

- **[Simulated Annealing Guide](simulated-annealing.md)** - Master the clustering algorithm
- **[Graph Generators](generators.md)** - Create test graphs for your research
- **[Custom Metric Spaces](custom-spaces.md)** - Extend beyond quantum graphs