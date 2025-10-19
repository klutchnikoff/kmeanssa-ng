# Installation

## Requirements

- Python 3.9 or higher
- PDM (Python Dependency Manager) - recommended
- Git

## Dependencies

The core dependencies are minimal and focus on performance:

- `numpy >= 1.24.0` - Numerical computations
- `networkx >= 3.0` - Graph data structures
- `pandas >= 2.0.0` - Data handling

## Installation Methods

### From Source (Current Method)

Since the package is not yet published on PyPI, install directly from the source repository:

```bash
# Clone the repository
git clone https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng.git
cd kmeanssa-ng

# Install with PDM (recommended)
pdm install

# Alternative: Install with pip
pip install -e .
```

### Development Installation

For development work, install with all development dependencies:

```bash
# Clone and enter directory
git clone https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng.git
cd kmeanssa-ng

# Install with dev dependencies
pdm install

# Verify installation
pdm run pytest
```

### Future: PyPI Installation

Once published to PyPI, you will be able to install simply with:

```bash
pip install kmeanssa-ng
```

## Verify Installation

Test your installation with a quick example:

```python
from kmeanssa_ng import generate_simple_graph, SimulatedAnnealing

# Create a simple test graph
graph = generate_simple_graph(n_a=5, n_aa=3, bridge_length=2.0)
graph.precomputing()

# Sample points
points = graph.sample_points(20)

# Run clustering
sa = SimulatedAnnealing(points, k=2)
centers = sa.run(initialization="kpp")

print(f"Found {len(centers)} centers")
print("Installation successful!")
```

## Common Issues

### PDM Not Found

If `pdm` is not installed:

```bash
pip install pdm
```

### Import Errors

If you get import errors, ensure the package is installed in development mode:

```bash
pip install -e .
```

### Graph Connectivity Warnings

If you see warnings about graph connectivity, ensure your test graphs are connected:

```python
import networkx as nx
from kmeanssa_ng import as_quantum_graph

# Check if your graph is connected
G = nx.karate_club_graph()
print(f"Connected: {nx.is_connected(G)}")

# Convert to quantum graph
qg = as_quantum_graph(G, edge_length=1.0)
```

## Next Steps

Once installed, check out:

- [Quick Start Guide](quickstart.md) - Your first clustering example
- [Core Concepts](concepts.md) - Understanding the fundamentals
- [Examples](../examples/basic-clustering.md) - Practical use cases