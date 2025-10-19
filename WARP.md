# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Development Commands

### Environment Setup
```bash
# Install dependencies (including dev dependencies)
pdm install

# Install for development without dev dependencies
pdm install --prod
```

### Testing
```bash
# Run all tests
pdm run pytest

# Run tests with coverage report
pdm run pytest --cov=kmeanssa_ng --cov-report=html

# Run specific test file
pdm run pytest tests/test_quantum_graph.py

# Run tests in parallel (when available)
pdm run pytest -n auto
```

### Code Quality
```bash
# Run linter (check for issues)
pdm run ruff check src/

# Format code
pdm run ruff format src/

# Run both linting and formatting
pdm run ruff check src/ && pdm run ruff format src/
```

### Documentation
```bash
# Build documentation (if available)
pdm run mkdocs build

# Serve documentation locally
pdm run mkdocs serve
```

## Project Architecture

### Core Design Pattern

The project follows a **three-layer abstraction** pattern for implementing k-means clustering on arbitrary metric spaces:

1. **Abstract Layer** (`src/kmeanssa_ng/core/abstract.py`):
   - `Point`: Immutable elements in a metric space
   - `Center`: Movable points that can perform Brownian motion and drift
   - `Space`: Metric space containing points and centers with distance computation

2. **Algorithm Layer** (`src/kmeanssa_ng/core/simulated_annealing.py`):
   - `SimulatedAnnealing`: Core algorithm that works with any metric space
   - Uses inhomogeneous Poisson process for temperature control
   - Alternates between Brownian motion (exploration) and drift (exploitation)

3. **Implementation Layer** (`src/kmeanssa_ng/quantum_graph/`):
   - `QuantumGraph`: Specific implementation for metric graphs using NetworkX
   - `QGPoint`, `QGCenter`: Quantum graph-specific point and center implementations
   - `QGSimulatedAnnealing`: Specialized simulated annealing for quantum graphs

### Key Components

#### Quantum Graph (`QuantumGraph`)
- Extends NetworkX Graph for metric graph functionality
- Points can exist anywhere on edges, not just at nodes
- Precomputes all-pairs shortest paths for efficient distance queries
- Validates edge `length` attributes and graph connectivity

#### Simulated Annealing Algorithm
- Two algorithm variants: `v1` (interleaved) and `v2` (sequential)
- k-means++ initialization support (`initialization="kpp"`)
- Robustification: averages results from final iterations for stability
- Configurable via `lambda_param`, `beta`, and `step_size` hyperparameters

#### Graph Generators (`generators.py`)
- `generate_sbm()`: Stochastic Block Model for clustered graphs
- `generate_simple_graph()`: Symmetric two-cluster test graphs
- `generate_random_sbm()`: SBM with random edge lengths
- `as_quantum_graph()`: Convert NetworkX graphs to quantum graphs

### Distance Computation Strategy

The quantum graph uses a **hybrid distance computation approach**:
1. **Between nodes**: Uses precomputed shortest paths (Dijkstra)
2. **Between points on edges**: Computes geodesic distance considering:
   - Distance from point to nearest node on same edge
   - Shortest path between nodes
   - Distance from node to target point on its edge

### Extension Points

To implement k-means on a new metric space:

1. **Inherit from abstract base classes**:
   ```python
   class MySpace(Space):
       def distance(self, p1: Point, p2: Point) -> float: ...
       def sample_points(self, n: int) -> list[Point]: ...
       def sample_centers(self, k: int) -> list[Center]: ...
   ```

2. **Use existing `SimulatedAnnealing`** or create specialized subclass for space-specific features

### Testing Strategy

- **Unit tests**: Individual component functionality
- **Integration tests**: End-to-end algorithm execution
- **Property-based testing**: Validation using pytest fixtures
- **Error handling tests**: Input validation and edge cases

### Dependencies and Constraints

- **Python 3.9+**: Uses modern type hints and features
- **Core dependencies**: numpy, networkx, pandas
- **Development tools**: pytest, ruff (linting/formatting), mkdocs
- **Graph connectivity**: Quantum graphs must be connected for distance precomputing
- **Edge validation**: All edges must have positive `length` attributes