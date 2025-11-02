# Future Refactoring: Sampling Strategies Pattern

**Date**: 2025-11-02
**Status**: Planned (not yet implemented)
**Branch**: To be created after merging `riemannian-manifold`

## Context

During the implementation of the `riemannian_manifold` module, we identified an opportunity to refactor sampling across both `QuantumGraph` and `RiemannianManifold` using a **Strategy Pattern**, similar to what exists for initialization (`InitializationStrategy`) and robustification (`RobustificationStrategy`).

## Current Implementation

### QuantumGraph
- `sample_points(n, where="Node")` supports:
  - **Node sampling**: Uses `weight` attribute for weighted sampling
  - **Edge sampling**: Uses `weight` and `distribution` attributes
  - Tracks `nb_obs` (number of observations) on nodes

### RiemannianManifold
- `sample_points(n)` currently only supports:
  - **Uniform sampling**: Uses `manifold.random_uniform()`
- Stores observations in `self.observations` for energy calculation

## Proposed Architecture

### 1. Abstract Strategy Class

```python
# In core/strategies/sampling.py (NEW FILE)
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

P = TypeVar('P', bound='Point')

class SamplingStrategy(ABC, Generic[P]):
    """Abstract base class for sampling strategies."""

    @abstractmethod
    def sample(self, space: Space, n: int) -> list[P]:
        """Sample n points from the space.

        Args:
            space: The space to sample from
            n: Number of points to sample

        Returns:
            List of n sampled points
        """
        raise NotImplementedError
```

### 2. Core Space API

```python
# In core/abstract.py
class Space(ABC):

    def __init__(self):
        from .strategies.sampling import UniformSampling
        self._sampling_strategy = UniformSampling()

    @abstractmethod
    def _sample_uniform(self, n: int) -> list[Point]:
        """Sample n points uniformly. MUST be implemented by subclasses."""
        raise NotImplementedError

    def set_sampling_strategy(self, strategy: SamplingStrategy) -> None:
        """Set the sampling strategy.

        Args:
            strategy: A SamplingStrategy instance
        """
        self._sampling_strategy = strategy

    def sample_points(self, n: int) -> list[Point]:
        """Sample n points using the configured strategy."""
        return self._sampling_strategy.sample(self, n)
```

### 3. Base Strategy Implementation

```python
# In core/strategies/sampling.py
class UniformSampling(SamplingStrategy[P]):
    """Uniform sampling strategy (uses space's _sample_uniform)."""

    def sample(self, space: Space, n: int) -> list[P]:
        return space._sample_uniform(n)
```

### 4. QuantumGraph Strategies

```python
# In quantum_graph/sampling_strategies.py (NEW FILE)
from ..core.strategies.sampling import SamplingStrategy
from .point import QGPoint
import networkx as nx

class WeightedNodeSampling(SamplingStrategy[QGPoint]):
    """Sample from nodes using weight attributes."""

    def __init__(self, node_weights: dict = None):
        self.node_weights = node_weights or {}

    def sample(self, space: QuantumGraph, n: int) -> list[QGPoint]:
        if self.node_weights:
            nx.set_node_attributes(space, self.node_weights, "weight")
        nx.set_node_attributes(space, 0, "nb_obs")
        return [space._sample_point("Node") for _ in range(n)]


class WeightedEdgeSampling(SamplingStrategy[QGPoint]):
    """Sample from edges using weight and distribution attributes."""

    def __init__(self, edge_weights: dict = None, edge_distributions: dict = None):
        self.edge_weights = edge_weights or {}
        self.edge_distributions = edge_distributions or {}

    def sample(self, space: QuantumGraph, n: int) -> list[QGPoint]:
        if self.edge_weights:
            nx.set_edge_attributes(space, self.edge_weights, "weight")
        if self.edge_distributions:
            nx.set_edge_attributes(space, self.edge_distributions, "distribution")
        return [space._sample_point("Edge") for _ in range(n)]
```

### 5. RiemannianManifold Strategies

```python
# In riemannian_manifold/sampling_strategies.py (NEW FILE)
from ..core.strategies.sampling import SamplingStrategy
from .point import RiemannianPoint
import numpy as np

class VonMisesFisherSampling(SamplingStrategy[RiemannianPoint]):
    """Von Mises-Fisher sampling on manifolds."""

    def __init__(self, mu: np.ndarray, kappa: float):
        self.mu = mu
        self.kappa = kappa

    def sample(self, space: RiemannianManifold, n: int) -> list[RiemannianPoint]:
        coords = space.manifold.random_von_mises_fisher(
            mu=self.mu,
            kappa=self.kappa,
            n_samples=n
        )
        return [RiemannianPoint(space, coords[i]) for i in range(n)]


class RiemannianNormalSampling(SamplingStrategy[RiemannianPoint]):
    """Riemannian normal distribution sampling."""

    def __init__(self, mean: np.ndarray, precision: np.ndarray):
        self.mean = mean
        self.precision = precision

    def sample(self, space: RiemannianManifold, n: int) -> list[RiemannianPoint]:
        coords = space.manifold.random_riemannian_normal(
            mean=self.mean,
            precision=self.precision,
            n_samples=n
        )
        return [RiemannianPoint(space, coords[i]) for i in range(n)]


class CustomDensitySampling(SamplingStrategy[RiemannianPoint]):
    """Rejection sampling with custom density function."""

    def __init__(self, density_fn):
        """
        Args:
            density_fn: Function coord -> weight (unnormalized probability)
        """
        self.density_fn = density_fn

    def sample(self, space: RiemannianManifold, n: int) -> list[RiemannianPoint]:
        coords = []
        # Find max density for rejection sampling
        test_samples = space.manifold.random_uniform(n_samples=1000)
        max_density = max(self.density_fn(p) for p in test_samples)

        while len(coords) < n:
            candidate = space.manifold.random_uniform(n_samples=1)[0]
            density = self.density_fn(candidate)
            if np.random.random() < density / max_density:
                coords.append(candidate)

        return [RiemannianPoint(space, c) for c in coords]
```

## Usage Examples

### QuantumGraph

```python
# Default: uniform node sampling
graph = QuantumGraph()
points = graph.sample_points(100)

# Weighted node sampling
graph.set_sampling_strategy(WeightedNodeSampling(node_weights={0: 0.5, 1: 0.3, 2: 0.2}))
points = graph.sample_points(100)

# Weighted edge sampling
graph.set_sampling_strategy(WeightedEdgeSampling(
    edge_weights={(0,1): 0.6, (1,2): 0.4},
    edge_distributions={(0,1): lambda: np.random.uniform(0, 1)}
))
points = graph.sample_points(100)
```

### RiemannianManifold

```python
# Default: uniform sampling
sphere = create_sphere(dim=2)
points = sphere.sample_points(100)

# Von Mises-Fisher (concentrated around a point)
mu = np.array([1, 0, 0])
sphere.set_sampling_strategy(VonMisesFisherSampling(mu=mu, kappa=10))
points = sphere.sample_points(100)

# Riemannian Normal
sphere.set_sampling_strategy(RiemannianNormalSampling(mean=mu, precision=...))
points = sphere.sample_points(100)

# Custom density (user-defined)
def my_density(coord):
    # Higher density in northern hemisphere
    return 1.0 if coord[2] > 0 else 0.1

sphere.set_sampling_strategy(CustomDensitySampling(density_fn=my_density))
points = sphere.sample_points(100)
```

## Benefits

1. **Extensibility**: Users can create custom sampling strategies
2. **Consistency**: Same API pattern as `InitializationStrategy` and `RobustificationStrategy`
3. **Flexibility**: Easy to add new distributions without modifying core classes
4. **Type Safety**: Generic typing ensures type correctness
5. **Testability**: Each strategy can be tested independently
6. **Separation of Concerns**: Sampling logic separated from space logic

## Implementation Plan

### Phase 1: Core Infrastructure
1. Create `core/strategies/sampling.py` with abstract base class
2. Add `_sample_uniform()` abstract method to `Space`
3. Implement default `set_sampling_strategy()` and `sample_points()` in `Space`

### Phase 2: QuantumGraph Refactoring
1. Implement `_sample_uniform()` in `QuantumGraph` (using existing `light_sample_points`)
2. Create `quantum_graph/sampling_strategies.py`
3. Implement `WeightedNodeSampling` and `WeightedEdgeSampling`
4. Update existing code to use new API (backward compatible)
5. Add tests for new strategies

### Phase 3: RiemannianManifold Extension
1. Implement `_sample_uniform()` in `RiemannianManifold` (already done via `sample_points`)
2. Create `riemannian_manifold/sampling_strategies.py`
3. Implement `VonMisesFisherSampling`, `RiemannianNormalSampling`, `CustomDensitySampling`
4. Add tests for new strategies
5. Update documentation

### Phase 4: Documentation & Examples
1. Update user documentation with examples
2. Add cookbook entries for common sampling patterns
3. Create migration guide for existing code

## Testing Strategy

- Unit tests for each strategy class
- Integration tests with `SimulatedAnnealing`
- Parametrized tests across different spaces
- Benchmark tests to ensure performance

## Open Questions

1. Should `_sample_uniform` be mandatory or have a default implementation?
2. How to handle `observations` storage for energy calculation?
   - Option A: Strategies handle storage
   - Option B: `Space` handles storage after strategy returns points
3. Should we support "stacking" strategies (e.g., weighted + custom density)?
4. Naming: `SamplingStrategy` vs `DistributionStrategy` vs `PointSampler`?

## Related Issues

- Consistent handling of probability measures across spaces
- Energy calculation modes ("uniform" vs "obs")
- Potential for batch sampling optimizations

## References

- Pattern used in `core/strategies/initialization.py`
- Pattern used in `core/strategies/robustification.py`
- Discussion on 2025-11-02 during riemannian_manifold implementation
