# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.0] - unreleased

Makes the graph/manifold geometry and the experimental pipeline correct enough
to back the companion paper, and prepares the package for archival and JOSS.
Contains breaking API changes.

### Added
- **Bolza surface** (genus 2, curvature −1) as a first-class, quotient-aware geodesic space.
- **Intrinsic ε-net graph construction** (`build_epsilon_net_graph`, `RepulsionNet`) — embedding-free, with an extrinsic-speedup variant.
- **Closed-form geodesic operations for the sphere** (a `Sphere` subclass), bypassing the geomstats overhead.
- Native **energy-history tracking** in `SimulatedAnnealing.run` (`record_energy`), and public quantum-graph node distances / `node_energy`.
- `register_observations` to set the per-node observation measure from a set of points.
- **Reproducible paper pipeline** under `examples/paper/`: one-command `reproduce.py` regenerating every figure and table (grid, SBM, sphere, Bolza, the rate-theorem toy, the geomstats overhead), with structured per-seed/-run/-method RNG, per-seed checkpoints and resume, `--seeds`/`--jobs` flags, and a `values.tex` feeding the article prose.
- JOSS `paper.md`/`paper.bib` and `CONTRIBUTING.md`.

### Changed
- **Breaking:** the per-node observation-measure attribute `nb_obs` is renamed **`obs_weight`** — it holds a measure weight, not a count.
- **Breaking:** `Space.calculate_energy(centers, how="uniform", observations=None)` — the reference measure is now explicit and observations are supplied by the algorithm rather than stored on the space; the documented convention is the **mean** squared distance to the nearest center.
- **Breaking:** `FrechetMeanUpdate` is renamed **`KarcherFrechetMean`** and runs an intrinsic Karcher iteration (no geomstats `FrechetMean`).
- **Breaking:** the `Center` ABC now requires `seed_rng()`; the annealer seeds centers through it instead of a private attribute.
- **Breaking:** `UniformManifoldSampling` raises `NotImplementedError` on manifolds without a generator-driven uniform sampler (e.g. hyperbolic); `RiemannianPoint` rejects off-manifold coordinates.
- The SA-based Fréchet mean now uses a genuine annealing schedule.
- `pdm.lock` is no longer tracked; the paper README's reproduction claim is scoped to same-machine determinism (numbers agree across machines up to BLAS/LAPACK numerics, not bit-for-bit).

### Fixed
- **Center dynamics:** Brownian and drift moves keep the center on-edge — no negative positions when crossing a vertex, and correct drift direction on reversed same-edge parametrisations.
- **Poisson clock:** correct time indexing (no observation silently excluded) and simulation of the residual interval.
- **Self-loop distances:** correct geodesic on loop edges (no spurious 0.0), across all distance kernels (now deduplicated).
- **Distance caches** are invalidated on graph edits, and a re-run `precomputing()` actually recomputes.
- `energy_mode="obs"` raises instead of silently returning 0.0 when no node carries a positive measure; the obs-energy kernel carries `obs_weight` as a float measure, so fractional population weights are no longer truncated to zero.
- **Lloyd** reseeds empty clusters (farthest point) instead of silently returning fewer than k centers.
- **Metric-correct Brownian motion** on manifolds via `random_tangent` (sphere and Bolza; unsupported manifolds raise).
- Paper protocol: equal restart budgets across methods, honest grid-timing attribution, and a harmonised population-measure selection criterion.
- Executable module-header examples; corrected `lambda0` docstring (it is the Poisson-clock intensity, not a Brownian scale factor).

### Removed
- **Breaking:** the `hash()`-feature Davies–Bouldin and Calinski–Harabasz metrics (meaningless on graph labels, and non-reproducible across processes).

## [0.7.0] - 2026-06-30

### Added
- Complete implementation of **Lloyd's Algorithm** for k-means clustering.
- Multiple Lloyd update strategies: `SimulatedAnnealingFrechetMean`, `MostFrequentNodeUpdate`, `MinimizeEnergyNodeUpdate`, and `FrechetMeanUpdate`.
- Propagated local generator (`random_state`) to all components (initialization, sampling strategies, and graph neighbor selections) for **100% reproducible results**.
- Pre-commit (linting/formatting with Ruff) and pre-push (testing with Pytest) hooks configuration.

### Fixed
- Fixed flakiness in Spherical Fréchet Mean update and Lloyd manifold tests by seeding state and adjusting distance tolerances.

## [0.5.0] - 2025-11-01

### Changed
- **Breaking:** Renamed `lambda_param` and `beta` arguments to `lambda0` and `beta0` in `SimulatedAnnealing` and parallel functions to clarify they are initial values.
- Improved parallel strategy handling by allowing strategy objects to be passed directly to `run_parallel`, enabling user-defined strategies.

### Refactor
- Renamed core energy calculation methods in `SimulatedAnnealing` for clarity:
    - `calculate_energy_for_centers` is now the primary public method `calculate_energy`.
    - The old `calculate_energy` is now `calculate_energy_fallback`.
- Refactored `MostFrequentNode` strategy to correctly implement the `RobustificationStrategy` interface.

### Fixed
- Added context validation to `MostFrequentNode` to ensure it is only used with `QuantumGraph` spaces.
- Corrected the entire test suite to align with recent refactoring and API changes, restoring full test pass rate.

### Docs
- Overhauled the benchmark documentation, separating user-facing performance results from the developer guide.
- Simplified the script for generating benchmark data from `pytest-benchmark` output.

## [0.3.0] - 2025-01-27

### Added
- `draw()` method for QuantumGraph visualization with matplotlib
- Performance benchmarks suite with pytest-benchmark
- Comprehensive quickstart guide and enhanced documentation
- Quarto-based documentation workflow integrated with MkDocs
- Contributing guidelines
- Citation section in README
- Logo and visualization assets

### Changed
- `MostFrequentNode` now returns `QGCenter` objects instead of node IDs
- Improved error handling in `get_edge_length` method
- Reorganized dependency groups in pyproject.toml (test dependencies)
- Enhanced ReadTheDocs configuration for Quarto rendering

### Fixed
- ReadTheDocs configuration and build process
- Documentation rendering pipeline (Quarto → Markdown → HTML)
- Formatting and linting issues

### Removed
- Redundant `QGSimulatedAnnealing` class (merged into base `SimulatedAnnealing`)

## [0.2.3] - 2025-01-23

### Fixed
- ReadTheDocs configuration (build → post_build transition)

## [0.2.2] - 2025-01-22

### Changed
- Updated tests for new `run_interleaved`/`run_sequential` API

## [0.2.1] - 2025-01-22

### Added
- Strategy pattern for initialization algorithms
- `KMeansPlusPlus`, `RandomCenters`, and `MostFrequentNode` strategies
- Quarto-based executable documentation
- GitLab CI pipeline with linting, formatting, and test coverage
- GitLab Pages documentation deployment
- PDM scripts for common tasks (format, check, benchmark)

### Changed
- Refactored simulated annealing to use strategy pattern
- Migrated documentation toolchain to Quarto
- Simplified README
- Improved test coverage to 99.7%

### Performance
- Replaced deepcopy with lightweight `clone()` for 3x speedup
- Added batch distance computation for 5-9% speedup
- Added Numba JIT compilation for 2x speedup on large datasets

### Fixed
- Added `sample_kpp_centers` abstract method to `Space(ABC)`
- Dependency conflict with quartodoc
- Coverage badge display

## [0.1.0] - 2025-01-19

### Added
- Initial implementation of k-means clustering on quantum graphs using simulated annealing
- Core abstract classes (`Point`, `Center`, `Space`) for extensible metric space support
- Quantum graph implementation with full edge-based point positioning
- Simulated annealing algorithm with Brownian motion and drift phases
- Two algorithm variants (v1: interleaved, v2: sequential)
- k-means++ initialization support
- Robustification through averaging of final iterations
- Graph generators:
  - Stochastic Block Model (SBM) with uniform and random edge lengths
  - Simple symmetric two-cluster graphs
  - Random graphs with Poisson branching
  - NetworkX graph conversion utilities
- Comprehensive input validation across all components
- Distance caching and precomputation for performance
- Specialized quantum graph methods for node-based clustering
- Complete type hints throughout the codebase
- Comprehensive test suite with 98% coverage
- Development tooling with ruff linting and formatting
- MkDocs documentation setup

### Technical Details
- Python 3.9+ support with modern type hints
- PDM package management
- NetworkX integration for graph operations
- NumPy and pandas dependencies for numerical operations
- MIT license

[0.8.0]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.7.0...main
[0.7.0]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.5.0...v0.7.0
[0.5.0]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.3.0...v0.5.0
[0.3.0]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.2.3...v0.3.0
[0.2.3]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.2.2...v0.2.3
[0.2.2]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.2.1...v0.2.2
[0.2.1]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.1.0...v0.2.1
[0.1.0]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/tags/v0.1.0
