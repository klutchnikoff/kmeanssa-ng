# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.3.0...main
[0.3.0]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.2.3...v0.3.0
[0.2.3]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.2.2...v0.2.3
[0.2.2]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.2.1...v0.2.2
[0.2.1]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/compare/v0.1.0...v0.2.1
[0.1.0]: https://plmlab.math.cnrs.fr/nicolas.klutchnikoff/kmeanssa-ng/-/tags/v0.1.0
