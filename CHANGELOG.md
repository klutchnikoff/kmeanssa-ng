# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Initial changelog documentation

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

[Unreleased]: https://github.com/your-username/kmeanssa-ng/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/your-username/kmeanssa-ng/releases/tag/v0.1.0