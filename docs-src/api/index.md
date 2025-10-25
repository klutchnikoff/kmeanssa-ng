# API Reference

## Architecture Overview

kmeanssa-ng is organized around a three-layer abstraction pattern:

- **[`kmeanssa_ng.core`](core.md)**: Abstract base classes defining metric spaces, points, centers, and the simulated annealing algorithm
- **[`kmeanssa_ng.quantum_graph`](quantum_graph.md)**: Concrete implementation for metric graphs (quantum graphs) built on NetworkX
- **[`kmeanssa_ng.quantum_graph.generators`](generators.md)**: Utilities for creating test graphs (SBM, random graphs, etc.)

## Quick Navigation

- **Getting started?** → [`QuantumGraph`](quantum_graph.md#quantumgraph), [`QGSimulatedAnnealing`](quantum_graph.md#qgsimulatedannealing)
- **Building custom spaces?** → [`Space`](core.md#space), [`Point`](core.md#point), [`Center`](core.md#center)
- **Understanding the algorithm?** → [`SimulatedAnnealing`](core.md#simulatedannealing)
- **Generating test graphs?** → [Generators](generators.md)
