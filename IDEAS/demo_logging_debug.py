"""Demo script to show DEBUG logging."""

import logging

# Configure logging to see DEBUG messages
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from kmeanssa_ng import QuantumGraph
from kmeanssa_ng.core.strategies import KMeansPlusPlus, MinimizeEnergy
from kmeanssa_ng.core.simulated_annealing import SimulatedAnnealing
from kmeanssa_ng.quantum_graph.generators import generate_simple_graph
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

print("=" * 80)
print("DEMO: Logging with DEBUG level (very detailed!)")
print("=" * 80)

# Create a small graph for demo (only 10 points to avoid too much output)
graph = generate_simple_graph(n_a=2)
points = graph.sample_points(10, strategy=UniformNodeSampling())

# Run simulated annealing - you'll see DEBUG logs
sa = SimulatedAnnealing(points, k=2, lambda0=1.0, beta0=2.0, step_size=0.01)

print("\n--- Running Interleaved SA with DEBUG logging ---")
centers = sa.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

print(f"\nResult: {len(centers)} centers found")
