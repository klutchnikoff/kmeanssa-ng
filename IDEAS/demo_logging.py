"""Demo script to show logging in action."""

import logging

# Configure logging to see INFO and DEBUG messages
logging.basicConfig(
    level=logging.INFO,  # Change to logging.DEBUG to see all details
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

from kmeanssa_ng import QuantumGraph
from kmeanssa_ng.core.strategies import KMeansPlusPlus, MinimizeEnergy
from kmeanssa_ng.core.simulated_annealing import SimulatedAnnealing
from kmeanssa_ng.quantum_graph.generators import generate_simple_graph
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

print("=" * 80)
print("DEMO: Logging with INFO level")
print("=" * 80)

# Create a simple graph and sample points
graph = generate_simple_graph(n_a=3)
points = graph.sample_points(50, strategy=UniformNodeSampling())

# Run simulated annealing - you'll see INFO logs
sa = SimulatedAnnealing(points, k=3, lambda0=1.0, beta0=2.0, step_size=0.01)

print("\n--- Running Interleaved SA ---")
centers = sa.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

print(f"\nResult: {len(centers)} centers found")

print("\n" + "=" * 80)
print("Now try running with DEBUG level to see detailed logs!")
print("Uncomment the line below and re-run:")
print("# logging.getLogger('kmeanssa_ng.core.simulated_annealing').setLevel(logging.DEBUG)")
print("=" * 80)
