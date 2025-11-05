"""Simple test to verify that at least shuffling is reproducible."""

import random
import numpy as np
from kmeanssa_ng.quantum_graph.generators import generate_simple_graph
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.core.simulated_annealing import SimulatedAnnealing

# Generate data once
random.seed(1000)
np.random.seed(1000)
graph = generate_simple_graph(n_a=2)
points = graph.sample_points(20, strategy=UniformNodeSampling())

print("Original points (first 5):")
for i, p in enumerate(points[:5]):
    print(f"  {i}: {p}")

# Test 1: Create SA with seed 42
sa1 = SimulatedAnnealing(points, k=3, random_state=42)
print("\nSA with seed=42, shuffled observations (first 5):")
for i, p in enumerate(sa1.observations[:5]):
    print(f"  {i}: {p}")

# Test 2: Create another SA with same seed
sa2 = SimulatedAnnealing(points, k=3, random_state=42)
print("\nAnother SA with seed=42, shuffled observations (first 5):")
for i, p in enumerate(sa2.observations[:5]):
    print(f"  {i}: {p}")

# Check if shuffles are identical
shuffles_identical = all(
    p1 == p2 for p1, p2 in zip(sa1.observations, sa2.observations)
)
print(f"\n✅ Shuffle reproductible: {shuffles_identical}")

if not shuffles_identical:
    print("\n❌ Les shuffles ne sont PAS identiques!")
    print("Vérification détaillée des 5 premiers:")
    for i in range(5):
        match = "✓" if sa1.observations[i] == sa2.observations[i] else "✗"
        print(f"  {match} Position {i}: sa1={sa1.observations[i]}, sa2={sa2.observations[i]}")
