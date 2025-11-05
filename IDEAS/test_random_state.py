"""Test script to verify random_state reproducibility."""

import numpy as np
import random
from kmeanssa_ng.quantum_graph.generators import generate_simple_graph
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.core.simulated_annealing import SimulatedAnnealing
from kmeanssa_ng.core.strategies import KMeansPlusPlus, MinimizeEnergy

print("=" * 80)
print("Testing random_state reproducibility")
print("=" * 80)

# Generate data with fixed seed (separate from SA tests)
random.seed(1000)
np.random.seed(1000)
graph = generate_simple_graph(n_a=3)
points = graph.sample_points(30, strategy=UniformNodeSampling())

print(f"\nGenerated {len(points)} points on graph with {graph.number_of_nodes()} nodes")
print("(Data generation used seed 1000 - separate from SA random_state tests)")

# Test 1: Same seed should give same results
print("\n--- Test 1: Same seed (42) should give identical results ---")

sa1 = SimulatedAnnealing(points, k=3, random_state=42)
centers1 = sa1.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

sa2 = SimulatedAnnealing(points, k=3, random_state=42)
centers2 = sa2.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

print(f"Run 1 - First center: {centers1[0]}")
print(f"Run 2 - First center: {centers2[0]}")

# Check if they're the same
same_positions = all(
    c1.position == c2.position and c1.edge == c2.edge
    for c1, c2 in zip(centers1, centers2)
)
print(f"✅ Identical results: {same_positions}")

# Test 2: Different seeds should give different results
print("\n--- Test 2: Different seeds should give different results ---")

sa3 = SimulatedAnnealing(points, k=3, random_state=123)
centers3 = sa3.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

print(f"Seed 42  - First center: {centers1[0]}")
print(f"Seed 123 - First center: {centers3[0]}")

different = any(
    c1.position != c3.position or c1.edge != c3.edge
    for c1, c3 in zip(centers1, centers3)
)
print(f"✅ Different results: {different}")

# Test 3: Using Generator directly
print("\n--- Test 3: Using np.random.Generator directly ---")

rng = np.random.default_rng(999)
sa4 = SimulatedAnnealing(points, k=3, random_state=rng)
centers4 = sa4.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

rng2 = np.random.default_rng(999)
sa5 = SimulatedAnnealing(points, k=3, random_state=rng2)
centers5 = sa5.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

print(f"Generator 999 (run 1) - First center: {centers4[0]}")
print(f"Generator 999 (run 2) - First center: {centers5[0]}")

same_with_gen = all(
    c4.position == c5.position and c4.edge == c5.edge
    for c4, c5 in zip(centers4, centers5)
)
print(f"✅ Identical with Generator: {same_with_gen}")

# Test 4: None should give different results each time
print("\n--- Test 4: None (no seed) should give different results ---")

sa6 = SimulatedAnnealing(points, k=3, random_state=None)
centers6 = sa6.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

sa7 = SimulatedAnnealing(points, k=3, random_state=None)
centers7 = sa7.run_interleaved(
    initialization_strategy=KMeansPlusPlus(),
    robustification_strategy=MinimizeEnergy(),
    robust_prop=0.1,
)

print(f"No seed (run 1) - First center: {centers6[0]}")
print(f"No seed (run 2) - First center: {centers7[0]}")

likely_different = any(
    c6.position != c7.position or c6.edge != c7.edge
    for c6, c7 in zip(centers6, centers7)
)
print(f"✅ Likely different (non-deterministic): {likely_different}")

print("\n" + "=" * 80)
print("All tests completed!")
print("=" * 80)
