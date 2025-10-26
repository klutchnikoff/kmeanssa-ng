"""Performance benchmarks for kmeanssa-ng.

These tests measure the performance of critical operations to detect regressions.
Run with: pdm run pytest tests/test_benchmarks.py --benchmark-only
"""

import pytest

from kmeanssa_ng import (
    SimulatedAnnealing,
    generate_sbm,
)
from kmeanssa_ng.core.strategies.initialization import KMeansPlusPlus
from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode


@pytest.fixture
def small_graph():
    """Small graph for quick benchmarks (40 nodes)."""
    graph = generate_sbm(sizes=[20, 20], p=[[0.8, 0.1], [0.1, 0.8]])
    return graph


@pytest.fixture
def small_graph_precomputed(small_graph):
    """Small graph with precomputed distances (40 nodes)."""
    small_graph.precomputing()
    return small_graph


@pytest.fixture
def medium_graph():
    """Medium graph for realistic benchmarks (100 nodes)."""
    graph = generate_sbm(sizes=[50, 50], p=[[0.8, 0.1], [0.1, 0.8]])
    return graph


@pytest.fixture
def medium_graph_precomputed(medium_graph):
    """Medium graph with precomputed distances (100 nodes)."""
    medium_graph.precomputing()
    return medium_graph


@pytest.fixture
def medium_graph_with_obs(medium_graph_precomputed):
    """Medium graph with observations for 'obs' mode benchmarks."""
    import networkx as nx
    import numpy as np

    graph = medium_graph_precomputed
    # Add some observations to the nodes
    for node in graph.nodes:
        nx.set_node_attributes(graph, {node: {"nb_obs": np.random.randint(0, 10)}})
    return graph


class TestBenchmarks:
    """Performance benchmark tests for critical operations."""

    def test_benchmark_precomputing_small(self, benchmark, small_graph):
        """Benchmark graph precomputing on small graph (40 nodes).

        This is a critical operation that caches all-pairs shortest paths.
        """
        result = benchmark(small_graph.precomputing)
        assert result is None

    def test_benchmark_precomputing_medium(self, benchmark, medium_graph):
        """Benchmark graph precomputing on medium graph (100 nodes).

        This tests scaling behavior of the precomputing step.
        """
        result = benchmark(medium_graph.precomputing)
        assert result is None

    def test_benchmark_batch_distances_small(self, benchmark, small_graph_precomputed):
        """Benchmark Numba-accelerated batch distance computation (5 centers).

        This operation is called repeatedly during simulated annealing.
        """
        centers = small_graph_precomputed.sample_centers(k=5)
        target = small_graph_precomputed.sample_points(n=1)[0]

        result = benchmark(
            small_graph_precomputed.distances_from_centers, centers, target
        )
        assert len(result) == 5

    def test_benchmark_batch_distances_medium(
        self, benchmark, medium_graph_precomputed
    ):
        """Benchmark batch distance computation on medium graph (10 centers).

        Tests scaling with more centers.
        """
        centers = medium_graph_precomputed.sample_centers(k=10)
        target = medium_graph_precomputed.sample_points(n=1)[0]

        result = benchmark(
            medium_graph_precomputed.distances_from_centers, centers, target
        )
        assert len(result) == 10

    def test_benchmark_kpp_initialization_small(
        self, benchmark, small_graph_precomputed
    ):
        """Benchmark k-means++ initialization (k=3, 40 nodes).

        This is used at the start of the simulated annealing algorithm.
        """
        result = benchmark(small_graph_precomputed.sample_kpp_centers, k=3)
        assert len(result) == 3

    def test_benchmark_kpp_initialization_medium(
        self, benchmark, medium_graph_precomputed
    ):
        """Benchmark k-means++ initialization (k=5, 100 nodes).

        Tests scaling of k-means++ with graph size.
        """
        result = benchmark(medium_graph_precomputed.sample_kpp_centers, k=5)
        assert len(result) == 5

    def test_benchmark_sa_interleaved_small(self, benchmark, small_graph_precomputed):
        """Benchmark interleaved SA algorithm on small graph (50 points, k=2).

        This is the main clustering algorithm.
        """
        points = small_graph_precomputed.sample_points(50)
        sa = SimulatedAnnealing(points, k=2, lambda_param=1, beta=1.0, step_size=0.1)

        result = benchmark(sa.run_interleaved, initialization_strategy=KMeansPlusPlus())
        assert len(result) == 2

    @pytest.mark.slow
    def test_benchmark_sa_interleaved_medium(self, benchmark, medium_graph_precomputed):
        """Benchmark interleaved SA algorithm on medium graph (150 points, k=3).

        This test is marked as slow and can be skipped with: -m "not slow"
        """
        points = medium_graph_precomputed.sample_points(150)
        sa = SimulatedAnnealing(points, k=3, lambda_param=1, beta=1.0, step_size=0.1)

        result = benchmark(sa.run_interleaved, initialization_strategy=KMeansPlusPlus())
        assert len(result) == 3

    def test_benchmark_sa_sequential_small(self, benchmark, small_graph_precomputed):
        """Benchmark sequential SA algorithm on small graph (50 points, k=2).

        Compares sequential vs interleaved algorithm performance.
        """
        points = small_graph_precomputed.sample_points(50)
        sa = SimulatedAnnealing(points, k=2, lambda_param=1, beta=1.0, step_size=0.1)

        result = benchmark(sa.run_sequential, initialization_strategy=KMeansPlusPlus())
        assert len(result) == 2

    @pytest.mark.slow
    def test_benchmark_sa_sequential_medium(self, benchmark, medium_graph_precomputed):
        """Benchmark sequential SA algorithm on medium graph (150 points, k=3).

        This test is marked as slow and can be skipped with: -m "not slow"
        """
        points = medium_graph_precomputed.sample_points(150)
        sa = SimulatedAnnealing(points, k=3, lambda_param=1, beta=1.0, step_size=0.1)

        result = benchmark(sa.run_sequential, initialization_strategy=KMeansPlusPlus())
        assert len(result) == 3

    @pytest.mark.slow
    def test_benchmark_sa_interleaved_mostfrequentnode_medium(
        self, benchmark, medium_graph_precomputed
    ):
        """Benchmark interleaved SA with MostFrequentNode strategy on medium graph."""
        points = medium_graph_precomputed.sample_points(150)
        sa = SimulatedAnnealing(points, k=3, lambda_param=1, beta=1.0, step_size=0.1)

        result = benchmark(
            sa.run_interleaved,
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MostFrequentNode(),
        )
        assert len(result) == 3


class TestEnergyCalculationBenchmark:
    """Benchmark tests for k-means energy calculation."""

    @pytest.fixture
    def centers_for_benchmark(self, medium_graph_precomputed):
        """Generate k=10 centers for the medium graph."""
        return medium_graph_precomputed.sample_kpp_centers(k=10)

    def test_benchmark_energy_numba_uniform(
        self, benchmark, medium_graph_precomputed, centers_for_benchmark
    ):
        """Benchmark Numba-accelerated energy calculation with how='uniform'."""
        benchmark(
            medium_graph_precomputed.calculate_energy_numba,
            centers_for_benchmark,
            how="uniform",
        )

    def test_benchmark_energy_python_uniform(
        self, benchmark, medium_graph_precomputed, centers_for_benchmark
    ):
        """Benchmark pure Python energy calculation with how='uniform'."""
        benchmark(
            medium_graph_precomputed.calculate_energy,
            centers_for_benchmark,
            how="uniform",
        )

    def test_benchmark_energy_python_obs(
        self, benchmark, medium_graph_with_obs, centers_for_benchmark
    ):
        """Benchmark pure Python energy calculation with how='obs'."""
        benchmark(
            medium_graph_with_obs.calculate_energy, centers_for_benchmark, how="obs"
        )

    def test_benchmark_energy_numba_obs(
        self, benchmark, medium_graph_with_obs, centers_for_benchmark
    ):
        """Benchmark Numba-accelerated energy calculation with how='obs'."""
        benchmark(
            medium_graph_with_obs.calculate_energy_numba,
            centers_for_benchmark,
            how="obs",
        )
