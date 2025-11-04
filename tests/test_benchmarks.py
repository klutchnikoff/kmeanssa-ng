"""Performance benchmarks for kmeanssa-ng.

These tests measure the performance of critical operations to detect regressions.
Run with: pdm run pytest tests/test_benchmarks.py --benchmark-only
"""

import pytest

from kmeanssa_ng import (
    SimulatedAnnealing,
    generate_sbm,
)
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.core.strategies.initialization import KMeansPlusPlus
from kmeanssa_ng.core.strategies.robustification import MinimizeEnergy
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
        points = small_graph_precomputed.sample_points(
            5, strategy=UniformNodeSampling()
        )
        centers = [small_graph_precomputed.center_from_point(p) for p in points]
        target = small_graph_precomputed.sample_points(
            1, strategy=UniformNodeSampling()
        )[0]

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
        points = medium_graph_precomputed.sample_points(
            10, strategy=UniformNodeSampling()
        )
        centers = [medium_graph_precomputed.center_from_point(p) for p in points]
        target = medium_graph_precomputed.sample_points(
            1, strategy=UniformNodeSampling()
        )[0]

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
        points = small_graph_precomputed.sample_points(
            50, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=3)
        result = benchmark(KMeansPlusPlus().initialize_centers, sa)
        assert len(result) == 3

    def test_benchmark_kpp_initialization_medium(
        self, benchmark, medium_graph_precomputed
    ):
        """Benchmark k-means++ initialization (k=5, 100 nodes).

        Tests scaling of k-means++ with graph size.
        """
        points = medium_graph_precomputed.sample_points(
            150, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=5)
        result = benchmark(KMeansPlusPlus().initialize_centers, sa)
        assert len(result) == 5

    def test_benchmark_sa_interleaved_small(self, benchmark, small_graph_precomputed):
        """Benchmark interleaved SA algorithm on small graph (50 points, k=2).

        This is the main clustering algorithm.
        """
        points = small_graph_precomputed.sample_points(
            50, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=2, lambda0=1.0, beta0=1.0, step_size=0.1)

        result = benchmark(
            sa.run_interleaved,
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
        )
        assert len(result) == 2

    @pytest.mark.slow
    def test_benchmark_sa_interleaved_medium(self, benchmark, medium_graph_precomputed):
        """Benchmark interleaved SA algorithm on medium graph (150 points, k=3).

        This test is marked as slow and can be skipped with: -m "not slow"
        """
        points = medium_graph_precomputed.sample_points(
            150, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=3, lambda0=1.0, beta0=1.0, step_size=0.1)

        result = benchmark(
            sa.run_interleaved,
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
        )
        assert len(result) == 3

    def test_benchmark_sa_sequential_small(self, benchmark, small_graph_precomputed):
        """Benchmark sequential SA algorithm on small graph (50 points, k=2).

        Compares sequential vs interleaved algorithm performance.
        """
        points = small_graph_precomputed.sample_points(
            50, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=2, lambda0=1.0, beta0=1.0, step_size=0.1)

        result = benchmark(
            sa.run_sequential,
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
        )
        assert len(result) == 2

    @pytest.mark.slow
    def test_benchmark_sa_sequential_medium(self, benchmark, medium_graph_precomputed):
        """Benchmark sequential SA algorithm on medium graph (150 points, k=3).

        This test is marked as slow and can be skipped with: -m "not slow"
        """
        points = medium_graph_precomputed.sample_points(
            150, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=3, lambda0=1.0, beta0=1.0, step_size=0.1)

        result = benchmark(
            sa.run_sequential,
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
        )
        assert len(result) == 3

    @pytest.mark.slow
    def test_benchmark_sa_interleaved_mostfrequentnode_medium(
        self, benchmark, medium_graph_precomputed
    ):
        points = medium_graph_precomputed.sample_points(
            150, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=3, lambda0=1.0, beta0=1.0, step_size=0.1)

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
        points = medium_graph_precomputed.sample_points(
            150, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=10)
        return KMeansPlusPlus().initialize_centers(sa)

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


class TestRobustificationBenchmark:
    """Benchmark tests for robustification strategies.

    These benchmarks isolate and measure the cost of the robustification strategy
    by simulating: initialize() + 15×collect() + get_result()
    This represents approximately 10% robustification on 150 points.
    """

    @pytest.fixture
    def sa_prepared(self, medium_graph_precomputed):
        """Prepare a SA instance with initialized centers."""
        points = medium_graph_precomputed.sample_points(
            150, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(points, k=3, lambda0=1.0, beta0=1.0, step_size=0.1)
        # Initialize centers using k-means++
        sa._centers = KMeansPlusPlus().initialize_centers(sa)
        return sa

    @pytest.fixture
    def sa_prepared_obs(self, medium_graph_with_obs):
        """Prepare a SA instance with initialized centers for obs mode."""
        points = medium_graph_with_obs.sample_points(
            150, strategy=UniformNodeSampling()
        )
        sa = SimulatedAnnealing(
            points, k=3, lambda0=1.0, beta0=1.0, step_size=0.1, energy_mode="obs"
        )
        # Initialize centers using k-means++
        sa._centers = KMeansPlusPlus().initialize_centers(sa)
        return sa

    def test_benchmark_robustification_mostfrequentnode(self, benchmark, sa_prepared):
        """Benchmark isolated cost of MostFrequentNode strategy.

        Measures: initialize() + 15×collect() + get_result()
        Each collect() calls _closest_node() for k=3 centers.
        """

        def run_robustification_strategy():
            strategy = MostFrequentNode()
            strategy.initialize(sa_prepared)
            # Simulate 15 collect calls (10% of 150 points)
            for _ in range(15):
                strategy.collect(sa_prepared)
            result = strategy.get_result()
            return result

        result = benchmark(run_robustification_strategy)
        assert result is not None

    def test_benchmark_robustification_minimize_energy_uniform(
        self, benchmark, sa_prepared
    ):
        """Benchmark isolated cost of MinimizeEnergy with energy_mode='uniform'.

        Measures: initialize() + 15×collect() + get_result()
        Each collect() calls calculate_energy_numba() with mode='uniform'.
        """
        # Override energy mode to uniform
        sa_prepared._energy_mode = "uniform"

        def run_robustification_strategy():
            strategy = MinimizeEnergy()
            strategy.initialize(sa_prepared)
            # Simulate 15 collect calls (10% of 150 points)
            for _ in range(15):
                strategy.collect(sa_prepared)
            result = strategy.get_result()
            return result

        result = benchmark(run_robustification_strategy)
        assert result is not None

    def test_benchmark_robustification_minimize_energy_obs(
        self, benchmark, sa_prepared_obs
    ):
        """Benchmark isolated cost of MinimizeEnergy with energy_mode='obs'.

        Measures: initialize() + 15×collect() + get_result()
        Each collect() calls calculate_energy_numba() with mode='obs'.
        """

        def run_robustification_strategy():
            strategy = MinimizeEnergy()
            strategy.initialize(sa_prepared_obs)
            # Simulate 15 collect calls (10% of 150 points)
            for _ in range(15):
                strategy.collect(sa_prepared_obs)
            result = strategy.get_result()
            return result

        result = benchmark(run_robustification_strategy)
        assert result is not None

    def test_benchmark_robustification_minimize_energy_uniform_python(
        self, benchmark, sa_prepared
    ):
        """Benchmark MinimizeEnergy with pure Python energy calculation (uniform).

        Measures: initialize() + 15×collect() + get_result()
        Each collect() calls calculate_energy() (Python) with mode='uniform'.
        This test temporarily disables Numba acceleration.
        """
        # Override energy mode and monkey-patch to use Python version
        sa_prepared._energy_mode = "uniform"
        original_method = getattr(sa_prepared.space, "calculate_energy_numba", None)

        # Temporarily replace numba version with None to force Python fallback
        if original_method is not None:
            sa_prepared.space.calculate_energy_numba = None

        def run_robustification_strategy():
            strategy = MinimizeEnergy()
            strategy.initialize(sa_prepared)
            # Simulate 15 collect calls (10% of 150 points)
            for _ in range(15):
                strategy.collect(sa_prepared)
            result = strategy.get_result()
            return result

        try:
            result = benchmark(run_robustification_strategy)
            assert result is not None
        finally:
            # Restore numba method
            if original_method is not None:
                sa_prepared.space.calculate_energy_numba = original_method

    def test_benchmark_robustification_minimize_energy_obs_python(
        self, benchmark, sa_prepared_obs
    ):
        """Benchmark MinimizeEnergy with pure Python energy calculation (obs).

        Measures: initialize() + 15×collect() + get_result()
        Each collect() calls calculate_energy() (Python) with mode='obs'.
        This test temporarily disables Numba acceleration.
        """
        original_method = getattr(sa_prepared_obs.space, "calculate_energy_numba", None)

        # Temporarily replace numba version with None to force Python fallback
        if original_method is not None:
            sa_prepared_obs.space.calculate_energy_numba = None

        def run_robustification_strategy():
            strategy = MinimizeEnergy()
            strategy.initialize(sa_prepared_obs)
            # Simulate 15 collect calls (10% of 150 points)
            for _ in range(15):
                strategy.collect(sa_prepared_obs)
            result = strategy.get_result()
            return result

        try:
            result = benchmark(run_robustification_strategy)
            assert result is not None
        finally:
            # Restore numba method
            if original_method is not None:
                sa_prepared_obs.space.calculate_energy_numba = original_method
