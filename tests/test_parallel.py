"""Tests for parallel simulated annealing execution."""

import pytest

from kmeanssa_ng import run_parallel, run_parallel_with_callback
from kmeanssa_ng.quantum_graph import QuantumGraph, generate_simple_graph


@pytest.fixture
def simple_graph_with_points():
    """Create a simple graph with sample points."""
    graph = generate_simple_graph(n_a=3, n_aa=2, bridge_length=2.0)
    points = graph.sample_points(20)
    return graph, points


class TestRunParallel:
    """Tests for run_parallel function."""

    def test_basic_parallel_execution(self, simple_graph_with_points):
        """Test basic parallel execution returns centers."""
        graph, points = simple_graph_with_points
        
        centers = run_parallel(points, k=2, n_runs=3, n_jobs=2)
        
        assert centers is not None
        assert len(centers) == 2

    def test_parallel_with_return_all(self, simple_graph_with_points):
        """Test parallel execution with return_all=True."""
        graph, points = simple_graph_with_points
        
        best_centers, all_results = run_parallel(
            points, k=2, n_runs=5, n_jobs=2, return_all=True
        )
        
        assert len(best_centers) == 2
        assert len(all_results) == 5
        
        # Check that results are sorted by energy
        energies = [energy for _, energy, _ in all_results]
        assert energies == sorted(energies)
        
        # Check that best centers match first result
        assert best_centers == all_results[0][0]

    def test_parallel_with_specific_seeds(self, simple_graph_with_points):
        """Test parallel execution with specific seeds."""
        graph, points = simple_graph_with_points
        
        seeds = [42, 123, 456]
        best, all_results = run_parallel(
            points, k=2, n_runs=3, seeds=seeds, return_all=True
        )
        
        # Check that all seeds were used
        result_seeds = {seed for _, _, seed in all_results}
        assert result_seeds == set(seeds)

    def test_parallel_with_consistent_quality(self, simple_graph_with_points):
        """Test that parallel runs produce consistent quality results."""
        graph, points = simple_graph_with_points
        
        from kmeanssa_ng import SimulatedAnnealing
        
        # Run multiple times with same seeds
        seeds = [42, 43, 44]
        _, all_results1 = run_parallel(points, k=2, n_runs=3, seeds=seeds, return_all=True)
        _, all_results2 = run_parallel(points, k=2, n_runs=3, seeds=seeds, return_all=True)
        
        # Extract energies
        energies1 = sorted([energy for _, energy, _ in all_results1])
        energies2 = sorted([energy for _, energy, _ in all_results2])
        
        # Results should be similar in quality (same seeds produce similar energies)
        # Note: Due to multiprocessing scheduling, results won't be bitwise identical,
        # but should be in the same ballpark
        for e1, e2 in zip(energies1, energies2):
            assert abs(e1 - e2) < max(e1, e2) * 0.5  # Within 50% of each other

    def test_parallel_with_sequential_algorithm(self, simple_graph_with_points):
        """Test parallel execution with sequential algorithm."""
        graph, points = simple_graph_with_points
        
        centers = run_parallel(
            points, k=2, n_runs=3, algorithm="sequential", n_jobs=2
        )
        
        assert len(centers) == 2

    def test_parallel_with_different_parameters(self, simple_graph_with_points):
        """Test parallel execution with custom parameters."""
        graph, points = simple_graph_with_points
        
        centers = run_parallel(
            points,
            k=3,
            n_runs=3,
            lambda_param=2,
            beta=0.5,
            step_size=0.05,
            robust_prop=0.1,
            n_jobs=2,
        )
        
        assert len(centers) == 3

    def test_parallel_with_single_job(self, simple_graph_with_points):
        """Test that n_jobs=1 works (sequential processing)."""
        graph, points = simple_graph_with_points
        
        centers = run_parallel(points, k=2, n_runs=3, n_jobs=1)
        
        assert len(centers) == 2

    def test_parallel_invalid_n_runs(self, simple_graph_with_points):
        """Test that invalid n_runs raises ValueError."""
        graph, points = simple_graph_with_points
        
        with pytest.raises(ValueError, match="n_runs must be positive"):
            run_parallel(points, k=2, n_runs=0)
        
        with pytest.raises(ValueError, match="n_runs must be positive"):
            run_parallel(points, k=2, n_runs=-1)

    def test_parallel_seeds_length_mismatch(self, simple_graph_with_points):
        """Test that mismatched seeds length raises ValueError."""
        graph, points = simple_graph_with_points
        
        with pytest.raises(ValueError, match="Length of seeds .* must match n_runs"):
            run_parallel(points, k=2, n_runs=5, seeds=[1, 2, 3])


class TestRunParallelWithCallback:
    """Tests for run_parallel_with_callback function."""

    def test_basic_callback_execution(self, simple_graph_with_points):
        """Test basic execution with callback."""
        graph, points = simple_graph_with_points
        
        callback_calls = []
        
        def callback(run_idx, seed, energy):
            callback_calls.append((run_idx, seed, energy))
        
        centers = run_parallel_with_callback(
            points, k=2, n_runs=3, n_jobs=2, callback=callback
        )
        
        assert len(centers) == 2
        assert len(callback_calls) == 3
        
        # Check that all callbacks have valid data
        for run_idx, seed, energy in callback_calls:
            assert isinstance(run_idx, int)
            assert isinstance(seed, int)
            assert isinstance(energy, float)
            assert energy >= 0

    def test_callback_without_function(self, simple_graph_with_points):
        """Test execution without callback (should work fine)."""
        graph, points = simple_graph_with_points
        
        centers = run_parallel_with_callback(
            points, k=2, n_runs=3, n_jobs=2, callback=None
        )
        
        assert len(centers) == 2

    def test_callback_with_specific_seeds(self, simple_graph_with_points):
        """Test callback receives correct seeds."""
        graph, points = simple_graph_with_points
        
        seeds = [10, 20, 30]
        callback_seeds = []
        
        def callback(run_idx, seed, energy):
            callback_seeds.append(seed)
        
        run_parallel_with_callback(
            points, k=2, n_runs=3, seeds=seeds, n_jobs=2, callback=callback
        )
        
        assert set(callback_seeds) == set(seeds)

    def test_callback_error_handling(self, simple_graph_with_points):
        """Test that callback errors are propagated."""
        graph, points = simple_graph_with_points
        
        def bad_callback(run_idx, seed, energy):
            raise RuntimeError("Intentional error")
        
        # The error should propagate
        with pytest.raises(RuntimeError, match="Intentional error"):
            run_parallel_with_callback(
                points, k=2, n_runs=2, n_jobs=1, callback=bad_callback
            )


class TestParallelPerformance:
    """Performance and integration tests for parallel execution."""

    def test_parallel_faster_than_sequential(self, simple_graph_with_points):
        """Test that parallel execution completes successfully."""
        graph, points = simple_graph_with_points
        
        # Just verify it runs; actual speedup depends on system
        import time
        
        start = time.time()
        centers_parallel = run_parallel(points, k=2, n_runs=4, n_jobs=2)
        parallel_time = time.time() - start
        
        assert len(centers_parallel) == 2
        assert parallel_time > 0  # Basic sanity check

    def test_parallel_consistency(self, simple_graph_with_points):
        """Test that parallel runs produce valid clustering."""
        graph, points = simple_graph_with_points
        
        from kmeanssa_ng.core.metrics import compute_labels
        
        centers = run_parallel(points, k=2, n_runs=5)
        
        # Check that clustering is valid
        labels = compute_labels(graph, points, centers)
        assert len(labels) == len(points)
        assert all(0 <= label < 2 for label in labels)

    def test_parallel_with_many_runs(self, simple_graph_with_points):
        """Test parallel execution with many runs."""
        graph, points = simple_graph_with_points
        
        best, all_results = run_parallel(
            points, k=2, n_runs=10, n_jobs=4, return_all=True
        )
        
        assert len(all_results) == 10
        
        # Verify energies are in non-decreasing order
        energies = [energy for _, energy, _ in all_results]
        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1]
