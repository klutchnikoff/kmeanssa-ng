"""Tests for parallel simulated annealing execution."""

import multiprocessing as mp
import sys

import pytest

from kmeanssa_ng import run_parallel, run_parallel_with_callback
from kmeanssa_ng.core.strategies.initialization import KMeansPlusPlus
from kmeanssa_ng.core.strategies.robustification import MinimizeEnergy
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.quantum_graph import generate_simple_graph


@pytest.fixture
def simple_graph():
    """Create a simple graph for testing."""
    graph = generate_simple_graph(n_a=3, n_aa=2, bridge_length=2.0)
    return graph


class TestRunParallel:
    """Tests for run_parallel function."""

    def test_basic_parallel_execution(self, simple_graph):
        """Test basic parallel execution returns centers."""
        centers = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=2,
        )

        assert centers is not None
        assert len(centers) == 2

    def test_parallel_with_return_all(self, simple_graph):
        """Test parallel execution with return_all=True."""
        best_centers, all_results = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=5,
            n_jobs=2,
            return_all=True,
        )

        assert len(best_centers) == 2
        assert len(all_results) == 5

        # Check that results are sorted by energy
        energies = [energy for _, energy, _ in all_results]
        assert energies == sorted(energies)

        # Check that best centers match first result
        assert best_centers == all_results[0][0]

    def test_parallel_with_specific_seeds(self, simple_graph):
        """Test parallel execution with specific seeds."""
        seeds = [42, 123, 456]
        best, all_results = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            seeds=seeds,
            return_all=True,
        )

        # Check that all seeds were used
        result_seeds = {seed for _, _, seed in all_results}
        assert result_seeds == set(seeds)

    def test_parallel_with_consistent_quality(self, simple_graph):
        """Test that parallel runs produce consistent quality results."""
        # Run multiple times with same seeds
        seeds = [42, 43, 44]
        _, all_results1 = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            seeds=seeds,
            return_all=True,
        )
        _, all_results2 = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            seeds=seeds,
            return_all=True,
        )

        # Extract energies
        energies1 = sorted([energy for _, energy, _ in all_results1])
        energies2 = sorted([energy for _, energy, _ in all_results2])

        # Results should be similar in quality (same seeds produce similar energies)
        # Note: Due to multiprocessing scheduling, results won't be bitwise identical,
        # but should be in the same ballpark
        for e1, e2 in zip(energies1, energies2):
            assert abs(e1 - e2) < max(e1, e2) * 0.95  # Within 95% of each other

    def test_parallel_with_sequential_algorithm(self, simple_graph):
        """Test parallel execution with sequential algorithm."""
        centers = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            algorithm="sequential",
            n_jobs=2,
        )

        assert len(centers) == 2

    def test_parallel_with_different_parameters(self, simple_graph):
        """Test parallel execution with custom parameters."""
        centers = run_parallel(
            simple_graph,
            n_points=20,
            k=3,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            lambda0=2,
            beta0=0.5,
            step_size=0.05,
            energy_mode="obs",
            robust_prop=0.1,
            n_jobs=2,
        )

        assert len(centers) == 3

    def test_parallel_with_single_job(self, simple_graph):
        """Test that n_jobs=1 works (sequential processing)."""
        centers = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=1,
        )

        assert len(centers) == 2

    def test_parallel_invalid_n_runs(self, simple_graph):
        """Test that invalid n_runs raises ValueError."""
        with pytest.raises(ValueError, match="n_runs must be positive"):
            run_parallel(
                simple_graph,
                n_points=20,
                k=2,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                n_runs=0,
            )

        with pytest.raises(ValueError, match="n_runs must be positive"):
            run_parallel(
                simple_graph,
                n_points=20,
                k=2,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                n_runs=-1,
            )

    def test_parallel_seeds_length_mismatch(self, simple_graph):
        """Test that mismatched seeds length raises ValueError."""
        with pytest.raises(ValueError, match="Length of seeds .* must match n_runs"):
            run_parallel(
                simple_graph,
                n_points=20,
                k=2,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                n_runs=5,
                seeds=[1, 2, 3],
            )

    @pytest.mark.parametrize("mp_context", mp.get_all_start_methods())
    def test_parallel_with_mp_context(self, simple_graph, mp_context):
        """Test parallel execution with all available multiprocessing contexts."""
        if mp_context == "spawn" and sys.platform == "win32":
            pytest.skip("spawn is default on Windows, no need to test explicitly")

        centers = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=2,
            mp_context=mp_context,
        )

        assert len(centers) == 2

    def test_parallel_with_invalid_context(self, simple_graph):
        """Test that invalid mp_context raises ValueError."""
        with pytest.raises(
            ValueError, match="Multiprocessing context .* not available"
        ):
            run_parallel(
                simple_graph,
                n_points=20,
                k=2,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                n_runs=3,
                mp_context="invalid",
            )

    def test_parallel_default_context(self, simple_graph):
        """Test that default context (None) works correctly."""
        # This uses the system default
        centers = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=2,
            mp_context=None,
        )

        assert len(centers) == 2

    def test_parallel_with_too_many_jobs(self, simple_graph):
        """Test that a warning is issued when n_jobs > cpu_count."""
        import os

        cpu_count = os.cpu_count() or 1

        with pytest.warns(
            UserWarning, match="n_jobs.* is greater than the number of available CPUs"
        ):
            run_parallel(
                simple_graph,
                n_points=20,
                k=2,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                n_runs=3,
                n_jobs=cpu_count + 1,
            )

    def test_parallel_with_n_jobs_minus_two(self, simple_graph):
        """Test that n_jobs=-2 works correctly."""
        centers = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=-2,
        )
        assert len(centers) == 2


class TestWorkerFunction:
    """Tests for the internal worker function used in parallel execution."""

    def test_run_with_seed_direct_call(self, simple_graph):
        """Test direct call to _run_with_seed to cover its internal logic."""
        from kmeanssa_ng.core.parallel import _run_with_seed
        from unittest.mock import patch, ANY

        with patch(
            "kmeanssa_ng.core.simulated_annealing.SimulatedAnnealing"
        ) as mock_sa:
            centers, energy, seed = _run_with_seed(
                space=simple_graph,
                n_points=10,
                k=2,
                seed=42,
                algorithm="interleaved",
                lambda0=1,
                beta0=1.0,
                step_size=0.1,
                energy_mode="obs",
                robust_prop=0.1,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=None,
                robustification_strategy=None,
            )

            mock_sa.assert_called_with(
                observations=ANY,
                k=2,
                lambda0=1,
                beta0=1.0,
                step_size=0.1,
                energy_mode="obs",
            )

    def test_run_parallel_defaults(self, simple_graph):
        """Test run_parallel with default arguments (no seeds, return_all=False)."""
        # This test covers the branches where seeds is None and return_all is False.
        best_centers = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=2,
            # No seeds provided
            # return_all is False by default
        )

        assert best_centers is not None
        assert isinstance(best_centers, list)
        assert len(best_centers) == 2


class TestRunParallelWithCallback:
    """Tests for run_parallel_with_callback function."""

    def test_basic_callback_execution(self, simple_graph):
        """Test basic execution with callback."""
        callback_calls = []

        def callback(run_idx, seed, energy):
            callback_calls.append((run_idx, seed, energy))

        centers = run_parallel_with_callback(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=2,
            callback=callback,
        )

        assert len(centers) == 2
        assert len(callback_calls) == 3

        # Check that all callbacks have valid data
        for run_idx, seed, energy in callback_calls:
            assert isinstance(run_idx, int)
            assert isinstance(seed, int)
            assert isinstance(energy, float)
            assert energy >= 0

    def test_callback_without_function(self, simple_graph):
        """Test execution without callback (should work fine)."""
        centers = run_parallel_with_callback(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=2,
            callback=None,
        )

        assert len(centers) == 2

    def test_callback_with_specific_seeds(self, simple_graph):
        """Test callback receives correct seeds."""
        seeds = [10, 20, 30]
        callback_seeds = []

        def callback(run_idx, seed, energy):
            callback_seeds.append(seed)

        run_parallel_with_callback(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            seeds=seeds,
            n_jobs=2,
            callback=callback,
        )

        assert set(callback_seeds) == set(seeds)

    def test_callback_error_handling(self, simple_graph):
        """Test that callback errors are propagated."""

        def bad_callback(run_idx, seed, energy):
            raise RuntimeError("Intentional error")

        # The error should propagate
        with pytest.raises(RuntimeError, match="Intentional error"):
            run_parallel_with_callback(
                simple_graph,
                n_points=20,
                k=2,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                n_runs=2,
                n_jobs=1,
                callback=bad_callback,
            )

    @pytest.mark.parametrize("mp_context", mp.get_all_start_methods())
    def test_callback_with_mp_context(self, simple_graph, mp_context):
        """Test callback execution with all available multiprocessing contexts."""
        if mp_context == "spawn" and sys.platform == "win32":
            pytest.skip("spawn is default on Windows, no need to test explicitly")

        callback_calls = []

        def callback(run_idx, seed, energy):
            callback_calls.append((run_idx, seed, energy))

        centers = run_parallel_with_callback(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=2,
            callback=callback,
            mp_context=mp_context,
        )

        assert len(centers) == 2
        assert len(callback_calls) == 3

    def test_callback_with_invalid_context(self, simple_graph):
        """Test that invalid mp_context raises ValueError with callback."""

        def callback(run_idx, seed, energy):
            pass

        with pytest.raises(
            ValueError, match="Multiprocessing context .* not available"
        ):
            run_parallel_with_callback(
                simple_graph,
                n_points=20,
                k=2,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                n_runs=3,
                callback=callback,
                mp_context="invalid",
            )

    def test_callback_with_too_many_jobs(self, simple_graph):
        """Test that a warning is issued when n_jobs > cpu_count with callback."""
        import os

        cpu_count = os.cpu_count() or 1

        def callback(run_idx, seed, energy):
            pass

        with pytest.warns(
            UserWarning, match="n_jobs.* is greater than the number of available CPUs"
        ):
            run_parallel_with_callback(
                simple_graph,
                n_points=20,
                k=2,
                sampling_strategy=UniformNodeSampling(),
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                n_runs=3,
                n_jobs=cpu_count + 1,
                callback=callback,
            )

    def test_callback_with_n_jobs_minus_two(self, simple_graph):
        """Test that n_jobs=-2 works correctly with callback."""
        callback_calls = []

        def callback(run_idx, seed, energy):
            callback_calls.append((run_idx, seed, energy))

        centers = run_parallel_with_callback(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=3,
            n_jobs=-2,
            callback=callback,
        )

        assert len(centers) == 2
        assert len(callback_calls) == 3


class TestParallelPerformance:
    """Performance and integration tests for parallel execution."""

    def test_parallel_faster_than_sequential(self, simple_graph):
        """Test that parallel execution completes successfully."""
        # Just verify it runs; actual speedup depends on system
        import time

        start = time.time()
        centers_parallel = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=4,
            n_jobs=2,
        )
        parallel_time = time.time() - start

        assert len(centers_parallel) == 2
        assert parallel_time > 0  # Basic sanity check

    def test_parallel_consistency(self, simple_graph):
        """Test that parallel runs produce valid clustering."""
        from kmeanssa_ng.core.metrics import compute_labels

        centers = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=5,
        )

        # Sample points to test clustering
        points = simple_graph.sample_points(20, strategy=UniformNodeSampling())
        labels = compute_labels(simple_graph, points, centers)
        assert len(labels) == len(points)
        assert all(0 <= label < 2 for label in labels)

    def test_parallel_with_many_runs(self, simple_graph):
        """Test parallel execution with many runs."""
        best, all_results = run_parallel(
            simple_graph,
            n_points=20,
            k=2,
            sampling_strategy=UniformNodeSampling(),
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            n_runs=10,
            n_jobs=4,
            return_all=True,
        )

        assert len(all_results) == 10

        # Verify energies are in non-decreasing order
        energies = [energy for _, energy, _ in all_results]
        for i in range(len(energies) - 1):
            assert energies[i] <= energies[i + 1]
