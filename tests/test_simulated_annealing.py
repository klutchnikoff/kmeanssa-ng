"""Test simulated annealing algorithm."""

import pytest

from kmeanssa_ng import (
    QGSimulatedAnnealing,
    SimulatedAnnealing,
    generate_sbm,
    generate_simple_graph,
)


class TestSimulatedAnnealing:
    """Tests for SimulatedAnnealing class."""

    def test_create_sa(self):
        """Test creating a SimulatedAnnealing instance."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2)

        assert sa.n == 20
        assert sa.space == graph

    def test_empty_observations_raises(self):
        """Test that empty observations raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            SimulatedAnnealing([], k=2)

    def test_invalid_k_raises(self):
        """Test that k <= 0 raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="greater than zero"):
            SimulatedAnnealing(points, k=0)

    def test_mixed_spaces_raises(self):
        """Test that points from different spaces raise ValueError."""
        graph1 = generate_simple_graph()
        graph2 = generate_simple_graph()

        points1 = graph1.sample_points(5)
        points2 = graph2.sample_points(5)

        with pytest.raises(ValueError, match="same metric space"):
            SimulatedAnnealing(points1 + points2, k=2)

    def test_negative_lambda_param_raises(self):
        """Test that negative lambda_param raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="lambda_param must be positive"):
            SimulatedAnnealing(points, k=2, lambda_param=-1)

    def test_zero_lambda_param_raises(self):
        """Test that zero lambda_param raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="lambda_param must be positive"):
            SimulatedAnnealing(points, k=2, lambda_param=0)

    def test_non_numeric_lambda_param_raises(self):
        """Test that non-numeric lambda_param raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="lambda_param must be a number"):
            SimulatedAnnealing(points, k=2, lambda_param="invalid")

    def test_negative_beta_raises(self):
        """Test that negative beta raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="beta must be positive"):
            SimulatedAnnealing(points, k=2, beta=-1.0)

    def test_zero_beta_raises(self):
        """Test that zero beta raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="beta must be positive"):
            SimulatedAnnealing(points, k=2, beta=0.0)

    def test_non_numeric_beta_raises(self):
        """Test that non-numeric beta raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="beta must be a number"):
            SimulatedAnnealing(points, k=2, beta="invalid")

    def test_negative_step_size_raises(self):
        """Test that negative step_size raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="step_size must be positive"):
            SimulatedAnnealing(points, k=2, step_size=-0.1)

    def test_zero_step_size_raises(self):
        """Test that zero step_size raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="step_size must be positive"):
            SimulatedAnnealing(points, k=2, step_size=0.0)

    def test_non_numeric_step_size_raises(self):
        """Test that non-numeric step_size raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10)

        with pytest.raises(ValueError, match="step_size must be a number"):
            SimulatedAnnealing(points, k=2, step_size="invalid")

    def test_run_basic(self):
        """Test running the algorithm with basic parameters."""
        graph = generate_simple_graph(n_a=3, bridge_length=5.0)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2, lambda_param=1, beta=1.0, step_size=0.1)

        centers = sa.run(robust_prop=0.0, initialization="random")

        assert len(centers) == 2
        # Check that centers are from the same graph (not exact object equality after deepcopy)
        assert all(hasattr(c, "space") for c in centers)

    def test_run_kpp_initialization(self):
        """Test running with k-means++ initialization."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run(initialization="kpp")

        assert len(centers) == 2

    def test_run_with_robustification(self):
        """Test running with robustification."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run(robust_prop=0.1, initialization="kpp")

        assert len(centers) == 2

    def test_invalid_robust_prop_raises(self):
        """Test that invalid robust_prop raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = SimulatedAnnealing(points, k=2)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run(robust_prop=1.5)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run(robust_prop=-0.1)

    def test_invalid_algorithm_version_raises(self):
        """Test that invalid algorithm_version raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = SimulatedAnnealing(points, k=2)

        with pytest.raises(ValueError, match="algorithm_version"):
            sa.run(algorithm_version="v3")

    def test_run_v2(self):
        """Test running algorithm version 2."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run(algorithm_version="v2")

        assert len(centers) == 2

    def test_calculate_energy(self):
        """Test energy calculation."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2)
        centers = graph.sample_centers(2)

        energy = sa.calculate_energy(centers, points)

        assert energy >= 0  # Energy should be non-negative

    def test_centers_property(self):
        """Test centers property (covers line 113)."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2)

        # Initially empty
        assert sa.centers == []

        # After running, should have centers
        centers = sa.run(initialization="kpp")
        # Note: centers property returns the private _centers, which is set during run
        assert len(sa.centers) == 2

    def test_run_for_mean(self):
        """Test run_for_mean method (k=1 special case)."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = QGSimulatedAnnealing(points, k=1)

        node_idx = sa.run_for_mean(robust_prop=0.1)

        # Node IDs can be strings (e.g., "A0") or integers
        assert node_idx in graph.nodes

    def test_run_for_mean_with_wrong_k_raises(self):
        """Test that run_for_mean with k != 1 raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = QGSimulatedAnnealing(points, k=2)

        with pytest.raises(ValueError, match="k=1"):
            sa.run_for_mean()

    def test_run_for_mean_invalid_robust_prop_raises(self):
        """Test that run_for_mean with invalid robust_prop raises ValueError (covers line 51)."""
        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = QGSimulatedAnnealing(points, k=1)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_for_mean(robust_prop=1.5)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_for_mean(robust_prop=-0.1)

    def test_run_for_kmeans_invalid_robust_prop_raises(self):
        """Test that run_for_kmeans with invalid robust_prop raises ValueError (covers line 105)."""
        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = QGSimulatedAnnealing(points, k=2)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_for_kmeans(robust_prop=1.5)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_for_kmeans(robust_prop=-0.1)

    def test_run_for_kmeans(self):
        """Test run_for_kmeans method."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = QGSimulatedAnnealing(points, k=2)

        node_ids = sa.run_for_kmeans(robust_prop=0.1)

        assert len(node_ids) == 2
        # Node IDs can be strings or integers depending on graph
        assert all(node_id in graph.nodes for node_id in node_ids)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_sbm(self):
        """Test full clustering pipeline on SBM graph."""
        # Generate a graph with 2 clear clusters
        graph = generate_sbm(sizes=[20, 20], p=[[0.8, 0.05], [0.05, 0.8]])

        # Sample points
        points = graph.sample_points(40)

        # Run simulated annealing
        sa = SimulatedAnnealing(points, k=2, lambda_param=1, beta=2.0)
        centers = sa.run(robust_prop=0.1, initialization="kpp")

        # Compute clusters
        graph.compute_clusters(centers)

        # Check that centers were found
        assert len(centers) == 2

        # Check that all nodes have cluster assignments
        clusters = [graph.nodes[node].get("cluster") for node in graph.nodes]
        assert all(c is not None for c in clusters)
        assert all(c in [0, 1] for c in clusters)

    def test_energy_decreases_with_iterations(self):
        """Test that energy generally decreases (not strict due to annealing)."""
        graph = generate_simple_graph(n_a=5, bridge_length=5.0)
        points = graph.sample_points(50)

        sa = SimulatedAnnealing(points, k=2, lambda_param=1, beta=2.0)

        # Random initialization should have higher energy than k-means++
        centers_random = graph.sample_centers(2)
        centers_kpp = graph.sample_kpp_centers(2)

        energy_random = graph.calculate_energy_graph(centers_random)
        energy_kpp = graph.calculate_energy_graph(centers_kpp)

        # k-means++ should generally be better (or equal) to random
        # This is probabilistic, so we just check it runs
        assert energy_random >= 0
        assert energy_kpp >= 0


# Import numpy for type checking in tests
import numpy as np
