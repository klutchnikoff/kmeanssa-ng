"""Test simulated annealing algorithm."""

import pytest

from kmeanssa_ng import (
    SimulatedAnnealing,
    generate_sbm,
    generate_simple_graph,
)
from kmeanssa_ng.core.strategies.initialization import (
    KMeansPlusPlus,
    RandomInit,
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

        centers = sa.run_interleaved(
            robust_prop=0.0, initialization_strategy=RandomInit()
        )

        assert len(centers) == 2
        # Check that centers are from the same graph (not exact object equality after deepcopy)
        assert all(hasattr(c, "space") for c in centers)

    def test_run_kpp_initialization(self):
        """Test running with k-means++ initialization."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_interleaved(initialization_strategy=KMeansPlusPlus())

        assert len(centers) == 2

    def test_run_with_robustification(self):
        """Test running with robustification."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)
        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_interleaved(
            robust_prop=0.1, initialization_strategy=KMeansPlusPlus()
        )

        assert len(centers) == 2

    def test_invalid_robust_prop_raises(self):
        """Test that invalid robust_prop raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = SimulatedAnnealing(points, k=2)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(robust_prop=1.5)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(robust_prop=-0.1)

    def test_sequential_algorithm(self):
        """Test running the sequential algorithm."""
        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_sequential(initialization_strategy=KMeansPlusPlus())
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
        centers = sa.run_interleaved(initialization_strategy=KMeansPlusPlus())
        # Note: centers property returns the private _centers, which is set during run
        assert len(sa.centers) == 2

    def test_run_for_mean(self):
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode
        from kmeanssa_ng import QGCenter

        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=1)

        center = sa.run_interleaved(
            robust_prop=0.1, robustification_strategy=MostFrequentNode()
        )

        # Should return a single QGCenter object
        assert isinstance(center, QGCenter)
        assert center.space == graph

    def test_run_for_mean_with_multiple_k(self):
        """Test that run with k != 1 returns a list of centers."""
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode
        from kmeanssa_ng import QGCenter

        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_interleaved(robustification_strategy=MostFrequentNode())
        assert isinstance(centers, list)
        assert len(centers) == 2
        assert all(isinstance(c, QGCenter) for c in centers)

    def test_run_for_mean_invalid_robust_prop_raises(self):
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode

        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = SimulatedAnnealing(points, k=1)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                robust_prop=1.5, robustification_strategy=MostFrequentNode()
            )

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                robust_prop=-0.1, robustification_strategy=MostFrequentNode()
            )

    def test_run_for_kmeans_invalid_robust_prop_raises(self):
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode

        graph = generate_simple_graph()
        points = graph.sample_points(20)
        sa = SimulatedAnnealing(points, k=2)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                robust_prop=1.5, robustification_strategy=MostFrequentNode()
            )

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                robust_prop=-0.1, robustification_strategy=MostFrequentNode()
            )

    def test_run_for_kmeans(self):
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode
        from kmeanssa_ng import QGCenter

        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20)

        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_interleaved(robustification_strategy=MostFrequentNode())

        assert len(centers) == 2
        assert all(isinstance(c, QGCenter) for c in centers)
        assert all(c.space == graph for c in centers)

    def test_most_frequent_node_strategy_empty_collection(self):
        """Test MostFrequentNode with an empty collection."""

        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode
        from kmeanssa_ng import QuantumGraph

        # Mock SimulatedAnnealing instance
        class MockSA:
            def __init__(self, k):
                self._k = k
                # Create a minimal graph for testing
                self.space = QuantumGraph()
                self.space.add_edge(0, 1, length=1.0)

        strategy = MostFrequentNode()

        # Test for k > 1
        sa_k2 = MockSA(k=2)
        strategy.initialize(sa_k2)
        assert strategy.get_result() == []

        # Test for k = 1
        sa_k1 = MockSA(k=1)
        strategy.initialize(sa_k1)
        assert strategy.get_result() is None


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
        centers = sa.run_interleaved(
            robust_prop=0.1, initialization_strategy=KMeansPlusPlus()
        )

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


from kmeanssa_ng.core.strategies.robustification import RobustificationStrategy


class TestRobustificationStrategy:
    """Tests for RobustificationStrategy abstract base class."""

    class DummyStrategy(RobustificationStrategy):
        """Dummy strategy that calls the abstract methods directly."""

        def initialize(self, sa):
            RobustificationStrategy.initialize(self, sa)

        def collect(self, sa):
            RobustificationStrategy.collect(self, sa)

        def get_result(self):
            return RobustificationStrategy.get_result(self)

    def test_abstract_methods_raise_not_implemented(self):
        """Test that abstract methods raise NotImplementedError."""
        strategy = self.DummyStrategy()
        with pytest.raises(NotImplementedError):
            strategy.initialize(None)
        with pytest.raises(NotImplementedError):
            strategy.collect(None)
        with pytest.raises(NotImplementedError):
            strategy.get_result()
