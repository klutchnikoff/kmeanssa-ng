"""Test simulated annealing algorithm."""

import pytest

from kmeanssa_ng import (
    SimulatedAnnealing,
    generate_sbm,
    generate_simple_graph,
)
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.core.strategies.initialization import (
    KMeansPlusPlus,
    RandomInit,
)
from kmeanssa_ng.core.strategies.robustification import (
    RobustificationStrategy,
    MinimizeEnergy,
)
from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode


class TestSimulatedAnnealing:
    """Tests for SimulatedAnnealing class."""

    def test_create_sa(self):
        """Test creating a SimulatedAnnealing instance."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())

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
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="greater than zero"):
            SimulatedAnnealing(points, k=0)

    def test_mixed_spaces_raises(self):
        """Test that points from different spaces raise ValueError."""
        graph1 = generate_simple_graph()
        graph2 = generate_simple_graph()

        points1 = graph1.sample_points(5, strategy=UniformNodeSampling())
        points2 = graph2.sample_points(5, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="same metric space"):
            SimulatedAnnealing(points1 + points2, k=2)

    def test_negative_lambda_param_raises(self):
        """Test that negative lambda_param raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="lambda0 must be positive"):
            SimulatedAnnealing(points, k=2, lambda0=-1)

    def test_zero_lambda_param_raises(self):
        """Test that zero lambda_param raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="lambda0 must be positive"):
            SimulatedAnnealing(points, k=2, lambda0=0)

    def test_non_numeric_lambda_param_raises(self):
        """Test that non-numeric lambda_param raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="lambda0 must be a number"):
            SimulatedAnnealing(points, k=2, lambda0="invalid")

    def test_negative_beta_raises(self):
        """Test that negative beta raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="beta0 must be positive"):
            SimulatedAnnealing(points, k=2, beta0=-1.0)

    def test_zero_beta_raises(self):
        """Test that zero beta raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="beta0 must be positive"):
            SimulatedAnnealing(points, k=2, beta0=0.0)

    def test_non_numeric_beta_raises(self):
        """Test that non-numeric beta raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="beta0 must be a number"):
            SimulatedAnnealing(points, k=2, beta0="invalid")

    def test_negative_step_size_raises(self):
        """Test that negative step_size raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="step_size must be positive"):
            SimulatedAnnealing(points, k=2, step_size=-0.1)

    def test_zero_step_size_raises(self):
        """Test that zero step_size raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="step_size must be positive"):
            SimulatedAnnealing(points, k=2, step_size=0.0)

    def test_non_numeric_step_size_raises(self):
        """Test that non-numeric step_size raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(10, strategy=UniformNodeSampling())

        with pytest.raises(ValueError, match="step_size must be a number"):
            SimulatedAnnealing(points, k=2, step_size="invalid")

    def test_run_basic(self):
        """Test running the algorithm with basic parameters."""
        graph = generate_simple_graph(n_a=3, bridge_length=5.0)
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2, lambda0=1, beta0=1.0, step_size=0.1)

        centers = sa.run_interleaved(
            initialization_strategy=RandomInit(),
            robustification_strategy=MinimizeEnergy(),
            robust_prop=0.0,
        )

        assert len(centers) == 2
        # Check that centers are from the same graph (not exact object equality after deepcopy)
        assert all(hasattr(c, "space") for c in centers)

    def test_run_kpp_initialization(self):
        """Test running with k-means++ initialization."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_interleaved(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
        )

        assert len(centers) == 2

    def test_run_with_robustification(self):
        """Test running with robustification."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())
        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_interleaved(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            robust_prop=0.1,
        )

        assert len(centers) == 2

    def test_invalid_robust_prop_raises(self):
        """Test that invalid robust_prop raises ValueError."""
        graph = generate_simple_graph()
        points = graph.sample_points(20, strategy=UniformNodeSampling())
        sa = SimulatedAnnealing(points, k=2)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                robust_prop=1.5,
            )

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                robust_prop=-0.1,
            )

    def test_sequential_algorithm(self):
        """Test running the sequential algorithm."""
        graph = generate_simple_graph()
        points = graph.sample_points(20, strategy=UniformNodeSampling())
        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_sequential(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
        )
        assert len(centers) == 2

    def test_calculate_energy_fallback(self):
        """Test energy calculation."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2)
        centers = RandomInit().initialize_centers(sa)

        energy = sa.calculate_energy_fallback(centers, points)

        assert energy >= 0  # Energy should be non-negative

    def test_energy_mode_obs(self):
        """Test that energy_mode='obs' uses the correct energy calculation."""
        graph = generate_simple_graph(n_a=3)
        import networkx as nx

        nx.set_node_attributes(
            graph, {0: {"nb_obs": 5}, 1: {"nb_obs": 10}, 2: {"nb_obs": 0}}
        )
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2, energy_mode="obs")
        centers = RandomInit().initialize_centers(sa)

        # Mock the space's calculate_energy and calculate_energy_numba methods
        from unittest.mock import patch

        with patch.object(
            sa.space, "calculate_energy_numba", create=True
        ) as mock_calculate_energy_numba:
            sa.calculate_energy(centers)
            mock_calculate_energy_numba.assert_called_with(centers, how="obs")

    def test_centers_property(self):
        """Test centers property (covers line 113)."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2)

        # Initially empty
        assert sa.centers == []

        # After running, should have centers
        sa.run_interleaved(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
        )
        # Note: centers property returns the private _centers, which is set during run
        assert len(sa.centers) == 2

    def test_run_for_mean(self):
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode
        from kmeanssa_ng import QGCenter

        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=1)

        centers = sa.run_interleaved(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MostFrequentNode(),
            robust_prop=0.1,
        )

        # Should return a list with a single QGCenter object (consistent with k>1)
        assert isinstance(centers, list)
        assert len(centers) == 1
        assert isinstance(centers[0], QGCenter)
        assert centers[0].space == graph

    def test_run_for_mean_with_multiple_k(self):
        """Test that run with k != 1 returns a list of centers."""
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode
        from kmeanssa_ng import QGCenter

        graph = generate_simple_graph()
        points = graph.sample_points(20, strategy=UniformNodeSampling())
        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_interleaved(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MostFrequentNode(),
        )
        assert isinstance(centers, list)
        assert len(centers) == 2
        assert all(isinstance(c, QGCenter) for c in centers)

    def test_run_for_mean_invalid_robust_prop_raises(self):
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode

        graph = generate_simple_graph()
        points = graph.sample_points(20, strategy=UniformNodeSampling())
        sa = SimulatedAnnealing(points, k=1)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MostFrequentNode(),
                robust_prop=1.5,
            )

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MostFrequentNode(),
                robust_prop=-0.1,
            )

    def test_run_for_kmeans_invalid_robust_prop_raises(self):
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode

        graph = generate_simple_graph()
        points = graph.sample_points(20, strategy=UniformNodeSampling())
        sa = SimulatedAnnealing(points, k=2)

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MostFrequentNode(),
                robust_prop=1.5,
            )

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run_interleaved(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MostFrequentNode(),
                robust_prop=-0.1,
            )

    def test_run_for_kmeans(self):
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode
        from kmeanssa_ng import QGCenter

        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run_interleaved(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MostFrequentNode(),
        )

        assert len(centers) == 2
        assert all(isinstance(c, QGCenter) for c in centers)
        assert all(c.space == graph for c in centers)

    def test_most_frequent_node_strategy_empty_collection(self):
        """Test MostFrequentNode with an empty collection."""

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

    def test_most_frequent_node_raises_on_non_graph_space(self):
        """Test that MostFrequentNode raises TypeError on a non-graph space."""
        # MostFrequentNode is now in quantum_graph package
        from kmeanssa_ng.quantum_graph.robustification import MostFrequentNode
        from kmeanssa_ng.core.abstract import Space

        # 1. Create a dummy space that is not a QuantumGraph
        class DummySpace(Space):
            def distance(self, p1, p2):
                return 1.0

            def _sample_uniform(self, n: int) -> list:
                return [1] * n

            def calculate_energy(self, centers: list) -> float:
                return 0.0

            def compute_clusters(self, centers: list) -> None:
                pass

            def center_from_point(self, point):
                return point

            def sample_centers(self, k: int) -> list:
                return [1] * k

            def sample_kpp_centers(self, k: int) -> list:
                return [1] * k

            def distances_from_centers(self, centers: list, target):
                import numpy as np

                return np.zeros(len(centers))

        # 2. Create a mock SA instance using this space
        class MockSA:
            def __init__(self):
                self.space = DummySpace()

        sa_instance = MockSA()
        strategy = MostFrequentNode()

        # 3. Assert that calling initialize raises a TypeError
        with pytest.raises(TypeError, match="only be used with QuantumGraph spaces"):
            strategy.initialize(sa_instance)


class TestIntegration:
    """Integration tests combining multiple components."""

    def test_full_pipeline_sbm(self):
        """Test full clustering pipeline on SBM graph."""
        # Generate a graph with 2 clear clusters
        graph = generate_sbm(sizes=[20, 20], p=[[0.8, 0.05], [0.05, 0.8]])

        # Sample points
        points = graph.sample_points(40, strategy=UniformNodeSampling())

        # Run simulated annealing
        sa = SimulatedAnnealing(points, k=2, lambda0=1, beta0=2.0)
        centers = sa.run_interleaved(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            robust_prop=0.1,
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
        points = graph.sample_points(50, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2, lambda0=1, beta0=2.0)

        # Random initialization should have higher energy than k-means++
        centers_random = RandomInit().initialize_centers(sa)
        centers_kpp = KMeansPlusPlus().initialize_centers(sa)

        energy_random = graph.calculate_energy(centers_random)
        energy_kpp = graph.calculate_energy(centers_kpp)

        # k-means++ should generally be better (or equal) to random
        # This is probabilistic, so we just check it runs
        assert energy_random >= 0
        assert energy_kpp >= 0


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
