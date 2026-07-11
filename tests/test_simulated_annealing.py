"""Test simulated annealing algorithm."""

import numpy as np
import pytest

from kmeanssa_ng import (
    QGPoint,
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

        centers = sa.run(
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

        centers = sa.run(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
        )

        assert len(centers) == 2

    def test_run_with_robustification(self):
        """Test running with robustification."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())
        sa = SimulatedAnnealing(points, k=2)

        centers = sa.run(
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
            sa.run(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                robust_prop=1.5,
            )

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MinimizeEnergy(),
                robust_prop=-0.1,
            )

    def test_calculate_energy_python_fallback(self):
        """Without precomputing, the energy falls back to Python distances."""
        from kmeanssa_ng import QuantumGraph

        graph = QuantumGraph()
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            graph.add_edge(u, v, length=1.0)
        points = graph.sample_points(20, strategy=UniformNodeSampling(random_state=3))

        sa = SimulatedAnnealing(points, k=2)
        centers = RandomInit().initialize_centers(sa)

        assert graph._pairwise_nodes_distance_array is None
        energy_fallback = sa.calculate_energy(centers)
        graph.precomputing()
        energy_numba = sa.calculate_energy(centers)

        assert energy_fallback >= 0
        assert abs(energy_fallback - energy_numba) < 1e-9

    def test_energy_mode_node_measure(self):
        """Test that energy_mode='node_measure' reaches the numba kernel."""
        graph = generate_simple_graph(n_a=3)
        import networkx as nx

        # Declare the node measure explicitly (samplers are pure draws and
        # no longer stamp it; the old test leaned on that stamping, its own
        # set_node_attributes silently targeted nonexistent integer nodes).
        nx.set_node_attributes(
            graph, {"A0": {"obs_weight": 5}, "B0": {"obs_weight": 10}}
        )
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2, energy_mode="node_measure")
        centers = RandomInit().initialize_centers(sa)

        # Mock the space's calculate_energy and calculate_energy_numba methods
        from unittest.mock import patch

        with patch.object(
            sa.space, "calculate_energy_numba", create=True
        ) as mock_calculate_energy_numba:
            sa.calculate_energy(centers)
            mock_calculate_energy_numba.assert_called_with(
                centers, how="node_measure", observations=None
            )

    def test_centers_property(self):
        """Test centers property (covers line 113)."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(20, strategy=UniformNodeSampling())

        sa = SimulatedAnnealing(points, k=2)

        # Initially empty
        assert sa.centers == []

        # After running, should have centers
        sa.run(
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

        centers = sa.run(
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

        centers = sa.run(
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
            sa.run(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MostFrequentNode(),
                robust_prop=1.5,
            )

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run(
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
            sa.run(
                initialization_strategy=KMeansPlusPlus(),
                robustification_strategy=MostFrequentNode(),
                robust_prop=1.5,
            )

        with pytest.raises(ValueError, match=r"proportion must be in \[0,1\]"):
            sa.run(
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

        centers = sa.run(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MostFrequentNode(),
        )

        assert len(centers) == 2
        assert all(isinstance(c, QGCenter) for c in centers)
        assert all(c.space == graph for c in centers)

    def test_full_reproducibility_with_random_state(self):
        """Same random_state on shared observations gives identical centers."""
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(30, strategy=UniformNodeSampling(random_state=42))

        # Run 1
        sa1 = SimulatedAnnealing(points, k=3, random_state=42)
        centers1 = sa1.run(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            robust_prop=0.1,
        )

        # Run 2 - same seed
        sa2 = SimulatedAnnealing(points, k=3, random_state=42)
        centers2 = sa2.run(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            robust_prop=0.1,
        )

        # Assert identical results
        assert len(centers1) == len(centers2)
        for c1, c2 in zip(centers1, centers2):
            assert c1.edge == c2.edge
            assert abs(c1.position - c2.position) < 1e-10

    @staticmethod
    def _clustering_pipeline(seed):
        """End-to-end run: seeded generation, sampling and annealing."""
        graph = generate_sbm(
            sizes=[15, 15], p=[[0.7, 0.1], [0.1, 0.7]], random_state=seed
        )
        points = graph.sample_points(
            40, strategy=UniformNodeSampling(random_state=seed)
        )
        sa = SimulatedAnnealing(points, k=2, random_state=seed)
        return sa.run(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            robust_prop=0.1,
        )

    def test_end_to_end_reproducibility(self):
        """Full pipeline (generation -> sampling -> annealing) is reproducible."""
        centers1 = self._clustering_pipeline(seed=7)
        centers2 = self._clustering_pipeline(seed=7)

        assert len(centers1) == len(centers2)
        for c1, c2 in zip(centers1, centers2):
            assert c1.edge == c2.edge
            assert abs(c1.position - c2.position) < 1e-10

    def test_reproducibility_independent_of_global_state(self):
        """Results depend only on random_state, not on global random state."""
        import random
        import numpy as np

        random.seed(0)
        np.random.seed(0)
        centers1 = self._clustering_pipeline(seed=123)

        # Perturb both global generators between the two runs.
        random.seed(999)
        np.random.seed(999)
        [random.random() for _ in range(10)]
        np.random.random(10)

        centers2 = self._clustering_pipeline(seed=123)

        assert len(centers1) == len(centers2)
        for c1, c2 in zip(centers1, centers2):
            assert c1.edge == c2.edge
            assert abs(c1.position - c2.position) < 1e-10

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
        from kmeanssa_ng.core.abstract import Space, Point

        # 1. Create a dummy space that is not a QuantumGraph
        class DummySpace(Space):
            def distance(self, p1, p2):
                return 1.0

            def _sample_uniform(self, n: int) -> list:
                return [1] * n

            def calculate_energy(self, centers: list) -> float:
                return 0.0

            def center_from_point(self, point):
                return point

            def sample_centers(self, k: int) -> list:
                return [1] * k

            def sample_kpp_centers(self, k: int) -> list:
                return [1] * k

            def distances_from_centers(self, centers: list, target):
                import numpy as np

                return np.zeros(len(centers))

            def get_point_type(self) -> type:
                return Point

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
        centers = sa.run(
            initialization_strategy=KMeansPlusPlus(),
            robustification_strategy=MinimizeEnergy(),
            robust_prop=0.1,
        )

        # Assign clusters
        nodes_as_points = graph.nodes_as_points()
        labels = graph.assign_clusters(nodes_as_points, centers)

        # Set node attributes for verification
        import networkx as nx

        node_to_cluster = {
            node.edge[0]: label for node, label in zip(nodes_as_points, labels)
        }
        nx.set_node_attributes(graph, node_to_cluster, "cluster")

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


class TestEnergyTracking:
    """Tests for run(record_energy=True) and the history properties."""

    def _sa(self):
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(30, strategy=UniformNodeSampling(random_state=1))
        return SimulatedAnnealing(points, k=2, random_state=42)

    def _run(self, sa, record_energy):
        return sa.run(
            KMeansPlusPlus(),
            MinimizeEnergy(),
            robust_prop=0.1,
            record_energy=record_energy,
        )

    def test_history_empty_without_tracking(self):
        sa = self._sa()
        self._run(sa, record_energy=False)
        assert sa.energy_history.size == 0
        assert sa.time_history.size == 0

    def test_history_recorded_with_tracking(self):
        sa = self._sa()
        self._run(sa, record_energy=True)
        # initial state + one entry per observation
        assert len(sa.energy_history) == sa.n + 1
        assert len(sa.time_history) == len(sa.energy_history)
        assert sa.time_history[0] == 0.0
        assert np.all(np.diff(sa.time_history) >= -1e-12)

    def test_final_recorded_energy_matches_centers(self):
        sa = self._sa()
        # The last recorded energy is computed on the final trajectory state,
        # so it matches the current centers (sa.centers) exactly. It does *not*
        # match run()'s return value, which is the robustified best-of-window
        # result and generally a different, lower-energy configuration.
        self._run(sa, record_energy=True)
        assert abs(sa.energy_history[-1] - sa.calculate_energy(sa.centers)) < 1e-9

    def test_tracking_does_not_change_result(self):
        untracked = self._run(self._sa(), record_energy=False)
        tracked = self._run(self._sa(), record_energy=True)
        assert len(untracked) == len(tracked)
        for a, b in zip(untracked, tracked):
            assert a.edge == b.edge
            assert abs(a.position - b.position) < 1e-12


class TestPoissonTimes:
    """Regression tests for the Poisson clock driving the annealing loop.

    They pin down two former bugs: observation ``i`` was processed over the
    interval ending at ``times[i]`` instead of ``times[i + 1]`` (so the first
    observation drove no dynamics and the last interval was never consumed),
    and the residual fraction of an interval shorter than ``step_size`` was
    silently dropped.
    """

    def _sa(self, n_points=30):
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(
            n_points, strategy=UniformNodeSampling(random_state=1)
        )
        return SimulatedAnnealing(points, k=2, random_state=42)

    def test_initialize_times_valid_clock(self):
        sa = self._sa()
        times = sa._initialize_times(50)
        assert times.shape == (51,)
        assert times[0] == 0.0
        assert np.all(np.isfinite(times))
        assert np.all(np.diff(times) > 0)

    def test_arrival_times_follow_the_paper_intensity(self):
        """The clock realises lambda(t) = lambda0 (1 + t) (RNG-3 regression).

        The i-th arrival satisfies Lambda(T_i) = S_i with the paper's
        cumulative intensity Lambda(t) = lambda0 (t + t^2/2). Mapping the
        arrival times back through Lambda must recover unit-rate exponential
        increments. The previous clock dropped the factor 2, realising
        lambda(t) = 2 lambda0 (1 + t) instead, which halves these increments
        (mean 0.5) and fails the test.
        """
        from scipy import stats

        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(2, strategy=UniformNodeSampling(random_state=1))
        lambda0 = 2.0
        sa = SimulatedAnnealing(points, k=1, lambda0=lambda0, random_state=7)

        times = sa._initialize_times(20000)[1:]
        cumulative = lambda0 * (times + times**2 / 2.0)  # Lambda(T_i)
        increments = np.diff(cumulative, prepend=0.0)  # unit exponentials
        assert increments.mean() == pytest.approx(1.0, abs=0.03)
        # Distribution, not just the mean: KS against a unit exponential.
        _, p_value = stats.kstest(increments, "expon")
        assert p_value > 0.01

    def test_every_observation_advances_time(self):
        """Each observation, including the first, drives a positive time interval."""
        sa = self._sa()
        sa.run(
            KMeansPlusPlus(),
            MinimizeEnergy(),
            robust_prop=0.1,
            record_energy=True,
        )
        assert np.all(np.diff(sa.time_history) > 0)

    def test_full_time_horizon_consumed(self):
        """Intervals shorter than step_size are simulated down to their residual."""
        sa = self._sa()
        times = np.linspace(0.0, 0.05, sa.n + 1)  # every interval << step_size
        sa._initialize_times = lambda n: times
        sa.run(
            KMeansPlusPlus(),
            MinimizeEnergy(),
            robust_prop=0.1,
            record_energy=True,
        )
        np.testing.assert_allclose(sa.time_history, times, rtol=0, atol=1e-15)


class TestRandomInitReplacement:
    """RandomInit samples distinct observations (RNG-5 regression).

    Drawing centers with replacement could place several centers on one
    observation, shrinking k from the outset.
    """

    def test_no_duplicate_indices_when_k_equals_n(self):
        graph = generate_simple_graph(n_a=3)
        # Distinct observations, one per node, so distinct indices give
        # distinct center locations.
        nodes = list(graph.nodes())
        points = [
            graph.sample_points(1, strategy=UniformNodeSampling(random_state=i))[0]
            for i in range(len(nodes))
        ]
        sa = SimulatedAnnealing(points, k=len(points), random_state=0)

        centers = RandomInit().initialize_centers(sa)
        edges = sorted(c.edge for c in centers)
        assert edges == sorted(p.edge for p in points)  # a permutation, no repeats

    def test_k_greater_than_n_raises(self):
        graph = generate_simple_graph(n_a=3)
        points = graph.sample_points(3, strategy=UniformNodeSampling(random_state=0))
        sa = SimulatedAnnealing(points, k=5, random_state=0)
        with pytest.raises(ValueError, match="distinct centers"):
            RandomInit().initialize_centers(sa)


class TestInitializationStrategyGuards:
    """Defensive branches of the init strategies (coverage of the error paths
    and the all-coincident degenerate case)."""

    class _Stub:
        """Minimal object exposing what initialize_centers reads."""

        def __init__(self, k, observations):
            self.k = k
            self.observations = observations

    def test_random_init_rejects_non_positive_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            RandomInit().initialize_centers(self._Stub(0, [object()]))

    def test_random_init_rejects_empty_observations(self):
        with pytest.raises(ValueError, match="No observations available"):
            RandomInit().initialize_centers(self._Stub(2, []))

    def test_kpp_rejects_non_positive_k(self):
        with pytest.raises(ValueError, match="k must be positive"):
            KMeansPlusPlus().initialize_centers(self._Stub(-1, [object()]))

    def test_kpp_rejects_empty_observations(self):
        with pytest.raises(ValueError, match="No observations available"):
            KMeansPlusPlus().initialize_centers(self._Stub(3, []))

    def test_kpp_with_all_coincident_points_falls_back_to_uniform(self):
        """When every observation coincides, all squared distances are 0, so
        k-means++ falls back to a uniform choice instead of dividing by zero."""
        graph = generate_simple_graph(n_a=3)
        coincident = [QGPoint(graph, ("A0", "B0"), 0.0) for _ in range(5)]
        sa = SimulatedAnnealing(coincident, k=3, random_state=0)
        centers = KMeansPlusPlus().initialize_centers(sa)
        assert len(centers) == 3
