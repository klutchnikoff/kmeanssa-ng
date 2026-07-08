"""Test quantum graph functionality."""

import networkx as nx
import numpy as np
import pytest

from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng import (
    QGCenter,
    QGPoint,
    QuantumGraph,
    generate_sbm,
    generate_simple_graph,
    generate_random_sbm,
    as_quantum_graph,
    complete_quantum_graph,
)


class TestQuantumGraph:
    """Tests for QuantumGraph class."""

    def test_create_simple_graph(self):
        """Test creating a simple quantum graph."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.add_edge(0, 2, length=3.0)

        assert graph.number_of_nodes() == 3
        assert graph.number_of_edges() == 3

    def test_precomputing(self):
        """Test precomputing pairwise distances."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)

        graph.precomputing()
        assert graph._pairwise_nodes_distance is not None

    def test_node_distance(self):
        """Test distance computation between nodes."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.precomputing()

        dist = graph.node_distance(0, 2)
        assert dist == 3.0

    def test_node_distance_from_array_without_dict(self, monkeypatch):
        """node_distance must use the precomputed array even if the dict form is
        absent (e.g. a graph rebuilt from a cached distance matrix). Otherwise it
        silently falls back to a full Dijkstra on every call -- correct but ~1000x
        slower on large graphs.
        """
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.precomputing()
        # Keep the array, drop the dict form (mimics loading from a cache).
        graph._pairwise_nodes_distance = None

        # Guard the fast path: any fall-back to a live Dijkstra must blow up.
        import kmeanssa_ng.quantum_graph.space as space_mod

        def _no_dijkstra(*args, **kwargs):
            raise AssertionError("node_distance fell back to networkx Dijkstra")

        monkeypatch.setattr(space_mod.nx, "shortest_path_length", _no_dijkstra)

        assert graph._pairwise_nodes_distance_array is not None
        assert graph.node_distance(0, 2) == 3.0
        assert graph.node_distance(2, 0) == 3.0
        assert graph.node_distance(1, 1) == 0.0

    def test_get_edge_length(self):
        """Test getting edge length."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.5)

        length = graph.get_edge_length(0, 1)
        assert length == 2.5

    def test_get_edge_length_nonexistent_edge_raises(self):
        """Test that getting length of nonexistent edge raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="does not exist in graph"):
            graph.get_edge_length(2, 3)

    def test_calculate_energy_with_no_observations_raises(self):
        """how='obs' without an observation measure fails loudly.

        Regression: it used to return 0.0 for every configuration of centers,
        silently disabling any energy-based selection.
        """
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        centers = [QGCenter(QGPoint(graph, (0, 1), 0.5))]

        with pytest.raises(ValueError, match="obs_weight"):
            graph.calculate_energy(centers, how="obs")

    def test_register_observations_enables_obs_energy(self):
        """Edge-sampled points can be registered as the observation measure."""
        from kmeanssa_ng.quantum_graph.sampling import UniformEdgeSampling

        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        graph.precomputing()

        points = graph.sample_points(20, strategy=UniformEdgeSampling(random_state=0))
        centers = [QGCenter(QGPoint(graph, (0, 1), 0.5))]

        graph.register_observations(points)
        energy = graph.calculate_energy(centers, how="obs")
        assert energy > 0

        # Registration replaces the previous measure entirely
        graph.register_observations([QGPoint(graph, (0, 1), 0.0)])
        counts = [graph.nodes[n].get("obs_weight", 0) for n in graph.nodes]
        assert sum(counts) == 1

    def test_obs_energy_with_fractional_measure(self):
        """Fractional obs_weight values (a population measure) are not truncated.

        Regression: the numba obs kernel built its weights as int32, so a
        measure like the paper's population density nu (all weights < 1) was
        truncated to 0, collapsing the whole obs-energy to 0.0 -- which in turn
        made the energy-minimizing robustification a no-op (0.0 < 0.0 is never
        true), silently returning the initial centers instead of the annealed
        ones.
        """
        graph = QuantumGraph()
        for u, v in [(0, 1), (1, 2), (2, 3), (3, 0)]:
            graph.add_edge(u, v, length=1.0)
        graph.precomputing()

        # A genuine fractional measure: weights sum to 1, each strictly below 1
        weights = {0: 0.4, 1: 0.3, 2: 0.2, 3: 0.1}
        nx.set_node_attributes(graph, weights, "obs_weight")
        centers = [QGCenter(QGPoint(graph, (0, 1), 0.2))]

        numba = graph.calculate_energy(centers, how="obs")
        python = graph._calculate_energy_python(centers, "obs")
        assert numba > 0
        assert numba == pytest.approx(python, abs=1e-12)

    def test_invalid_energy_mode_raises(self):
        """A typo in the energy mode fails instead of silently meaning 'obs'."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()
        centers = [QGCenter(QGPoint(graph, (0, 1), 0.5))]

        with pytest.raises(ValueError, match="uniform"):
            graph.calculate_energy(centers, how="unifrom")

    def test_calculate_energy_numba_obs(self):
        """Test Numba-accelerated energy calculation with how='obs'."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        nx.set_node_attributes(
            graph, {0: {"obs_weight": 5}, 1: {"obs_weight": 10}, 2: {"obs_weight": 0}}
        )
        graph.precomputing()

        centers = [QGCenter(QGPoint(graph, (0, 1), 0.5))]

        # Calculate with pure Python (calculate_energy itself dispatches to
        # numba on a precomputed graph, so target the fallback directly)
        energy_python = graph._calculate_energy_python(centers, how="obs")

        # Calculate with Numba
        energy_numba = graph.calculate_energy_numba(centers, how="obs")

        assert np.isclose(energy_python, energy_numba)

    def test_batch_distances_special_cases(self):
        """Test batch_distances for same-edge and reversed-edge cases."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.precomputing()

        # Case 1: Center and target on the same edge
        c1 = QGCenter(QGPoint(graph, (0, 1), 0.2))
        p1 = QGPoint(graph, (0, 1), 0.8)
        dist1 = graph.distances_from_centers([c1], p1)
        assert np.isclose(dist1[0], 0.6)

        # Case 2: Center and target on reversed edge
        c2 = QGCenter(QGPoint(graph, (0, 1), 0.2))
        p2 = QGPoint(graph, (1, 0), 0.2)  # Same as (0, 1) at 0.8
        dist2 = graph.distances_from_centers([c2], p2)
        assert np.isclose(dist2[0], 0.6)

    def test_distances_from_centers(self):
        """Test distances_from_centers method."""
        graph = generate_simple_graph(n_a=3)
        graph.precomputing()
        points = graph.sample_points(3, strategy=UniformNodeSampling())
        centers = [graph.center_from_point(p) for p in points]
        target = graph.sample_points(1, strategy=UniformNodeSampling())[0]

        distances = graph.distances_from_centers(centers, target)

        assert isinstance(distances, np.ndarray)
        assert distances.shape == (3,)
        assert np.all(distances >= 0)

        # Manual check for one distance
        manual_dist = graph.distance(centers[0], target)
        assert np.isclose(distances[0], manual_dist)

    def test_batch_distances_without_precomputing_raises(self):
        """Test that batch_distances raises without precomputing."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        centers = [QGCenter(QGPoint(graph, (0, 1), 0.5))]
        target = QGPoint(graph, (1, 2), 0.3)

        with pytest.raises(ValueError, match="Must call precomputing"):
            graph.distances_from_centers(centers, target)


class TestQGPoint:
    """Tests for QGPoint class."""

    def test_create_point(self):
        """Test creating a point on a quantum graph."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        assert point.edge == (0, 1)
        assert point.position == 0.5
        assert point.space == graph

    def test_closest_node(self):
        """Test finding closest node to a point."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        # Point closer to node 0
        point1 = QGPoint(graph, edge=(0, 1), position=0.3)
        assert point1.closest_node() == 0

        # Point closer to node 1
        point2 = QGPoint(graph, edge=(0, 1), position=0.7)
        assert point2.closest_node() == 1

    def test_reverse(self):
        """Test reversing edge orientation."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.3)
        point.reverse()

        assert point.edge == (1, 0)
        assert abs(point.position - 0.7) < 1e-10

    def test_create_point_with_none_graph_raises(self):
        """Test that creating a point with None graph raises ValueError."""
        with pytest.raises(ValueError, match="quantum_graph cannot be None"):
            QGPoint(None, edge=(0, 1), position=0.5)

    def test_create_point_with_invalid_edge_type_raises(self):
        """Test that invalid edge type raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="must be a tuple of two nodes"):
            QGPoint(graph, edge=[0, 1], position=0.5)

    def test_create_point_with_nonexistent_edge_raises(self):
        """Test that point on non-existent edge raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="does not exist in the graph"):
            QGPoint(graph, edge=(2, 3), position=0.5)

    def test_create_point_with_negative_position_raises(self):
        """Test that negative position raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="must be non-negative"):
            QGPoint(graph, edge=(0, 1), position=-0.5)

    def test_create_point_with_position_exceeding_length_raises(self):
        """Test that position exceeding edge length raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="exceeds edge length"):
            QGPoint(graph, edge=(0, 1), position=1.5)

    def test_create_point_with_non_numeric_position_raises(self):
        """Test that non-numeric position raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        with pytest.raises(ValueError, match="must be a number"):
            QGPoint(graph, edge=(0, 1), position="invalid")

    def test_create_point_at_edge_boundaries(self):
        """Test creating points at edge boundaries (0 and edge_length)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)

        # Position at start
        point1 = QGPoint(graph, edge=(0, 1), position=0.0)
        assert point1.position == 0.0

        # Position at end
        point2 = QGPoint(graph, edge=(0, 1), position=2.0)
        assert point2.position == 2.0

    def test_set_edge_with_nonexistent_edge_raises(self):
        """Test that setting a non-existent edge raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)

        point = QGPoint(graph, edge=(0, 1), position=1.5)

        # Try to change to non-existent edge - should fail
        with pytest.raises(ValueError, match="does not belong to the graph"):
            point.edge = (5, 6)

    def test_set_edge_succeeds_and_edge_property_checks(self):
        """Test that setting edge and edge property coverage (covers 104-107, 153-154)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)
        graph.add_edge(1, 2, length=3.0)

        point = QGPoint(graph, edge=(0, 1), position=1.5)
        point.edge = (1, 2)  # Should succeed

        assert point.edge == (1, 2)
        # Position is not automatically updated when changing edge
        assert point.position == 1.5

        # Cover __str__ representation path (includes graph name if present)
        graph.name = "TestGraph"
        s = str(point)
        assert "TestGraph" in s

        # Force edge property to pass through "edge not in graph" branch with self-loop
        point.edge = (3, 3)  # Self-loop allowed by property
        assert point.edge == (3, 3)

    def test_edge_property_raises_when_not_in_graph(self):
        """Test edge property raises when edge not in graph (covers line 107)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)

        # Force the edge to be something not in the graph and not a self-loop
        point._edge = (5, 6)  # Set directly to bypass setter validation

        # Now accessing the property should raise
        with pytest.raises(ValueError, match="does not belong to the graph"):
            _ = point.edge


class TestQGCenter:
    """Tests for QGCenter class."""

    def test_create_center(self):
        """Test creating a center from a point."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        assert center.edge == (0, 1)
        assert center.position == 0.5

    def test_brownian_motion(self):
        """Test Brownian motion (just check it doesn't crash)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        center.brownian_motion(0.01)  # Small time step

        # Position should have changed (probabilistically)
        # Just check it doesn't crash for now
        assert center.edge is not None

    def test_drift(self):
        """Test drift toward target point."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)

        center = QGCenter(center_point)
        initial_pos = center.position

        # Drift halfway toward target
        center.drift(target_point, 0.5)

        # Should have moved closer to target
        assert center.position > initial_pos

    def test_brownian_motion_with_negative_time_raises(self):
        """Test that negative time_to_travel raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        with pytest.raises(ValueError, match="must be non-negative"):
            center.brownian_motion(-0.1)

    def test_brownian_motion_with_non_numeric_time_raises(self):
        """Test that non-numeric time_to_travel raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        with pytest.raises(ValueError, match="must be a number"):
            center.brownian_motion("invalid")

    def test_brownian_motion_with_zero_time_succeeds(self):
        """Test that zero time_to_travel is valid."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        center.brownian_motion(0.0)  # Should not crash

        # Position might change slightly due to random normal, but should be close
        assert center.position is not None

    def test_drift_with_none_target_raises(self):
        """Test that None target_point raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        with pytest.raises(ValueError, match="target_point cannot be None"):
            center.drift(None, 0.5)

    def test_drift_with_negative_prop_raises(self):
        """Test that negative prop_to_travel raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)
        center = QGCenter(center_point)

        with pytest.raises(ValueError, match="must be in"):
            center.drift(target_point, -0.1)

    def test_drift_with_prop_greater_than_one_raises(self):
        """Test that prop_to_travel > 1 raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)
        center = QGCenter(center_point)

        with pytest.raises(ValueError, match="must be in"):
            center.drift(target_point, 1.5)

    def test_drift_with_non_numeric_prop_raises(self):
        """Test that non-numeric prop_to_travel raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)
        center = QGCenter(center_point)

        with pytest.raises(ValueError, match="must be a number"):
            center.drift(target_point, "invalid")

    def test_drift_with_boundary_values_succeeds(self):
        """Test that prop_to_travel at boundaries (0 and 1) succeed."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        center_point = QGPoint(graph, edge=(0, 1), position=0.2)
        target_point = QGPoint(graph, edge=(0, 1), position=0.8)

        # Test with prop = 0 (no movement)
        center1 = QGCenter(center_point)
        initial_pos = center1.position
        center1.drift(target_point, 0.0)
        assert center1.position == initial_pos

        # Test with prop = 1 (full movement)
        center2 = QGCenter(center_point)
        center2.drift(target_point, 1.0)
        assert abs(center2.position - target_point.position) < 1e-10

    def test_find_best_neighbor_same_nodes(self):
        """Test _find_best_neighbor when n1 == n2 (covers line 65)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(0, 2, length=2.0)
        graph.precomputing()

        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)

        # When n1 == n2, should return n1
        result = center._find_best_neighbor(0, 0)
        assert result == 0

    def test_drift_on_same_edge_different_parametrizations(self):
        """Test drift when edges have different parametrizations (covers lines 131-137)."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.precomputing()

        # Create points on "same" edge but different parametrizations
        center_point = QGPoint(graph, edge=(0, 1), position=0.3)
        target_point = QGPoint(graph, edge=(1, 0), position=0.2)  # Reversed edge

        center = QGCenter(center_point)
        initial_pos = center.position

        # Target sits at 0.8 in the center's frame: drift must move forward
        center.drift(target_point, 0.1)
        assert center.position > initial_pos

        # Center at 0.7, target at 0.8 in the center's frame: still forward
        center_point2 = QGPoint(graph, edge=(0, 1), position=0.7)
        target_point2 = QGPoint(graph, edge=(1, 0), position=0.2)  # Reversed edge

        center2 = QGCenter(center_point2)
        initial_pos2 = center2.position

        center2.drift(target_point2, 0.1)
        assert center2.position > initial_pos2

        # Test line 131: center.position > target.position on same orientation
        center_point3 = QGPoint(graph, edge=(0, 1), position=0.8)
        target_point3 = QGPoint(graph, edge=(0, 1), position=0.2)

        center3 = QGCenter(center_point3)
        initial_pos3 = center3.position

        # This should trigger line 131
        center3.drift(target_point3, 0.1)
        assert center3.position < initial_pos3  # Should move backward


class _ForcedNormalRng:
    """Generator stand-in with a fixed normal draw and seeded vertex routing."""

    def __init__(self, normal_value: float, seed: int = 0):
        self._normal_value = normal_value
        self._rng = np.random.default_rng(seed)

    def standard_normal(self) -> float:
        return self._normal_value

    def integers(self, high: int) -> int:
        return self._rng.integers(high)


class TestQGCenterDynamicsRegression:
    """Regression tests for the elementary center moves (brownian + drift).

    They pin down two former bugs: a backward brownian step through a vertex
    left the center at a negative position, and a drift between opposite
    parametrizations of the same edge moved away from the target.
    """

    @staticmethod
    def _star_graph():
        graph = QuantumGraph()
        for node in ("b", "c", "d"):
            graph.add_edge("a", node, length=1.0)
        graph.precomputing()
        return graph

    def test_brownian_backward_through_vertex_lands_forward(self):
        """A backward step crossing a vertex continues forward on the new edge."""
        graph = self._star_graph()
        center = QGCenter(
            QGPoint(graph, edge=("a", "b"), position=0.2),
            rng=_ForcedNormalRng(-0.5),
        )

        center.brownian_motion(1.0)  # signed step of -0.5: crosses "a", 0.3 left

        assert center.edge[0] == "a"
        assert center.position == pytest.approx(0.3)

    def test_brownian_forward_through_vertex_lands_forward(self):
        """A forward step crossing a vertex continues forward on the new edge."""
        graph = self._star_graph()
        center = QGCenter(
            QGPoint(graph, edge=("b", "a"), position=0.8),
            rng=_ForcedNormalRng(0.5),
        )

        center.brownian_motion(1.0)  # signed step of +0.5: crosses "a", 0.3 left

        assert center.edge[0] == "a"
        assert center.position == pytest.approx(0.3)

    def test_drift_opposite_parametrization_reaches_target(self):
        """Full drift between opposite parametrizations lands on the target."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=10.0)
        graph.precomputing()

        center = QGCenter(QGPoint(graph, edge=(0, 1), position=1.0))
        target = QGPoint(graph, edge=(1, 0), position=2.0)  # = (0, 1) at 8.0

        center.drift(target, 1.0)

        assert center.position == pytest.approx(8.0)
        assert graph.distance(center, target) == pytest.approx(0.0)

    def test_drift_opposite_parametrization_backward_case(self):
        """Drift moves backward when the target is behind in the center's frame."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=10.0)
        graph.precomputing()

        center = QGCenter(QGPoint(graph, edge=(0, 1), position=9.0))
        target = QGPoint(graph, edge=(1, 0), position=3.0)  # = (0, 1) at 7.0

        center.drift(target, 1.0)

        assert center.position == pytest.approx(7.0)
        assert graph.distance(center, target) == pytest.approx(0.0)

    def test_drift_opposite_parametrization_partial_moves_closer(self):
        """Partial drifts shrink the distance by exactly the travelled fraction."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=10.0)
        graph.precomputing()

        target = QGPoint(graph, edge=(1, 0), position=2.0)
        for prop in (0.25, 0.5, 0.75):
            center = QGCenter(QGPoint(graph, edge=(0, 1), position=1.0))
            initial_dist = graph.distance(center, target)
            center.drift(target, prop)
            assert graph.distance(center, target) == pytest.approx(
                (1 - prop) * initial_dist
            )

    def test_position_stays_within_edge_after_many_moves(self):
        """Invariant 0 <= position <= edge length after every brownian/drift move."""
        graph = QuantumGraph()
        lengths = {(0, 1): 1.0, (1, 2): 2.0, (2, 3): 0.5, (3, 0): 1.5, (0, 2): 0.8}
        for (u, v), length in lengths.items():
            graph.add_edge(u, v, length=length)
        graph.precomputing()

        rng = np.random.default_rng(0)
        center = QGCenter(QGPoint(graph, edge=(0, 1), position=0.5), rng=rng)
        targets = [
            QGPoint(graph, edge=edge, position=pos * graph.get_edge_length(*edge))
            for edge in [(0, 1), (1, 0), (2, 3), (0, 2), (3, 0)]
            for pos in (0.0, 0.3, 0.9)
        ]

        for step in range(3000):
            center.brownian_motion(0.01)
            length = graph.get_edge_length(*center.edge)
            assert 0.0 <= center.position <= length, f"brownian step {step}"
            if step % 3 == 0:
                target = targets[rng.integers(len(targets))]
                center.drift(target, rng.uniform())
                length = graph.get_edge_length(*center.edge)
                assert 0.0 <= center.position <= length, f"drift step {step}"


class TestSelfLoopDistances:
    """Regression tests for distances and drift on self-loop edges.

    They pin down a former bug: for two points on the same loop, the
    reversed-edge shortcut ``|L - p1 - p2|`` (only valid between opposite
    parametrizations of a regular edge) matched spuriously and could return
    a distance of 0 for distinct points.
    """

    @staticmethod
    def _rose_graph():
        """Two loops at node 0 plus a pendant edge, all of unit-ish lengths."""
        graph = QuantumGraph()
        graph.add_edge(0, 0, length=1.0)
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 1, length=2.0)
        graph.precomputing()
        return graph

    def test_two_points_on_loop_direct_arc(self):
        graph = self._rose_graph()
        p1 = QGPoint(graph, edge=(0, 0), position=0.4)
        p2 = QGPoint(graph, edge=(0, 0), position=0.6)
        assert graph.distance(p1, p2) == pytest.approx(0.2)

    def test_two_points_on_loop_wrap_around_arc(self):
        graph = self._rose_graph()
        p1 = QGPoint(graph, edge=(0, 0), position=0.1)
        p2 = QGPoint(graph, edge=(0, 0), position=0.9)
        assert graph.distance(p1, p2) == pytest.approx(0.2)

    def test_loop_to_other_edge_goes_through_vertex(self):
        graph = self._rose_graph()
        on_loop = QGPoint(graph, edge=(0, 0), position=0.2)
        on_edge = QGPoint(graph, edge=(0, 1), position=0.5)
        on_other_loop = QGPoint(graph, edge=(1, 1), position=0.3)
        assert graph.distance(on_loop, on_edge) == pytest.approx(0.2 + 0.5)
        assert graph.distance(on_loop, on_other_loop) == pytest.approx(0.2 + 1.0 + 0.3)

    def test_numba_kernels_match_quantum_path_with_loops(self):
        """The Numba kernels and quantum_path implement the same metric."""
        graph = QuantumGraph()
        graph.add_edge(0, 0, length=1.5)
        graph.add_edge(0, 1, length=1.2)
        graph.add_edge(1, 2, length=0.7)
        graph.add_edge(2, 0, length=2.0)
        graph.add_edge(2, 2, length=0.9)
        graph.precomputing()

        rng = np.random.default_rng(3)
        edges = [(0, 0), (0, 1), (1, 0), (1, 2), (2, 0), (2, 2)]

        def random_point():
            edge = edges[rng.integers(len(edges))]
            position = rng.uniform(0, graph.get_edge_length(*edge))
            return QGPoint(graph, edge=edge, position=position)

        for _ in range(50):
            center = QGCenter(random_point())
            target = random_point()
            batch = graph.distances_from_centers([center], target)[0]
            reference = graph.distance(center, target)
            assert batch == pytest.approx(reference, abs=1e-12)
            # The metric is symmetric
            assert graph.distance(target, center) == pytest.approx(reference, abs=1e-12)

    def test_energy_numba_matches_python_with_loops(self):
        graph = QuantumGraph()
        graph.add_edge(0, 0, length=1.5)
        graph.add_edge(0, 1, length=1.2)
        graph.add_edge(1, 2, length=0.7)
        graph.add_edge(2, 2, length=0.9)
        graph.precomputing()
        for i, node in enumerate(graph.nodes()):
            graph.nodes[node]["obs_weight"] = i + 1

        centers = [
            QGCenter(QGPoint(graph, edge=(0, 0), position=1.1)),
            QGCenter(QGPoint(graph, edge=(1, 2), position=0.3)),
        ]
        for how in ("uniform", "obs"):
            assert graph.calculate_energy_numba(centers, how=how) == pytest.approx(
                graph._calculate_energy_python(centers, how), abs=1e-12
            )

    def test_drift_on_loop_direct_arc(self):
        graph = self._rose_graph()
        center = QGCenter(QGPoint(graph, edge=(0, 0), position=0.2))
        target = QGPoint(graph, edge=(0, 0), position=0.5)

        center.drift(target, 1.0)

        assert center.edge == (0, 0)
        assert center.position == pytest.approx(0.5)

    def test_drift_on_loop_wraps_through_vertex(self):
        graph = self._rose_graph()
        center = QGCenter(QGPoint(graph, edge=(0, 0), position=0.1))
        target = QGPoint(graph, edge=(0, 0), position=0.9)

        center.drift(target, 1.0)
        assert graph.distance(center, target) == pytest.approx(0.0)

        # Partial drift moves backward through the vertex, never off-edge
        center2 = QGCenter(QGPoint(graph, edge=(0, 0), position=0.1))
        center2.drift(target, 0.5)
        assert graph.distance(center2, target) == pytest.approx(0.1)
        assert 0.0 <= center2.position <= 1.0

    def test_drift_from_loop_to_edge(self):
        graph = self._rose_graph()
        target = QGPoint(graph, edge=(0, 1), position=0.5)
        # From both sides of the loop: the geodesic exits via the shorter arc
        for start in (0.2, 0.8):
            center = QGCenter(QGPoint(graph, edge=(0, 0), position=start))
            assert graph.distance(center, target) == pytest.approx(0.7)
            center.drift(target, 1.0)
            assert graph.distance(center, target) == pytest.approx(0.0)

    def test_drift_from_edge_to_loop(self):
        graph = self._rose_graph()
        center = QGCenter(QGPoint(graph, edge=(1, 0), position=0.5))
        target = QGPoint(graph, edge=(0, 0), position=0.9)

        assert graph.distance(center, target) == pytest.approx(0.5 + 0.1)
        center.drift(target, 1.0)
        assert graph.distance(center, target) == pytest.approx(0.0)

    def test_position_invariant_with_loops(self):
        """0 <= position <= edge length holds through brownian and drift moves."""
        graph = self._rose_graph()
        rng = np.random.default_rng(1)
        center = QGCenter(QGPoint(graph, edge=(0, 1), position=0.5), rng=rng)
        targets = [
            QGPoint(graph, edge=edge, position=pos * graph.get_edge_length(*edge))
            for edge in [(0, 0), (0, 1), (1, 1)]
            for pos in (0.05, 0.5, 0.95)
        ]

        for step in range(2000):
            center.brownian_motion(0.01)
            length = graph.get_edge_length(*center.edge)
            assert 0.0 <= center.position <= length, f"brownian step {step}"
            if step % 3 == 0:
                target = targets[rng.integers(len(targets))]
                center.drift(target, rng.uniform())
                length = graph.get_edge_length(*center.edge)
                assert 0.0 <= center.position <= length, f"drift step {step}"


class TestDistanceCacheInvalidation:
    """Editing the graph must never leave stale precomputed distances."""

    @staticmethod
    def _path_graph():
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        graph.precomputing()
        return graph

    def test_add_edge_invalidates_and_recompute_sees_it(self):
        graph = self._path_graph()
        assert graph.node_distance(0, 2) == 2.0

        graph.add_edge(0, 2, length=0.5)  # shortcut
        assert graph._pairwise_nodes_distance_array is None
        graph.precomputing()
        assert graph.node_distance(0, 2) == 0.5

    def test_stale_fast_path_refuses_after_edit(self):
        graph = self._path_graph()
        center = QGCenter(QGPoint(graph, (0, 1), 0.5))
        target = QGPoint(graph, (1, 2), 0.5)
        graph.distances_from_centers([center], target)  # works when precomputed

        graph.add_edge(0, 2, length=0.5)
        with pytest.raises(ValueError, match="precomputing"):
            graph.distances_from_centers([center], target)

    def test_new_node_reachable_after_recompute(self):
        graph = self._path_graph()
        graph.add_edge(2, 3, length=1.0)
        graph.precomputing()
        assert graph.node_distance(0, 3) == 3.0

    def test_diameter_refreshes_after_edit(self):
        graph = self._path_graph()
        assert graph.diameter == 2.0
        graph.add_edge(0, 2, length=0.5)
        graph.precomputing()
        assert graph.diameter == 1.0  # the 0-2 shortcut shrinks the diameter


class TestGenerators:
    """Tests for graph generators."""

    def test_generate_simple_graph(self):
        """Test simple graph generator."""
        graph = generate_simple_graph(n_a=3, n_aa=2, bridge_length=2.0)

        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0
        assert graph._pairwise_nodes_distance is not None  # Should be precomputed

    def test_generate_sbm(self):
        """Test SBM generator."""
        graph = generate_sbm(sizes=[10, 10], p=[[0.7, 0.1], [0.1, 0.7]])

        assert graph.number_of_nodes() == 20
        assert graph._pairwise_nodes_distance is not None

    def test_sample_points(self):
        """Test sampling points from a graph."""
        graph = generate_simple_graph(n_a=3)

        points = graph.sample_points(10, strategy=UniformNodeSampling())
        assert len(points) == 10
        assert all(isinstance(p, QGPoint) for p in points)

    def test_sample_centers(self):
        """Test sampling centers from a graph."""
        graph = generate_simple_graph(n_a=3)

        points = graph.sample_points(3, strategy=UniformNodeSampling())
        centers = [graph.center_from_point(p) for p in points]
        assert len(centers) == 3
        assert all(isinstance(c, QGCenter) for c in centers)

    @pytest.mark.parametrize(
        "invalid_graph, expected_error_match",
        [
            (None, "`graph` must be a networkx.Graph object."),
            ("not a graph", "`graph` must be a networkx.Graph object."),
            (123, "`graph` must be a networkx.Graph object."),
        ],
    )
    def test_as_quantum_graph_invalid_graph_raises_value_error(
        self, invalid_graph, expected_error_match
    ):
        """Test that invalid 'graph' raises ValueError."""
        with pytest.raises(ValueError, match=expected_error_match):
            as_quantum_graph(invalid_graph)

    @pytest.mark.parametrize(
        "invalid_weight, expected_error_match",
        [
            (0, "`node_weight` must be a positive number."),
            (-1, "`node_weight` must be a positive number."),
            ("invalid", "`node_weight` must be a positive number."),
        ],
    )
    def test_as_quantum_graph_invalid_node_weight_raises_value_error(
        self, invalid_weight, expected_error_match
    ):
        """Test that invalid 'node_weight' raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(ValueError, match=expected_error_match):
            as_quantum_graph(G, node_weight=invalid_weight)

    @pytest.mark.parametrize(
        "invalid_length, expected_error_match",
        [
            (0, "`edge_length` must be a positive number."),
            (-1, "`edge_length` must be a positive number."),
            ("invalid", "`edge_length` must be a positive number."),
        ],
    )
    def test_as_quantum_graph_invalid_edge_length_raises_value_error(
        self, invalid_length, expected_error_match
    ):
        """Test that invalid 'edge_length' raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(ValueError, match=expected_error_match):
            as_quantum_graph(G, edge_length=invalid_length)

    @pytest.mark.parametrize(
        "invalid_weight, expected_error_match",
        [
            (0, "`edge_weight` must be a positive number."),
            (-1, "`edge_weight` must be a positive number."),
            ("invalid", "`edge_weight` must be a positive number."),
        ],
    )
    def test_as_quantum_graph_invalid_edge_weight_raises_value_error(
        self, invalid_weight, expected_error_match
    ):
        """Test that invalid 'edge_weight' raises ValueError."""
        G = nx.Graph()
        G.add_edge(0, 1)
        with pytest.raises(ValueError, match=expected_error_match):
            as_quantum_graph(G, edge_weight=invalid_weight)

    @pytest.mark.parametrize(
        "invalid_objects, expected_error_match",
        [
            ([], "`objects` must be a non-empty list."),
            ("not a list", "`objects` must be a non-empty list."),
            (123, "`objects` must be a non-empty list."),
        ],
    )
    def test_complete_quantum_graph_invalid_objects_raises_value_error(
        self, invalid_objects, expected_error_match
    ):
        """Test that invalid 'objects' raise ValueError."""
        with pytest.raises(ValueError, match=expected_error_match):
            complete_quantum_graph(invalid_objects)

    @pytest.mark.parametrize(
        "invalid_similarities, expected_error_match",
        [
            ("not an array", "`similarities` must be a numpy array."),
            (
                np.array([[1, 2]]),
                "`similarities` must be a square matrix of size 2x2.",
            ),  # Not square
            (
                np.array([[-1, 2], [3, 4]]),
                "Elements of `similarities` must be non-negative.",
            ),  # Negative value
        ],
    )
    def test_complete_quantum_graph_invalid_similarities_raises_value_error(
        self, invalid_similarities, expected_error_match
    ):
        """Test that invalid 'similarities' raise ValueError."""
        objects = [1, 2]
        with pytest.raises(ValueError, match=expected_error_match):
            complete_quantum_graph(objects, similarities=invalid_similarities)

    @pytest.mark.parametrize(
        "invalid_labels, expected_error_match",
        [
            ("not a list", "`true_labels` must be a list."),
            (
                [1],
                r"`true_labels` must have the same length as `objects` \(\d+\).",
            ),  # Incorrect length
        ],
    )
    def test_complete_quantum_graph_invalid_true_labels_raises_value_error(
        self, invalid_labels, expected_error_match
    ):
        """Test that invalid 'true_labels' raise ValueError."""
        objects = [1, 2]
        with pytest.raises(ValueError, match=expected_error_match):
            complete_quantum_graph(objects, true_labels=invalid_labels)


class TestGenerateRandomSBM:
    """Tests for generate_random_sbm input validation."""

    def test_generate_random_sbm_default_params(self):
        """Test generate_random_sbm with default parameters."""
        graph = generate_random_sbm()
        assert graph.number_of_nodes() == 100  # 50 + 50
        assert graph.number_of_edges() > 0

    @pytest.mark.parametrize(
        "invalid_sizes, expected_error_match",
        [
            (None, None),  # Default behavior, no error
            ([], "`sizes` must be a non-empty list of positive integers."),
            ([0], "`sizes` must be a non-empty list of positive integers."),
            ([-10], "`sizes` must be a non-empty list of positive integers."),
            ([10, -5], "`sizes` must be a non-empty list of positive integers."),
            ([10.5, 20], "`sizes` must be a non-empty list of positive integers."),
            ("not a list", "`sizes` must be a non-empty list of positive integers."),
            ([10, "invalid"], "`sizes` must be a non-empty list of positive integers."),
        ],
    )
    def test_generate_random_sbm_invalid_sizes_raises_value_error(
        self, invalid_sizes, expected_error_match
    ):
        """Test that invalid 'sizes' raise ValueError."""
        if expected_error_match is None:
            graph = generate_random_sbm(sizes=invalid_sizes)
            assert graph.number_of_nodes() == 100
        else:
            with pytest.raises(ValueError, match=expected_error_match):
                # For empty sizes, provide empty p, weights, lengths to avoid early validation errors
                if invalid_sizes == []:
                    generate_random_sbm(
                        sizes=invalid_sizes, p=[], weights=[], lengths=[]
                    )
                else:
                    generate_random_sbm(sizes=invalid_sizes)

    @pytest.mark.parametrize(
        "invalid_p, expected_error_match",
        [
            (None, None),  # Default behavior, no error
            ([], r"`p` must be a square matrix of size \d+x\d+."),
            ([[0.5]], r"`p` must be a square matrix of size \d+x\d+."),  # Not square
            (
                [[0.5, 0.5]],
                r"`p` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[0.5, 0.5], [0.5]],
                r"`p` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[0.5, 0.5], [0.5, 1.5]],
                "Elements of `p` must be floats or integers between 0 and 1.",
            ),  # Value > 1
            (
                [[-0.1, 0.5], [0.5, 0.5]],
                "Elements of `p` must be floats or integers between 0 and 1.",
            ),  # Value < 0
            (
                [[0.5, "invalid"], [0.5, 0.5]],
                "Elements of `p` must be floats or integers between 0 and 1.",
            ),  # Non-numeric
            ("not a list", r"`p` must be a square matrix of size \d+x\d+."),
        ],
    )
    def test_generate_random_sbm_invalid_p_raises_value_error(
        self, invalid_p, expected_error_match
    ):
        """Test that invalid 'p' raises ValueError."""
        if expected_error_match is None:
            graph = generate_random_sbm(p=invalid_p)
            assert graph.number_of_nodes() == 100
        else:
            with pytest.raises(ValueError, match=expected_error_match):
                generate_random_sbm(sizes=[50, 50], p=invalid_p)

    @pytest.mark.parametrize(
        "invalid_weights, expected_error_match",
        [
            (None, None),  # Default behavior, no error
            ([], r"`weights` must be a list of size \d+."),
            ([0], r"`weights` must be a list of size \d+."),
            ([-1], r"`weights` must be a list of size \d+."),
            ([1, -0.5], "Elements of `weights` must be positive numbers."),
            ([1, "invalid"], "Elements of `weights` must be positive numbers."),
            ("not a list", r"`weights` must be a list of size \d+."),
        ],
    )
    def test_generate_random_sbm_invalid_weights_raises_value_error(
        self, invalid_weights, expected_error_match
    ):
        """Test that invalid 'weights' raise ValueError."""
        if expected_error_match is None:
            graph = generate_random_sbm(weights=invalid_weights)
            assert graph.number_of_nodes() == 100
        else:
            with pytest.raises(ValueError, match=expected_error_match):
                generate_random_sbm(sizes=[50, 50], weights=invalid_weights)

    @pytest.mark.parametrize(
        "invalid_lengths, expected_error_match",
        [
            (None, None),  # Default behavior, no error
            ([], r"`lengths` must be a square matrix of size \d+x\d+."),
            (
                [[1]],
                r"`lengths` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[1, 2]],
                r"`lengths` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[1, 2], [3]],
                r"`lengths` must be a square matrix of size \d+x\d+.",
            ),  # Not square
            (
                [[1, -2], [3, 4]],
                "Elements of `lengths` must be positive numbers.",
            ),  # Negative value
            (
                [[1, "invalid"], [3, 4]],
                "Elements of `lengths` must be positive numbers.",
            ),  # Non-numeric
            ("not a list", r"`lengths` must be a square matrix of size \d+x\d+."),
        ],
    )
    def test_generate_random_sbm_invalid_lengths_raises_value_error(
        self, invalid_lengths, expected_error_match
    ):
        """Test that invalid 'lengths' raise ValueError."""
        if expected_error_match is None:
            graph = generate_random_sbm(lengths=invalid_lengths)
            assert graph.number_of_nodes() == 100
        else:
            with pytest.raises(ValueError, match=expected_error_match):
                generate_random_sbm(sizes=[50, 50], lengths=invalid_lengths)


class TestDistance:
    """Tests for distance computation."""

    def test_distance_same_edge(self):
        """Test distance between points on the same edge."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)

        p1 = QGPoint(graph, edge=(0, 1), position=0.2)
        p2 = QGPoint(graph, edge=(0, 1), position=0.8)

        dist = graph.distance(p1, p2)
        assert abs(dist - 0.6) < 1e-10

    def test_distance_different_edges(self):
        """Test distance between points on different edges."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)
        graph.precomputing()

        p1 = QGPoint(graph, edge=(0, 1), position=0.5)
        p2 = QGPoint(graph, edge=(1, 2), position=0.5)

        dist = graph.distance(p1, p2)
        # Should be (0.5 from p1 to node 1) + (0.5 from node 1 to p2) = 1.0
        assert abs(dist - 1.0) < 1e-10


class TestValidation:
    """Tests for input validation in QuantumGraph."""

    def test_add_edge_without_length_raises(self):
        """Test that adding an edge without length raises ValueError."""
        graph = QuantumGraph()
        with pytest.raises(ValueError, match="must have a 'length' attribute"):
            graph.add_edge(0, 1)

    def test_add_edge_with_zero_length_raises(self):
        """Test that zero length raises ValueError."""
        graph = QuantumGraph()
        with pytest.raises(ValueError, match="must be positive"):
            graph.add_edge(0, 1, length=0)

    def test_add_edge_with_negative_length_raises(self):
        """Test that negative length raises ValueError."""
        graph = QuantumGraph()
        with pytest.raises(ValueError, match="must be positive"):
            graph.add_edge(0, 1, length=-1.5)

    def test_add_edge_with_non_numeric_length_raises(self):
        """Test that non-numeric length raises ValueError."""
        graph = QuantumGraph()
        with pytest.raises(ValueError, match="must be a number"):
            graph.add_edge(0, 1, length="invalid")

    def test_validate_edge_lengths_with_missing_length(self):
        """Test validation fails when edge is missing length attribute."""
        graph = QuantumGraph()
        # Bypass validation by using parent class method
        nx.Graph.add_edge(graph, 0, 1, weight=1.0)  # No length attribute

        with pytest.raises(ValueError, match="missing 'length' attribute"):
            graph.validate_edge_lengths()

    def test_validate_edge_lengths_with_invalid_length(self):
        """Test validation fails when edge has invalid length."""
        graph = QuantumGraph()
        # Bypass validation by using parent class method
        nx.Graph.add_edge(graph, 0, 1, length=-2.0)

        with pytest.raises(ValueError, match="invalid length.*must be positive"):
            graph.validate_edge_lengths()

    def test_validate_edge_lengths_success(self):
        """Test validation succeeds with valid edges."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)

        # Should not raise
        graph.validate_edge_lengths()

    def test_precomputing_disconnected_graph_raises(self):
        """Test that precomputing on disconnected graph raises ValueError."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(2, 3, length=1.0)  # Separate component

        with pytest.raises(
            ValueError, match="must be connected.*2 connected components"
        ):
            graph.precomputing()

    def test_precomputing_invalid_edges_raises(self):
        """Test that precomputing validates edge lengths."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        # Manually corrupt an edge length
        graph[0][1]["length"] = -1.0

        with pytest.raises(ValueError, match="invalid length"):
            graph.precomputing()

    def test_precomputing_connected_graph_success(self):
        """Test that precomputing succeeds on valid connected graph."""
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.add_edge(0, 2, length=3.0)

        # Should not raise
        graph.precomputing()
        assert graph._pairwise_nodes_distance is not None


class TestNodeCenterAPI:
    """Tests for node_center_distances / node_labels / node_energy."""

    def _graph_and_centers(self):
        qg = generate_sbm(sizes=[20, 20], p=[[0.7, 0.1], [0.1, 0.7]], random_state=0)
        nodes = list(qg.nodes())
        centers = [qg.node_as_center(nodes[i]) for i in (0, 20, 10)]
        return qg, nodes, centers

    def test_distances_shape_and_nonnegative(self):
        qg, nodes, centers = self._graph_and_centers()
        d = qg.node_center_distances(centers)
        assert d.shape == (len(nodes), len(centers))
        assert np.all(d >= 0)

    def test_argmin_distances_match_compute_labels(self):
        from kmeanssa_ng.core.metrics import compute_labels

        qg, nodes, centers = self._graph_and_centers()
        points = [QGPoint(qg, (n, next(iter(qg.neighbors(n)))), 0) for n in nodes]
        expected = np.asarray(compute_labels(qg, points, centers))
        labels = np.argmin(qg.node_center_distances(centers), axis=1)
        np.testing.assert_array_equal(labels, expected)

    def test_energy_default_weights(self):
        qg, nodes, centers = self._graph_and_centers()
        d = qg.node_center_distances(centers)
        # SBM nodes carry weight 1, so the default equals the plain sum of min^2.
        assert qg.node_energy(centers) == pytest.approx(
            float((d.min(axis=1) ** 2).sum())
        )

    def test_energy_with_explicit_weights(self):
        qg, nodes, centers = self._graph_and_centers()
        d = qg.node_center_distances(centers)
        nu = np.ones(len(nodes)) / len(nodes)
        assert qg.node_energy(centers, weights=nu) == pytest.approx(
            float((nu * d.min(axis=1) ** 2).sum())
        )

    def test_requires_precomputing(self):
        qg = generate_sbm(sizes=[5, 5], precompute=False)
        _, _, centers = self._graph_and_centers()
        with pytest.raises(ValueError, match="precomputing"):
            qg.node_center_distances(centers)
