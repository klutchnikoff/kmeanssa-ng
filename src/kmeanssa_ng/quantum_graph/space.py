"""Quantum graph implementation as a metric space."""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx
import numpy as np
from numba import njit

from ..core import Space
from .center import QGCenter
from .point import QGPoint

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


@njit(cache=True, fastmath=True, inline="always")
def _pair_distance_numba(
    c_edge_0: int,
    c_edge_1: int,
    c_pos: float,
    c_length: float,
    p_edge_0: int,
    p_edge_1: int,
    p_pos: float,
    p_length: float,
    node_dist_matrix: np.ndarray,
) -> float:
    """Geodesic distance between two on-edge points of a quantum graph.

    This is the single scalar distance kernel shared by every Numba entry
    point: a geodesic either goes through one endpoint of each edge (4
    candidate paths) or stays within the common physical edge.

    Args:
        c_edge_0, c_edge_1: Node indices of the first point's edge.
        c_pos, c_length: Its position along the edge, and the edge length.
        p_edge_0, p_edge_1: Node indices of the second point's edge.
        p_pos, p_length: Its position along the edge, and the edge length.
        node_dist_matrix: Precomputed node distances (n, n).

    Returns:
        The geodesic distance.
    """
    # 4 candidate paths through the edge endpoints
    d0 = node_dist_matrix[c_edge_0, p_edge_0] + c_pos + p_pos
    d1 = node_dist_matrix[c_edge_0, p_edge_1] + c_pos + (p_length - p_pos)
    d2 = node_dist_matrix[c_edge_1, p_edge_0] + (c_length - c_pos) + p_pos
    d3 = node_dist_matrix[c_edge_1, p_edge_1] + (c_length - c_pos) + (p_length - p_pos)
    d_min = min(d0, d1, d2, d3)

    # Same physical edge cases
    if c_edge_0 == c_edge_1:
        # Self-loop: both points on the same loop reach each other along the
        # direct arc or the complementary one (the 4-path candidates already
        # cover every route through the vertex).
        if p_edge_0 == c_edge_0 and p_edge_1 == c_edge_0:
            arc = abs(c_pos - p_pos)
            d_direct = min(arc, c_length - arc)
            if d_direct < d_min:
                d_min = d_direct
    else:
        if c_edge_0 == p_edge_1 and c_edge_1 == p_edge_0:
            d_same_rev = abs(c_length - c_pos - p_pos)
            if d_same_rev < d_min:
                d_min = d_same_rev

        if c_edge_0 == p_edge_0 and c_edge_1 == p_edge_1:
            d_same = abs(c_pos - p_pos)
            if d_same < d_min:
                d_min = d_same

    return d_min


@njit(cache=True, fastmath=True)
def _batch_distances_numba(
    center_edges_0: np.ndarray,
    center_edges_1: np.ndarray,
    center_positions: np.ndarray,
    center_lengths: np.ndarray,
    target_edge_0: int,
    target_edge_1: int,
    target_pos: float,
    target_length: float,
    node_dist_matrix: np.ndarray,
) -> np.ndarray:
    """Numba-accelerated batch distance computation.

    Computes distances from k centers to one target point.

    Args:
        center_edges_0: First node of each center's edge (k,)
        center_edges_1: Second node of each center's edge (k,)
        center_positions: Position of each center (k,)
        center_lengths: Length of each center's edge (k,)
        target_edge_0: First node of target's edge
        target_edge_1: Second node of target's edge
        target_pos: Position of target
        target_length: Length of target's edge
        node_dist_matrix: Precomputed node distances (n, n)

    Returns:
        Array of distances from each center to target (k,)
    """
    k = len(center_positions)
    distances = np.empty(k, dtype=np.float64)

    for i in range(k):
        distances[i] = _pair_distance_numba(
            center_edges_0[i],
            center_edges_1[i],
            center_positions[i],
            center_lengths[i],
            target_edge_0,
            target_edge_1,
            target_pos,
            target_length,
            node_dist_matrix,
        )

    return distances


@njit(cache=True, fastmath=True)
def _calculate_energy_uniform_numba(
    center_edges_0: np.ndarray,
    center_edges_1: np.ndarray,
    center_positions: np.ndarray,
    center_lengths: np.ndarray,
    point_edges_0: np.ndarray,
    point_edges_1: np.ndarray,
    point_positions: np.ndarray,
    point_lengths: np.ndarray,
    node_dist_matrix: np.ndarray,
) -> float:
    """Numba-accelerated k-means energy calculation.

    Computes the average squared distance from each point to its nearest center.

    Args:
        center_edges_0: First node of each center's edge (k,)
        center_edges_1: Second node of each center's edge (k,)
        center_positions: Position of each center (k,)
        center_lengths: Length of each center's edge (k,)
        point_edges_0: First node of each point's edge (n,)
        point_edges_1: Second node of each point's edge (n,)
        point_positions: Position of each point (n,)
        point_lengths: Length of each point's edge (n,)
        node_dist_matrix: Precomputed node distances (m, m)

    Returns:
        Average squared distance (k-means energy)
    """
    n_points = len(point_positions)
    k = len(center_positions)
    total_energy = 0.0

    for p_idx in range(n_points):
        p_edge_0 = point_edges_0[p_idx]
        p_edge_1 = point_edges_1[p_idx]
        p_pos = point_positions[p_idx]
        p_length = point_lengths[p_idx]

        # Find minimum distance to any center
        min_dist_sq = np.inf

        for c_idx in range(k):
            d_min = _pair_distance_numba(
                center_edges_0[c_idx],
                center_edges_1[c_idx],
                center_positions[c_idx],
                center_lengths[c_idx],
                p_edge_0,
                p_edge_1,
                p_pos,
                p_length,
                node_dist_matrix,
            )
            if d_min * d_min < min_dist_sq:
                min_dist_sq = d_min * d_min

        total_energy += min_dist_sq

    return total_energy / n_points


@njit(cache=True, fastmath=True)
def _calculate_energy_obs_numba(
    center_edges_0: np.ndarray,
    center_edges_1: np.ndarray,
    center_positions: np.ndarray,
    center_lengths: np.ndarray,
    point_edges_0: np.ndarray,
    point_edges_1: np.ndarray,
    point_positions: np.ndarray,
    point_lengths: np.ndarray,
    point_obs_weight: np.ndarray,
    node_dist_matrix: np.ndarray,
) -> float:
    n_points = len(point_positions)
    k = len(center_positions)
    total_energy = 0.0
    total_weight = 0.0

    for p_idx in range(n_points):
        obs_weight = point_obs_weight[p_idx]
        if obs_weight == 0:
            continue

        p_edge_0 = point_edges_0[p_idx]
        p_edge_1 = point_edges_1[p_idx]
        p_pos = point_positions[p_idx]
        p_length = point_lengths[p_idx]

        min_dist_sq = np.inf

        for c_idx in range(k):
            d_min = _pair_distance_numba(
                center_edges_0[c_idx],
                center_edges_1[c_idx],
                center_positions[c_idx],
                center_lengths[c_idx],
                p_edge_0,
                p_edge_1,
                p_pos,
                p_length,
                node_dist_matrix,
            )
            if d_min * d_min < min_dist_sq:
                min_dist_sq = d_min * d_min

        total_energy += min_dist_sq * obs_weight
        total_weight += obs_weight

    return total_energy / total_weight if total_weight > 0 else 0.0


class QuantumGraph(nx.Graph, Space):
    """A quantum graph is a metric graph where points can lie on edges.

    This class extends NetworkX Graph to provide:
    - Distance computation between points on edges
    - Sampling of random points and centers
    - k-means clustering support

    Each edge should have a 'length' attribute representing its metric length.
    Nodes and edges can have 'weight' and 'distribution' attributes for sampling.

    Attributes:
        diameter: The diameter of the graph (max distance between nodes).
        node_position: Layout positions for visualization.

    Example:
        ```python
        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=2.0)
        graph.precomputing()  # Cache pairwise distances

        points = graph.sample_points(100)
        centers = graph.sample_centers(5)
        ```
    """

    def __init__(
        self, incoming_graph_data=None, precompute: bool = False, **attr
    ) -> None:
        """Initialize a quantum graph.

        Args:
            incoming_graph_data: Input graph data (see networkx.Graph).
            precompute: If True, automatically precompute distances after initialization.
            **attr: Additional graph attributes.
        """
        super().__init__(incoming_graph_data, **attr)
        self._pairwise_nodes_distance: dict[int, dict[int, float]] | None = None
        self._pairwise_nodes_distance_array: np.ndarray | None = None
        self._node_to_index: dict[int, int] | None = None
        self._diameter: float = 0.0
        self._node_position: dict | None = None

        if precompute and self.number_of_nodes() > 0:
            self.precomputing()

    def _invalidate_distance_cache(self) -> None:
        """Drop every cache derived from the graph structure.

        Called by the graph mutators so that stale distances are never served
        after an edit: distance queries then fail loudly (or recompute) until
        ``precomputing()`` is run again.
        """
        self._pairwise_nodes_distance = None
        self._pairwise_nodes_distance_array = None
        self._node_to_index = None
        self._diameter = 0.0
        self._node_position = None

    def add_edge(self, u_for_edge, v_for_edge, **attr) -> None:
        """Add an edge with validation of the length attribute.

        Invalidates the precomputed distance cache: call ``precomputing()``
        again after editing the graph.

        Args:
            u_for_edge: First node.
            v_for_edge: Second node.
            **attr: Edge attributes. Must include 'length' with a positive value.

        Raises:
            ValueError: If 'length' is missing, not positive, or not a number.
        """
        if "length" not in attr:
            raise ValueError(
                f"Edge ({u_for_edge}, {v_for_edge}) must have a 'length' attribute"
            )

        length = attr["length"]

        # Check if length is a number
        try:
            length_float = float(length)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Edge ({u_for_edge}, {v_for_edge}) length must be a number, got {type(length).__name__}"
            ) from e

        # Check if length is positive
        if length_float <= 0:
            raise ValueError(
                f"Edge ({u_for_edge}, {v_for_edge}) length must be positive, got {length_float}"
            )

        super().add_edge(u_for_edge, v_for_edge, **attr)
        self._invalidate_distance_cache()

    def add_node(self, node_for_adding, **attr) -> None:
        """Add a node; invalidates the precomputed distance cache."""
        super().add_node(node_for_adding, **attr)
        self._invalidate_distance_cache()

    def remove_edge(self, u, v) -> None:
        """Remove an edge; invalidates the precomputed distance cache."""
        super().remove_edge(u, v)
        self._invalidate_distance_cache()

    def remove_node(self, n) -> None:
        """Remove a node; invalidates the precomputed distance cache."""
        super().remove_node(n)
        self._invalidate_distance_cache()

    @property
    def diameter(self) -> float:
        """Compute and cache the graph diameter.

        Returns:
            Maximum distance between any two nodes.
        """
        if self._diameter == 0.0:
            for n1 in self.nodes:
                for n2 in self.nodes:
                    d = self.node_distance(n1, n2)
                    if d > self._diameter:
                        self._diameter = d
        return self._diameter

    def validate_edge_lengths(self) -> None:
        """Validate that all edges have positive length attributes.

        Raises:
            ValueError: If any edge is missing 'length' or has invalid length.
        """
        for u, v, data in self.edges(data=True):
            if "length" not in data:
                raise ValueError(f"Edge ({u}, {v}) missing 'length' attribute")

            length = data["length"]
            try:
                length_float = float(length)
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"Edge ({u}, {v}) length must be a number, got {type(length).__name__}"
                ) from e

            if length_float <= 0:
                raise ValueError(
                    f"Edge ({u}, {v}) has invalid length {length_float}, must be positive"
                )

    def precomputing(self) -> None:
        """Precompute and cache all pairwise node distances.

        This significantly speeds up distance queries.
        Should be called once after graph construction.

        Raises:
            ValueError: If graph is not connected or has invalid edge lengths.
        """
        # Validate edge lengths first
        self.validate_edge_lengths()

        # Check connectivity
        if self.number_of_nodes() > 0 and not nx.is_connected(self):
            num_components = nx.number_connected_components(self)
            raise ValueError(
                f"Graph must be connected for distance precomputing. "
                f"Found {num_components} connected components."
            )

        if self._pairwise_nodes_distance is None:
            self._pairwise_nodes_distance = dict(
                nx.all_pairs_dijkstra_path_length(self, weight="length")
            )

            # Create numpy array version for Numba-accelerated computations
            node_list = list(self.nodes())
            n = len(node_list)
            self._node_to_index = {node: i for i, node in enumerate(node_list)}
            self._pairwise_nodes_distance_array = np.zeros((n, n), dtype=np.float64)

            for i, node_i in enumerate(node_list):
                for j, node_j in enumerate(node_list):
                    self._pairwise_nodes_distance_array[i, j] = (
                        self._pairwise_nodes_distance[node_i][node_j]
                    )

    @property
    def node_position(self) -> dict:
        """Compute layout positions for visualization.

        Returns:
            Dictionary mapping nodes to (x, y) positions.
        """
        if self._node_position is None:
            drawing = nx.get_edge_attributes(self, "length")
            drawing_weights = {
                (i, j): {"drawing": 1 / drawing[(i, j)]} for i, j in drawing
            }
            nx.set_edge_attributes(self, drawing_weights)
            # Fixed seed: the layout is the force-directed embedding of the
            # graph, only used for drawing, but leaving it unseeded made every
            # figure non-reproducible, breaking the package's determinism
            # guarantee.
            self._node_position = nx.layout.spring_layout(
                self, weight="drawing", seed=0
            )
        return self._node_position

    def node_distance(self, n1: int, n2: int) -> float:
        """Compute shortest path distance between two nodes.

        Args:
            n1: First node.
            n2: Second node.

        Returns:
            Shortest path length using 'length' edge attribute.
        """
        # Prefer the precomputed array (O(1)); it may be present without the dict
        # form (e.g. loaded from a cache). Fall back to the dict, then to a live
        # Dijkstra when nothing is precomputed.
        if self._pairwise_nodes_distance_array is not None and self._node_to_index:
            return float(
                self._pairwise_nodes_distance_array[
                    self._node_to_index[n1], self._node_to_index[n2]
                ]
            )
        if self._pairwise_nodes_distance is not None:
            return self._pairwise_nodes_distance[n1][n2]
        return nx.shortest_path_length(self, n1, n2, weight="length")

    def get_edge_length(self, n1: int, n2: int) -> float:
        """Get the length of an edge.

        Args:
            n1: First node of the edge.
            n2: Second node of the edge.

        Returns:
            Edge length.

        Raises:
            ValueError: If edge does not exist.
        """
        edge_data = self.get_edge_data(n1, n2)
        if edge_data is None:
            raise ValueError(f"Edge ({n1}, {n2}) does not exist in graph")
        return edge_data["length"]

    def quantum_path(self, p1: QGPoint, p2: QGPoint) -> dict[str, float | tuple | None]:
        """Compute the geodesic between two points on the graph.

        Args:
            p1: First point.
            p2: Second point.

        Returns:
            Dictionary with:
                - 'distance': geodesic distance
                - 'path': (node_from_p1_edge, node_from_p2_edge) or None if same edge
        """
        edge1, edge2 = p1.edge, p2.edge
        pos1, pos2 = p1.position, p2.position
        length1 = self.get_edge_length(*edge1)
        length2 = self.get_edge_length(*edge2)

        # Both points on the same self-loop: the geodesic stays on the loop
        # (direct arc or the complementary one, possibly through the vertex).
        if edge1[0] == edge1[1] and edge2[0] == edge2[1] and edge1[0] == edge2[0]:
            arc = abs(pos1 - pos2)
            return {"distance": min(arc, length1 - arc), "path": None}

        # Compute distances for all 4 possible paths
        d0 = self.node_distance(edge1[0], edge2[0]) + pos1 + pos2
        d1 = self.node_distance(edge1[0], edge2[1]) + pos1 + (length2 - pos2)
        d2 = self.node_distance(edge1[1], edge2[0]) + (length1 - pos1) + pos2
        d3 = (
            self.node_distance(edge1[1], edge2[1]) + (length1 - pos1) + (length2 - pos2)
        )

        # Find minimum distance (break ties deterministically)
        distances = np.array([d0, d1, d2, d3])
        all_idx = np.where(distances == distances.min())[0]
        idx = all_idx[0]
        d_min = distances[idx]

        # Check if points are on the same (non-loop) physical edge
        if (
            edge1[0] != edge1[1]
            and edge1[0] == edge2[1]
            and edge1[1] == edge2[0]
            and d_min > abs(length1 - pos1 - pos2)
        ):
            return {"distance": abs(length1 - pos1 - pos2), "path": None}

        if edge1[0] != edge1[1] and edge1 == edge2 and abs(pos1 - pos2) < d_min:
            return {"distance": abs(pos1 - pos2), "path": None}

        # Determine path nodes
        if idx == 0:
            path = (edge1[0], edge2[0])
        elif idx == 1:
            path = (edge1[0], edge2[1])
        elif idx == 2:
            path = (edge1[1], edge2[0])
        else:
            path = (edge1[1], edge2[1])

        return {"distance": d_min, "path": path}

    def distance(self, p1: QGPoint, p2: QGPoint) -> float:
        """Compute distance between two points on the graph.

        Args:
            p1: First point.
            p2: Second point.

        Returns:
            Geodesic distance.
        """
        return self.quantum_path(p1, p2)["distance"]

    def distances_from_centers(
        self, centers: list[QGCenter], target: QGPoint
    ) -> np.ndarray:
        """Compute distances from multiple centers to a single target point.

        This is a Numba-accelerated operation that efficiently computes distances
        from all centers to one target point. Works for any target location
        (on nodes or edges).

        Args:
            centers: List of k centers to compute distances from.
            target: The target point (can be on a node or edge).

        Returns:
            Array of shape (k,) with distances from each center to target.

        Raises:
            ValueError: If pairwise distances not precomputed.

        Example:
            ```python
            centers = graph.sample_centers(5)
            target = graph.sample_points(1)[0]
            distances = graph.distances_from_centers(centers, target)
            closest_idx = np.argmin(distances)
            closest_center = centers[closest_idx]
            ```
        """
        if self._pairwise_nodes_distance_array is None:
            raise ValueError("Must call precomputing() before distances_from_centers")

        k = len(centers)

        # Extract center data into numpy arrays for Numba
        center_edges_0 = np.empty(k, dtype=np.int32)
        center_edges_1 = np.empty(k, dtype=np.int32)
        center_positions = np.empty(k, dtype=np.float64)
        center_lengths = np.empty(k, dtype=np.float64)

        for i, center in enumerate(centers):
            edge = center.edge
            center_edges_0[i] = self._node_to_index[edge[0]]
            center_edges_1[i] = self._node_to_index[edge[1]]
            center_positions[i] = center.position
            center_lengths[i] = self.get_edge_length(*edge)

        # Extract target information
        target_edge = target.edge
        target_edge_0 = self._node_to_index[target_edge[0]]
        target_edge_1 = self._node_to_index[target_edge[1]]
        target_pos = target.position
        target_length = self.get_edge_length(*target_edge)

        # Call Numba-accelerated function
        return _batch_distances_numba(
            center_edges_0,
            center_edges_1,
            center_positions,
            center_lengths,
            target_edge_0,
            target_edge_1,
            target_pos,
            target_length,
            self._pairwise_nodes_distance_array,
        )

    def node_center_distances(self, centers: list[QGCenter]) -> np.ndarray:
        """Geodesic distance from every node to every center.

        Args:
            centers: List of k centers.

        Returns:
            Array of shape (n_nodes, k), with rows in ``list(self.nodes())``
            order: entry (v, i) is the geodesic distance from node v to center i.

        Raises:
            ValueError: If pairwise distances have not been precomputed.
        """
        if self._pairwise_nodes_distance_array is None:
            raise ValueError("Must call precomputing() before node_center_distances")

        distances = self._pairwise_nodes_distance_array  # (n, n), node order
        index = self._node_to_index
        out = np.empty((distances.shape[0], len(centers)))
        for i, center in enumerate(centers):
            a, b = center.edge
            length = self.get_edge_length(a, b)
            # A center sits at `position` along edge (a, b); a node reaches it
            # via whichever endpoint is closer.
            out[:, i] = np.minimum(
                center.position + distances[:, index[a]],
                (length - center.position) + distances[:, index[b]],
            )
        return out

    def node_energy(
        self, centers: list[QGCenter], weights: np.ndarray | None = None
    ) -> float:
        """Node-weighted k-means energy ``sum_v w[v] * min_i d(v, c_i)^2``.

        Args:
            centers: List of cluster centers.
            weights: Per-node weights in ``list(self.nodes())`` order. Defaults
                to the nodes' ``weight`` attribute (1.0 when absent).

        Returns:
            The weighted energy.
        """
        min_sq = self.node_center_distances(centers).min(axis=1) ** 2
        if weights is None:
            weights = np.array(
                [self.nodes[node].get("weight", 1.0) for node in self.nodes()]
            )
        return float(np.sum(np.asarray(weights) * min_sq))

    def center_from_point(self, point: QGPoint, rng=None) -> QGCenter:
        """Create a QGCenter object from a QGPoint object."""
        return QGCenter(point, rng=rng)

    def nodes_as_points(self) -> list[QGPoint]:
        """Convert all nodes to points.

        Returns:
            List of points, one at each node.
        """
        points = []
        for node in self.nodes:
            neighbor = next(self.neighbors(node))
            points.append(QGPoint(self, (node, neighbor), 0))
        return points

    def node_as_center(self, node: int, rng=None) -> QGCenter:
        """Create a center at a specific node.

        Args:
            node: The node to place the center at.
            rng: Optional random number generator for neighbor selection.

        Returns:
            A center located at the node.
        """
        if rng is not None:
            nbrs = list(self.neighbors(node))
            neighbor = nbrs[rng.integers(len(nbrs))]
        else:
            neighbor = next(self.neighbors(node))
        point = QGPoint(self, (node, neighbor), 0)
        return QGCenter(point, rng=rng)

    def calculate_energy(
        self,
        centers: list[QGCenter],
        how: str = "uniform",
        observations: list[QGPoint] | None = None,
    ) -> float:
        """Calculate k-means energy for given centers.

        Dispatches to the Numba kernels when pairwise node distances are
        precomputed (``precomputing()``), and falls back to pure-Python
        distance computations otherwise — correct on any graph, just slower.

        Args:
            centers: List of cluster centers.
            how: Which reference measure to average under (see
                :meth:`Space.calculate_energy` for the full contract):
                - "uniform": average over nodes with equal weight; rejects
                  ``observations``.
                - "empirical": average over the ``observations`` list
                  (required), each point exactly where it lies on its edge —
                  no rounding to nodes.
                - "node_measure": average over nodes weighted by their
                  ``obs_weight``, set explicitly by the caller (e.g. through
                  ``register_observations``); rejects ``observations``.
            observations: The points defining the empirical measure
                (``how="empirical"`` only).

        Returns:
            Average squared distance to nearest center.

        Raises:
            ValueError: If ``how`` is unknown (including the retired
                ``"obs"``), if ``"empirical"`` lacks observations, if
                ``"uniform"``/``"node_measure"`` receive some, or if no node
                carries a positive ``obs_weight`` in ``"node_measure"`` mode
                (the energy would silently be 0.0 for every configuration
                of centers).
        """
        if how == "obs":
            raise ValueError(
                "energy mode 'obs' was split into two explicit modes: use "
                "'empirical' to average over an explicit list of observation "
                "points, or 'node_measure' to average under the per-node "
                "'obs_weight' measure registered on the graph."
            )
        if how not in ("uniform", "empirical", "node_measure"):
            raise ValueError(
                f"how must be 'uniform', 'empirical' or 'node_measure', got {how!r}"
            )
        if how == "empirical":
            if not observations:
                raise ValueError(
                    "energy mode 'empirical' requires the explicit "
                    "'observations' list defining the empirical measure"
                )
        elif observations is not None:
            raise ValueError(
                f"energy mode {how!r} defines its own reference measure and "
                "does not take 'observations'; pass them with how='empirical' "
                "instead"
            )
        if how == "node_measure" and not any(
            data.get("obs_weight", 0) > 0 for _, data in self.nodes(data=True)
        ):
            raise ValueError(
                "energy mode 'node_measure' requires a measure on the graph "
                "nodes, but no node carries a positive 'obs_weight'. Register "
                "one explicitly: register_observations(points) counts each "
                "point at its closest node, or set the 'obs_weight' node "
                "attribute directly (e.g. to a population measure)."
            )
        if self._pairwise_nodes_distance_array is not None:
            return self.calculate_energy_numba(
                centers, how=how, observations=observations
            )
        return self._calculate_energy_python(centers, how, observations=observations)

    def register_observations(self, points: list[QGPoint]) -> None:
        """Set the per-node observation measure ``obs_weight`` from these points.

        Each point counts at its closest node. This **replaces** any previous
        measure — the measure describes one observation set — so callers
        combining several samples must register the union explicitly.

        Args:
            points: The observation points (any points of this graph).
        """
        nx.set_node_attributes(self, 0, "obs_weight")
        counts: dict = {}
        for point in points:
            node = point.closest_node()
            counts[node] = counts.get(node, 0) + 1
        nx.set_node_attributes(self, counts, "obs_weight")

    def _calculate_energy_python(
        self,
        centers: list[QGCenter],
        how: str,
        observations: list[QGPoint] | None = None,
    ) -> float:
        """Pure-Python energy fallback (no precomputed distances needed)."""
        if how == "uniform":
            energy = 0.0
            for node in self.nodes:
                neighbor = next(self.neighbors(node))
                point = QGPoint(self, (node, neighbor), 0)
                min_dist_sq = min(
                    self.distance(center, point) ** 2 for center in centers
                )
                energy += min_dist_sq
            return energy / self.number_of_nodes()
        elif how == "empirical":
            energy = sum(
                min(self.distance(center, point) ** 2 for center in centers)
                for point in observations
            )
            return energy / len(observations)
        else:  # how == "node_measure"
            energy = 0.0
            total_weight = 0
            for node, data in self.nodes(data=True):
                obs_weight = data.get("obs_weight", 0)
                if obs_weight > 0:
                    neighbor = next(self.neighbors(node))
                    point = QGPoint(self, (node, neighbor), 0)
                    min_dist_sq = min(
                        self.distance(center, point) ** 2 for center in centers
                    )
                    energy += min_dist_sq * obs_weight
                    total_weight += obs_weight

            return energy / total_weight if total_weight > 0 else 0.0

    def _points_as_arrays(
        self, points: list[QGPoint]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Edge indices, positions and edge lengths of points, for the kernels."""
        n = len(points)
        edges_0 = np.empty(n, dtype=np.int32)
        edges_1 = np.empty(n, dtype=np.int32)
        positions = np.empty(n, dtype=np.float64)
        lengths = np.empty(n, dtype=np.float64)
        for i, point in enumerate(points):
            edge = point.edge
            edges_0[i] = self._node_to_index[edge[0]]
            edges_1[i] = self._node_to_index[edge[1]]
            positions[i] = point.position
            lengths[i] = self.get_edge_length(*edge)
        return edges_0, edges_1, positions, lengths

    def _nodes_as_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Every node as an on-edge point at position 0, for the kernels."""
        nodes = list(self.nodes())
        n = len(nodes)
        edges_0 = np.empty(n, dtype=np.int32)
        edges_1 = np.empty(n, dtype=np.int32)
        positions = np.zeros(n, dtype=np.float64)
        lengths = np.empty(n, dtype=np.float64)
        for i, node in enumerate(nodes):
            neighbor = next(self.neighbors(node))
            edges_0[i] = self._node_to_index[node]
            edges_1[i] = self._node_to_index[neighbor]
            lengths[i] = self.get_edge_length(node, neighbor)
        return edges_0, edges_1, positions, lengths

    def calculate_energy_numba(
        self,
        centers: list[QGCenter],
        how: str = "uniform",
        observations: list[QGPoint] | None = None,
    ) -> float:
        """Numba-accelerated energy calculation for centers.

        Uses the precomputed distance matrix; ``calculate_energy`` dispatches
        here automatically once ``precomputing()`` has run.

        Args:
            centers: List of cluster centers.
            how: Reference measure — "uniform", "empirical" or "node_measure"
                (see ``calculate_energy`` for the contract; this method
                assumes the arguments were already validated there).
            observations: The points of the empirical measure
                (``how="empirical"`` only).

        Returns:
            Average squared distance to nearest center.

        Raises:
            ValueError: If pairwise distances not precomputed.
        """
        if self._pairwise_nodes_distance_array is None:
            raise ValueError("Must call precomputing() before calculate_energy_numba")

        center_arrays = self._points_as_arrays(centers)

        if how == "empirical":
            # The empirical measure of the observations: a uniform average
            # over the points themselves, exactly where they lie on their
            # edges (no rounding to nodes).
            point_arrays = self._points_as_arrays(observations)
            return _calculate_energy_uniform_numba(
                *center_arrays,
                *point_arrays,
                self._pairwise_nodes_distance_array,
            )

        point_arrays = self._nodes_as_arrays()
        if how == "uniform":
            return _calculate_energy_uniform_numba(
                *center_arrays,
                *point_arrays,
                self._pairwise_nodes_distance_array,
            )
        else:  # how == "node_measure"
            # obs_weight is a measure, not a count: integer counts come from
            # register_observations, but the paper experiments set the
            # fractional population measure nu. Casting to int here truncated
            # every fractional weight to zero, so the whole energy collapsed
            # to 0.0 -- silently, since 0 is a valid energy. Keep it float.
            point_obs_weight = np.array(
                [data.get("obs_weight", 0) for _, data in self.nodes(data=True)],
                dtype=np.float64,
            )
            return _calculate_energy_obs_numba(
                *center_arrays,
                *point_arrays,
                point_obs_weight,
                self._pairwise_nodes_distance_array,
            )

    def distance_matrix(self) -> np.ndarray:
        """Compute the pairwise distance matrix between all nodes.

        Returns:
            n×n matrix of pairwise distances.

        Raises:
            ValueError: If pairwise distances not precomputed.
        """
        if self._pairwise_nodes_distance is None:
            raise ValueError("Must call precomputing() first")

        n = self.number_of_nodes()
        node_list = list(self.nodes())
        matrix_distances = np.zeros((n, n))

        for i, node_i in enumerate(node_list):
            for j, node_j in enumerate(node_list):
                matrix_distances[i, j] = self._pairwise_nodes_distance[node_i][node_j]

        self.matrix_distance = matrix_distances
        return matrix_distances

    def index_to_centers(self, indices: list[int]) -> list[QGCenter]:
        """Convert node indices to centers.

        Args:
            indices: List of node indices.

        Returns:
            List of centers at the specified nodes.
        """
        node_list = list(self.nodes())
        return [self.node_as_center(node_list[idx]) for idx in indices]

    def draw(
        self,
        color_by: str | None = "cluster",
        centers: list[QGCenter] | None = None,
        node_size_by_obs: bool = True,
        ax: "plt.Axes" | None = None,
        **kwargs,
    ):
        """Draws the graph using matplotlib.

        This method provides a flexible way to visualize the graph, its clusters,
        and the results of the k-means algorithm. To use this method, you need
        to install the 'plot' extras: `pip install kmeanssa-ng[plot]`

        Args:
            color_by: Node attribute to use for coloring.
                - "cluster": Colors nodes by their assigned cluster.
                - "block": Colors nodes by a pre-defined 'block' attribute.
                - None: All nodes will have the same default color. (Default: "skyblue")
            centers: A list of QGCenter objects to be highlighted on the graph.
            node_size_by_obs: If True, sizes nodes based on 'obs_weight' attribute.
            ax: A matplotlib axes object to draw on. If None, a new figure
                and axes are created.
            **kwargs: Additional arguments passed to nx.draw().
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:
            raise ImportError(
                "Plotting requires matplotlib. Please install it with "
                "`pip install kmeanssa-ng[plot]`"
            ) from e

        if ax is None:
            fig, ax = plt.subplots(figsize=kwargs.pop("figsize", (12, 12)))

        # Use pre-computed layout for consistency
        pos = self.node_position

        # Determine node colors
        node_colors = "skyblue"  # Default color
        if color_by:
            # Check if any node has the attribute
            if not any(color_by in n[1] for n in self.nodes(data=True)):
                print(
                    f"Warning: Node attribute '{color_by}' not found. Using default color."
                )
            else:
                node_colors = [self.nodes[n].get(color_by, 0) for n in self.nodes()]

        # Determine node sizes
        node_sizes = 300  # Default size
        if node_size_by_obs:
            if not any("obs_weight" in n[1] for n in self.nodes(data=True)):
                print(
                    "Warning: Node attribute 'obs_weight' not found. Using default size."
                )
            else:
                # Scale size by number of observations, with a minimum size
                node_sizes = [
                    self.nodes[n].get("obs_weight", 0) * 50 + 100 for n in self.nodes()
                ]

        # Draw the graph
        nx.draw(
            self,
            pos=pos,
            ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            with_labels=False,
            **kwargs,
        )

        # Highlight centers
        if centers:
            center_nodes = [c.edge[0] for c in centers]
            node_list = list(self.nodes)
            center_node_indices = [node_list.index(n) for n in center_nodes]

            center_sizes = (
                [node_sizes[i] * 2 for i in center_node_indices]
                if isinstance(node_sizes, list)
                else node_sizes * 2
            )

            nx.draw_networkx_nodes(
                self,
                pos,
                nodelist=center_nodes,
                ax=ax,
                node_size=center_sizes,
                edgecolors="grey",
                linewidths=2,
            )

    def get_point_type(self) -> type[QGPoint]:
        """Return the type of points in this space."""
        return QGPoint
