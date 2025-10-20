"""Quantum graph implementation as a metric space."""

from __future__ import annotations

import random as rd

import networkx as nx
import numpy as np
from numba import njit

from ..core import Space
from .center import QGCenter
from .point import QGPoint


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

    Computes distances from k centers to one target point using
    the quantum graph distance formula with 4 possible paths.

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
        c_edge_0 = center_edges_0[i]
        c_edge_1 = center_edges_1[i]
        c_pos = center_positions[i]
        c_length = center_lengths[i]

        # Compute 4 possible paths
        d0 = node_dist_matrix[c_edge_0, target_edge_0] + c_pos + target_pos
        d1 = (
            node_dist_matrix[c_edge_0, target_edge_1]
            + c_pos
            + (target_length - target_pos)
        )
        d2 = node_dist_matrix[c_edge_1, target_edge_0] + (c_length - c_pos) + target_pos
        d3 = (
            node_dist_matrix[c_edge_1, target_edge_1]
            + (c_length - c_pos)
            + (target_length - target_pos)
        )

        # Take minimum
        d_min = min(d0, d1, d2, d3)

        # Check same edge cases
        if c_edge_0 == target_edge_1 and c_edge_1 == target_edge_0:
            d_same_rev = abs(c_length - c_pos - target_pos)
            if d_same_rev < d_min:
                d_min = d_same_rev

        if c_edge_0 == target_edge_0 and c_edge_1 == target_edge_1:
            d_same = abs(c_pos - target_pos)
            if d_same < d_min:
                d_min = d_same

        distances[i] = d_min

    return distances


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

    def __init__(self, incoming_graph_data=None, **attr) -> None:
        """Initialize a quantum graph.

        Args:
            incoming_graph_data: Input graph data (see networkx.Graph).
            **attr: Additional graph attributes.
        """
        super().__init__(incoming_graph_data, **attr)
        self._pairwise_nodes_distance: dict[int, dict[int, float]] | None = None
        self._pairwise_nodes_distance_array: np.ndarray | None = None
        self._node_to_index: dict[int, int] | None = None
        self._diameter: float = 0.0
        self._node_position: dict | None = None

    def add_edge(self, u_for_edge, v_for_edge, **attr) -> None:
        """Add an edge with validation of the length attribute.

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

    @property
    def diameter(self) -> float:
        """Compute and cache the graph diameter.

        Returns:
            Maximum distance between any two nodes.
        """
        if self._diameter == 0.0:
            for n1 in self.nodes:
                for n2 in self.nodes:
                    d = self.distance_between_nodes(n1, n2)
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
            self._node_position = nx.layout.spring_layout(self, weight="drawing")
        return self._node_position

    def distance_between_nodes(self, n1: int, n2: int) -> float:
        """Compute shortest path distance between two nodes.

        Args:
            n1: First node.
            n2: Second node.

        Returns:
            Shortest path length using 'length' edge attribute.
        """
        if self._pairwise_nodes_distance is not None:
            return self._pairwise_nodes_distance[n1][n2]
        else:
            return nx.shortest_path_length(self, n1, n2, weight="length")

    def get_edge_length(self, n1: int, n2: int) -> float:
        """Get the length of an edge.

        Args:
            n1: First node of the edge.
            n2: Second node of the edge.

        Returns:
            Edge length (defaults to 0 if not found).
        """
        edge_data = self.get_edge_data(n1, n2, default={"length": 0})
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

        # Compute distances for all 4 possible paths
        d0 = self.distance_between_nodes(edge1[0], edge2[0]) + pos1 + pos2
        d1 = self.distance_between_nodes(edge1[0], edge2[1]) + pos1 + (length2 - pos2)
        d2 = self.distance_between_nodes(edge1[1], edge2[0]) + (length1 - pos1) + pos2
        d3 = (
            self.distance_between_nodes(edge1[1], edge2[1])
            + (length1 - pos1)
            + (length2 - pos2)
        )

        # Find minimum distance (break ties randomly)
        distances = np.array([d0, d1, d2, d3])
        all_idx = np.where(distances == distances.min())[0]
        idx = rd.choice(all_idx)
        d_min = distances[idx]

        # Check if points are on the same edge
        if (
            edge1[0] == edge2[1]
            and edge1[1] == edge2[0]
            and d_min > abs(length1 - pos1 - pos2)
        ):
            return {"distance": abs(length1 - pos1 - pos2), "path": None}

        if edge1 == edge2 and abs(pos1 - pos2) < d_min:
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

    def batch_distances_from_centers(
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
            distances = graph.batch_distances_from_centers(centers, target)
            closest_idx = np.argmin(distances)
            closest_center = centers[closest_idx]
            ```
        """
        if self._pairwise_nodes_distance_array is None:
            raise ValueError(
                "Must call precomputing() before batch_distances_from_centers"
            )

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

    def _sample_point(self, where: str = "Node") -> QGPoint:
        """Sample a random point on the graph.

        Args:
            where: Sampling mode ("Node" or "Edge").

        Returns:
            A randomly sampled point.

        Raises:
            ValueError: If 'where' is not "Node" or "Edge".
            NotImplementedError: If required attributes are missing.
        """
        if where == "Node":
            node_weights = dict(nx.get_node_attributes(self, "weight"))
            if node_weights:
                keys = list(node_weights.keys())
                values = list(node_weights.values())
                node = rd.choices(keys, weights=values, k=1)[0]
                nb_obs = nx.get_node_attributes(self, "nb_obs").get(node, 0) + 1
                nx.set_node_attributes(self, {node: {"nb_obs": nb_obs}})
                neighbor = rd.choice(list(self.neighbors(node)))
                return QGPoint(self, (node, neighbor), 0)
            else:
                raise NotImplementedError("Node sampling requires 'weight' attribute")

        elif where == "Edge":
            edge_weights = dict(nx.get_edge_attributes(self, "weight"))
            edge_distributions = dict(nx.get_edge_attributes(self, "distribution"))
            if edge_weights and edge_distributions:
                keys = list(edge_weights.keys())
                values = list(edge_weights.values())
                edge = rd.choices(keys, weights=values, k=1)[0]
                position = edge_distributions[edge]()
                return QGPoint(self, edge, position)
            else:
                raise NotImplementedError(
                    "Edge sampling requires 'weight' and 'distribution' attributes"
                )
        else:
            raise ValueError('The parameter "where" must be either "Node" or "Edge".')

    def sample_points(self, n: int, where: str = "Node") -> list[QGPoint]:
        """Sample n random points from the graph.

        Args:
            n: Number of points to sample.
            where: Sampling mode ("Node" or "Edge").

        Returns:
            List of n sampled points.
        """
        nx.set_node_attributes(self, 0, "nb_obs")
        return [self._sample_point(where) for _ in range(n)]

    def light_sample_points(self, n: int) -> list[QGPoint]:
        """Fast sampling of points at random nodes.

        Args:
            n: Number of points to sample.

        Returns:
            List of n points at random nodes.
        """
        nodes = rd.choices(list(self.nodes()), k=n)
        points = []
        for node in nodes:
            neighbor = rd.choice(list(self.neighbors(node)))
            points.append(QGPoint(self, (node, neighbor), 0))
        return points

    def _sample_center(self, where: str = "Node") -> QGCenter:
        """Sample a random center.

        Args:
            where: Sampling mode ("Node" or "Edge").

        Returns:
            A randomly sampled center.
        """
        return QGCenter(self._sample_point(where))

    def sample_centers(self, k: int, where: str = "Node") -> list[QGCenter]:
        """Sample k random centers.

        Args:
            k: Number of centers to sample.
            where: Sampling mode ("Node" or "Edge").

        Returns:
            List of k sampled centers.
        """
        return [self._sample_center(where) for _ in range(k)]

    def nodes_as_points(self) -> list[QGPoint]:
        """Convert all nodes to points.

        Returns:
            List of points, one at each node.
        """
        points = []
        for node in self.nodes:
            neighbor = rd.choice(list(self.neighbors(node)))
            points.append(QGPoint(self, (node, neighbor), 0))
        return points

    def node_as_center(self, node: int) -> QGCenter:
        """Create a center at a specific node.

        Args:
            node: The node to place the center at.

        Returns:
            A center located at the node.
        """
        neighbor = rd.choice(list(self.neighbors(node)))
        point = QGPoint(self, (node, neighbor), 0)
        return QGCenter(point)

    def sample_kpp_centers(self, k: int, where: str = "Node") -> list[QGCenter]:
        """Sample centers using k-means++ initialization.

        Args:
            k: Number of centers to sample.
            where: Sampling mode (currently only "Node" is fully supported).

        Returns:
            List of k centers sampled using k-means++.

        Raises:
            ValueError: If pairwise distances not precomputed.
        """
        if self._pairwise_nodes_distance is None:
            raise ValueError("Must call precomputing() before sample_kpp_centers")

        n = self.number_of_nodes()
        node_list = list(self.nodes())

        # Build distance matrix
        matrix_distances = np.zeros((n, n))
        for i, node_i in enumerate(node_list):
            for j, node_j in enumerate(node_list):
                matrix_distances[i, j] = self._pairwise_nodes_distance[node_i][node_j]

        # k-means++ initialization
        rng = np.random.default_rng()
        node_index = rd.sample(range(n), 1)
        centers_index = list(node_index)
        quantum_centers = [self.node_as_center(node_list[node_index[0]])]

        for _ in range(k - 1):
            dist_centers = matrix_distances[centers_index] ** 2
            if len(dist_centers.shape) == 1:
                min_dist = dist_centers
            else:
                min_dist = dist_centers.min(axis=0)

            prob = min_dist / sum(min_dist)
            new_center_index = rng.choice(n, size=1, replace=False, p=prob)
            centers_index.append(int(new_center_index[0]))
            new_center = node_list[new_center_index[0]]
            quantum_centers.append(self.node_as_center(new_center))

        return quantum_centers

    def compute_clusters(self, centers: list[QGCenter]) -> None:
        """Assign each node to its nearest center.

        Updates node 'cluster' attribute.

        Args:
            centers: List of cluster centers.
        """
        for node in self.nodes:
            distances = np.array(
                [
                    self.distance_between_nodes(center.edge[0], node)
                    for center in centers
                ]
            )
            nx.set_node_attributes(self, {node: {"cluster": np.argmin(distances)}})

    def calculate_energy_graph(
        self,
        centers: list[QGCenter],
        how: str = "uniform",
    ) -> float:
        """Calculate k-means energy for given centers.

        Args:
            centers: List of cluster centers.
            how: Energy calculation mode:
                - "uniform": Use uniform distribution over nodes
                - "obs": Weight by observed point counts at nodes

        Returns:
            Average squared distance to nearest center.
        """
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
        else:  # how == "obs"
            energy = 0.0
            total_obs = 0
            for node, data in self.nodes(data=True):
                nb_obs = data.get("nb_obs", 0)
                if nb_obs > 0:
                    neighbor = rd.choice(list(self.neighbors(node)))
                    point = QGPoint(self, (node, neighbor), 0)
                    min_dist_sq = min(
                        self.distance(center, point) ** 2 for center in centers
                    )
                    energy += min_dist_sq * nb_obs
                    total_obs += nb_obs

            return energy / total_obs if total_obs > 0 else 0.0

    def compute_matrix_distance(self) -> np.ndarray:
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
