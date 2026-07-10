"""Center class for quantum graphs with Brownian motion and drift."""

from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng

from ..core import Center
from .point import QGPoint


class QGCenter(QGPoint, Center):
    """A movable cluster center on a quantum graph.

    Centers can perform:
    - Brownian motion: Random walk for exploration
    - Drift: Directed movement toward target points

    Attributes:
        space: The quantum graph this center belongs to.
        edge: The edge containing this center.
        position: Position along the edge.

    Example:
        ```python
        graph = QuantumGraph(...)
        point = QGPoint(graph, edge=(0, 1), position=0.5)
        center = QGCenter(point)
        center.brownian_motion(0.1)
        center.drift(target_point, 0.5)
        ```
    """

    def __init__(
        self,
        point: QGPoint,
        rng: Generator | None = None,
    ) -> None:
        """Initialize a center from a point.

        Args:
            point: The initial point location.
            rng: Random number generator. If None, creates a new default_rng().
        """
        super().__init__(point.space, point.edge, point.position)
        self._rng = rng if rng is not None else default_rng()

    def _find_best_neighbor(self, n1: int, n2: int) -> int:
        """Find the neighbor of n1 whose edge starts a shortest walk to n2.

        The walk physically traverses the edge (n1, neighbor), so the cost of
        a candidate is the *length of that edge* plus the shortest path from
        the neighbor onward. Scoring with ``node_distance(n1, neighbor)``
        instead would tie a long direct edge with the true route whenever a
        shortcut exists (the shortest-path relaxation hides the edge length),
        and the drift would wander off the geodesic.

        Args:
            n1: Starting node.
            n2: Target node.

        Returns:
            A neighbor of n1 minimizing the walk to n2 (ties broken at random
            among neighbors that all realize the same minimal walk length).
        """
        if n1 == n2:
            return n1

        neighbors = list(self.space.neighbors(n1))
        min_distance = np.inf
        best_neighbors = []

        for neighbor in neighbors:
            distance = self.space.get_edge_length(n1, neighbor) + (
                self.space.node_distance(neighbor, n2)
            )
            if distance < min_distance:
                min_distance = distance
                best_neighbors = [neighbor]
            elif distance == min_distance:
                best_neighbors.append(neighbor)

        return best_neighbors[self._rng.integers(len(best_neighbors))]

    def drift(self, target_point: QGPoint, prop_to_travel: float) -> None:
        """Move toward a target point.

        Moves a proportion of the distance to the target point along
        the geodesic path in the quantum graph.

        Args:
            target_point: The point to move toward.
            prop_to_travel: Proportion of distance to travel (0 to 1).

        Raises:
            ValueError: If target_point is None, prop_to_travel is not numeric,
                or prop_to_travel is not in [0, 1].
        """
        # Validate target_point
        if target_point is None:
            raise ValueError("target_point cannot be None")

        # Validate prop_to_travel
        try:
            prop_float = float(prop_to_travel)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"prop_to_travel must be a number, got {type(prop_to_travel).__name__}"
            ) from e

        if prop_float < 0 or prop_float > 1:
            raise ValueError(f"prop_to_travel must be in [0, 1], got {prop_float}")

        quantum_path = self.space.quantum_path(self, target_point)
        path = quantum_path["path"]
        dist_to_target = quantum_path["distance"]
        dist_to_travel = prop_to_travel * dist_to_target

        # Case 1: Same edge (simple movement)
        if path is None:
            self._drift_on_same_edge(target_point, dist_to_travel)
        else:
            # Case 2: Different edges (traverse graph)
            self._drift_across_edges(target_point, path, dist_to_travel)

    def _drift_on_same_edge(self, target_point: QGPoint, dist_to_travel: float) -> None:
        """Handle drift when center and target are on the same physical edge."""
        edge_length = self.space.get_edge_length(*self.edge)

        if self.edge[0] == self.edge[1]:
            # Self-loop: move along the shorter arc, wrapping at the vertex.
            delta = target_point.position - self.position
            if delta > edge_length / 2:
                delta -= edge_length
            elif delta < -edge_length / 2:
                delta += edge_length
            step = dist_to_travel if delta >= 0 else -dist_to_travel
            self.position = (self.position + step) % edge_length
            return

        if self.edge == target_point.edge:
            target_position = target_point.position
        else:
            # Same physical edge, opposite parametrization: in this center's
            # frame the target sits at (edge length - target position).
            target_position = edge_length - target_point.position

        if self.position < target_position:
            self.position += dist_to_travel
        else:
            self.position -= dist_to_travel

    def _drift_across_edges(
        self,
        target_point: QGPoint,
        path: tuple[int, int],
        dist_to_travel: float,
    ) -> None:
        """Handle drift when center and target are on different edges."""
        next_node, target_node = path

        # Orient this center so the traversal exits forward (position
        # increasing). On a self-loop, reverse() is a reflection across the
        # vertex -- a *different physical point*, not a reparametrization --
        # so it is only safe when the walk provably leaves the loop (the
        # reflected and true trajectories then meet at the vertex). A walk
        # that stops on the loop must move along the true exit arc directly.
        if self.edge[0] == self.edge[1]:
            edge_length = self.space.get_edge_length(*self.edge)
            backward = self.position < edge_length - self.position
            arc = self.position if backward else edge_length - self.position
            if dist_to_travel < arc:
                self.position += -dist_to_travel if backward else dist_to_travel
                return
            if backward:
                self.reverse()
        elif self.edge[0] == next_node:
            self.reverse()

        # Orient the target locally: the observation belongs to the caller
        # and must never be mutated (on a self-loop, reverse() would even
        # move it to a different point of the graph).
        target_edge = target_point.edge
        target_length = self.space.get_edge_length(*target_edge)
        enter_far_arc = False
        if target_edge[0] == target_edge[1]:
            # Self-loop target: the geodesic enters via the shorter arc. Via
            # the far end, the coordinate runs backward from the loop length.
            enter_far_arc = (
                target_point.position > target_length - target_point.position
            )
        elif target_edge[1] == target_node:
            target_edge = (target_edge[1], target_edge[0])

        remaining_dist = dist_to_travel
        dist_to_next_node = self.space.get_edge_length(*self.edge) - self.position
        on_target_edge = False

        # Traverse edges until we've traveled the required distance
        while remaining_dist > dist_to_next_node:
            remaining_dist -= dist_to_next_node

            if target_node == next_node:
                # Reached target edge
                self.position = 0
                self.edge = target_edge
                on_target_edge = True
                dist_to_next_node = remaining_dist + 1  # Exit condition
            else:
                # Move to next edge
                self.position = 0
                cur_node = next_node
                next_node = self._find_best_neighbor(cur_node, target_node)
                self.edge = (cur_node, next_node)
                dist_to_next_node = self.space.get_edge_length(*self.edge)

        if on_target_edge and enter_far_arc:
            self.position = target_length - remaining_dist
        else:
            self.position += remaining_dist

    def clone(self) -> QGCenter:
        """Create an independent copy of this center.

        The cloned center shares the same quantum graph (space) but has
        independent edge and position attributes. This is much faster than
        deepcopy as it doesn't duplicate the entire graph structure.

        Note: This creates a shallow copy of the center's state, bypassing
        __init__ validation: the state comes from an existing center, so it
        is already valid, and cloning happens on the hot path of the
        annealing loop.

        Returns:
            A new QGCenter with the same location but independent state.

        Example:
            ```python
            original = QGCenter(...)
            copy = original.clone()
            original.brownian_motion(0.1)  # Doesn't affect copy
            ```
        """
        new_center = object.__new__(QGCenter)
        new_center._quantum_graph = self._quantum_graph
        new_center._edge = self._edge
        new_center.position = self.position
        new_center._rng = self._rng
        return new_center

    def brownian_motion(self, time_to_travel: float) -> None:
        """Perform Brownian motion on the quantum graph.

        The center performs a random walk with step size proportional
        to sqrt(time_to_travel).

        Args:
            time_to_travel: Time parameter (distance ~ sqrt(time)).

        Raises:
            ValueError: If time_to_travel is negative or not numeric.
        """
        # Validate time_to_travel
        try:
            time_float = float(time_to_travel)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"time_to_travel must be a number, got {type(time_to_travel).__name__}"
            ) from e

        if time_float < 0:
            raise ValueError(f"time_to_travel must be non-negative, got {time_float}")

        dist_to_travel = np.sqrt(time_float) * self._rng.standard_normal()
        edge_length = self.space.get_edge_length(*self.edge)

        forward = dist_to_travel > 0
        if forward:
            next_node = self.edge[1]
            dist_to_next_node = edge_length - self.position
        else:
            next_node = self.edge[0]
            dist_to_next_node = self.position

        remaining_dist = abs(dist_to_travel)

        # Traverse edges if motion exceeds current edge
        while remaining_dist >= dist_to_next_node:
            remaining_dist -= dist_to_next_node
            cur_node = next_node
            # Brownian motion on a metric graph leaves a vertex through an
            # *edge-end* chosen uniformly (Kirchhoff conditions). A self-loop
            # is attached to the vertex by both of its ends, so it counts
            # twice -- once entered forward (from position 0), once backward
            # (from the loop length).
            ends = []
            for nbr in self.space.neighbors(cur_node):
                ends.append((nbr, True))
                if nbr == cur_node:
                    ends.append((nbr, False))
            next_node, forward = ends[self._rng.integers(len(ends))]
            self.edge = (cur_node, next_node)
            edge_length = self.space.get_edge_length(*self.edge)
            # Whatever the sign of the initial draw, the leftover motion
            # continues away from the vertex: toward next_node on an edge
            # entered forward, toward decreasing positions on a self-loop
            # entered through its far end.
            self.position = 0 if forward else edge_length
            dist_to_next_node = edge_length

        # Final position update
        self.position += remaining_dist if forward else -remaining_dist

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"QGCenter(edge={self.edge}, position={self.position:.3f})"

    def __str__(self) -> str:
        """User-friendly string representation."""
        closest = self.closest_node()
        return f"Center near node {closest} [edge {self.edge}, pos={self.position:.3f}]"
