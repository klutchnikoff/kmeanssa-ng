"""Point class for quantum graphs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..core import Point

if TYPE_CHECKING:
    from .space import QuantumGraph


class QGPoint(Point):
    """A point on a quantum graph.

    A quantum graph point is located on an edge at a specific position.
    The edge is represented as a tuple (node1, node2) and the position
    is the distance from node1 along the edge.

    Attributes:
        space: The quantum graph this point belongs to.
        edge: The edge (node1, node2) containing this point.
        position: Distance from node1 along the edge.

    Example:
        ```python
        graph = QuantumGraph(...)
        point = QGPoint(graph, edge=(0, 1), position=0.5)
        ```
    """

    def __init__(
        self,
        quantum_graph: QuantumGraph,
        edge: tuple[int, int],
        position: float,
    ) -> None:
        """Initialize a point on a quantum graph.

        Args:
            quantum_graph: The quantum graph containing this point.
            edge: Tuple (node1, node2) representing the edge.
            position: Distance from node1 along the edge.

        Raises:
            ValueError: If quantum_graph is None, edge doesn't exist in graph,
                or position is outside [0, edge_length].
        """
        if quantum_graph is None:
            raise ValueError("quantum_graph cannot be None")

        if not isinstance(edge, tuple) or len(edge) != 2:
            raise ValueError(f"edge must be a tuple of two nodes, got {edge}")

        # Check edge exists in graph (allow self-loops)
        if edge[0] != edge[1] and edge not in quantum_graph.edges:
            # Try reversed edge
            if (edge[1], edge[0]) not in quantum_graph.edges:
                raise ValueError(f"Edge {edge} does not exist in the graph")

        self._quantum_graph = quantum_graph
        self._edge = edge

        # Validate and set position
        self._validate_and_set_position(position)

    def _validate_and_set_position(self, position: float) -> None:
        """Validate and set the position on the edge.

        Args:
            position: Position along the edge from node1.

        Raises:
            ValueError: If position is not numeric or outside [0, edge_length].
        """
        # Check if position is numeric
        try:
            position_float = float(position)
        except (TypeError, ValueError) as e:
            raise ValueError(
                f"Position must be a number, got {type(position).__name__}"
            ) from e

        # Check if position is non-negative
        if position_float < 0:
            raise ValueError(f"Position must be non-negative, got {position_float}")

        # Check if position is within edge length
        edge_length = self.space.get_edge_length(*self._edge)
        if position_float > edge_length:
            raise ValueError(
                f"Position {position_float} exceeds edge length {edge_length} for edge {self._edge}"
            )

        self.position = position_float

    @property
    def space(self) -> QuantumGraph:
        """The quantum graph this point belongs to."""
        return self._quantum_graph

    @property
    def edge(self) -> tuple[int, int]:
        """The edge containing this point."""
        if self._edge not in self.space.edges:
            if self._edge[0] == self._edge[1]:
                return self._edge
            else:
                raise ValueError("The edge does not belong to the graph")
        return self._edge

    @edge.setter
    def edge(self, new_edge: tuple[int, int]) -> None:
        """Set the edge containing this point.

        Args:
            new_edge: New edge (node1, node2).

        Raises:
            ValueError: If the edge doesn't belong to the graph.

        Note:
            Position validation is not performed here to allow internal
            operations (brownian_motion, drift) to change edge first, then
            update position. Position validation is done in __init__.
        """
        if not ((new_edge in self.space.edges) or (new_edge[0] == new_edge[1])):
            raise ValueError("The edge does not belong to the graph")

        self._edge = new_edge

    def _closest_node(self) -> int:
        """Get the closest node to this point.

        Returns:
            The node (edge[0] or edge[1]) closest to this point.
        """
        edge_length = self.space.get_edge_length(*self.edge)
        if self.position < edge_length / 2:
            return self.edge[0]
        else:
            return self.edge[1]

    def reverse(self) -> None:
        """Reverse the edge orientation and adjust position.

        Changes edge from (a, b) to (b, a) and updates position accordingly.
        """
        self.edge = (self.edge[1], self.edge[0])
        edge_length = self.space.get_edge_length(*self.edge)
        self.position = edge_length - self.position

    def __str__(self) -> str:
        """String representation of the point."""
        name = (
            f" '{self.space.name}'"
            if hasattr(self.space, "name") and self.space.name
            else ""
        )
        return f"QGPoint on{name} edge {self.edge} at position {self.position:.3f}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"QGPoint(edge={self.edge}, position={self.position})"
