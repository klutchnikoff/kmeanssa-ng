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
        """
        self._quantum_graph = quantum_graph
        self._edge = edge
        self.position = position

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
        """
        if (new_edge in self.space.edges) or (new_edge[0] == new_edge[1]):
            self._edge = new_edge
        else:
            raise ValueError("The edge does not belong to the graph")

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
        name = f" '{self.space.name}'" if hasattr(self.space, "name") and self.space.name else ""
        return f"QGPoint on{name} edge {self.edge} at position {self.position:.3f}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"QGPoint(edge={self.edge}, position={self.position})"
