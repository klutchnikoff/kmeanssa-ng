"""Sampling strategies for quantum graphs.

This module provides sampling strategies specific to quantum graphs,
allowing different probability distributions for selecting points on the graph.
"""

from __future__ import annotations

import random as rd
from typing import TYPE_CHECKING

import networkx as nx

from ..core.strategies.sampling import SamplingStrategy

if TYPE_CHECKING:
    from ..core.abstract import Point, Space


class UniformNodeSampling(SamplingStrategy):
    """Uniform sampling over graph nodes (discrete uniform distribution).

    This strategy samples points uniformly at random from the graph nodes,
    where each node has equal probability of being selected (ignoring node weights).

    This is a discrete uniform distribution for quantum graphs.

    Example:
        ```python
        from kmeanssa_ng import QuantumGraph
        from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling

        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        # Sample 100 points uniformly on nodes
        strategy = UniformNodeSampling()
        points = graph.sample_points(100, strategy=strategy)
        ```

    See Also:
        - UniformEdgeSampling: Continuous uniform sampling along edges
        - WeightedNodeSampling: Node sampling weighted by node weights
    """

    def sample(self, space: Space, n: int) -> list[Point]:
        """Sample n points uniformly from graph nodes.

        Args:
            space: The quantum graph to sample from.
            n: Number of points to sample.

        Returns:
            List of n points sampled uniformly from nodes.
        """
        from ..quantum_graph.space import QGPoint

        nx.set_node_attributes(space, 0, "nb_obs")
        points = []
        nodes = list(space.nodes())

        for _ in range(n):
            node = rd.choice(nodes)
            nb_obs = nx.get_node_attributes(space, "nb_obs").get(node, 0) + 1
            nx.set_node_attributes(space, {node: {"nb_obs": nb_obs}})
            neighbor = rd.choice(list(space.neighbors(node)))
            points.append(QGPoint(space, (node, neighbor), 0))

        return points


class UniformEdgeSampling(SamplingStrategy):
    """Uniform continuous sampling along graph edges.

    This strategy samples points uniformly along the edges of the graph,
    where the probability is proportional to edge length. Each point on
    the edges has equal probability density.

    This is a continuous uniform distribution for quantum graphs.

    Example:
        ```python
        from kmeanssa_ng import QuantumGraph
        from kmeanssa_ng.quantum_graph.sampling import UniformEdgeSampling

        graph = QuantumGraph()
        graph.add_edge(0, 1, length=2.0)
        graph.add_edge(1, 2, length=1.0)

        # Sample 100 points uniformly along edges
        # Edge (0,1) will get ~2/3 of points, edge (1,2) will get ~1/3
        strategy = UniformEdgeSampling()
        points = graph.sample_points(100, strategy=strategy)
        ```

    Note:
        Requires edges to have 'length' attribute (standard for QuantumGraph).
        Points are distributed proportionally to edge lengths.

    See Also:
        - UniformNodeSampling: Discrete uniform sampling at nodes
        - WeightedNodeSampling: Node sampling weighted by node weights
    """

    def sample(self, space: Space, n: int) -> list[Point]:
        """Sample n points uniformly along graph edges.

        Args:
            space: The quantum graph to sample from.
            n: Number of points to sample.

        Returns:
            List of n points sampled uniformly along edges.

        Raises:
            ValueError: If graph has no edges or edges lack 'length' attribute.
        """
        from ..quantum_graph.space import QGPoint

        edges = list(space.edges())
        if not edges:
            raise ValueError("Cannot sample from graph with no edges")

        # Get edge lengths
        edge_lengths = nx.get_edge_attributes(space, "length")
        if not edge_lengths:
            raise ValueError("Edges must have 'length' attribute for edge sampling")

        # Create list of edges and weights (lengths)
        edge_list = list(edge_lengths.keys())
        weights = [edge_lengths[e] for e in edge_list]

        points = []
        for _ in range(n):
            # Choose edge proportionally to its length
            edge = rd.choices(edge_list, weights=weights, k=1)[0]
            # Sample position uniformly along the edge
            position = rd.uniform(0, edge_lengths[edge])
            points.append(QGPoint(space, edge, position))

        return points


class WeightedNodeSampling(SamplingStrategy):
    """Weighted sampling over graph nodes using node weights.

    This strategy samples points from graph nodes with probability
    proportional to node weights. Nodes with higher 'weight' attribute
    are sampled more frequently.

    This is useful when nodes have different importance or when you want
    to oversample certain regions of the graph.

    Example:
        ```python
        from kmeanssa_ng import QuantumGraph
        from kmeanssa_ng.quantum_graph.sampling import WeightedNodeSampling

        graph = QuantumGraph()
        graph.add_edge(0, 1, length=1.0)
        graph.add_edge(1, 2, length=1.0)

        # Set node weights
        nx.set_node_attributes(graph, {0: 1.0, 1: 10.0, 2: 1.0}, "weight")

        # Sample 100 points - node 1 will get ~10x more points than nodes 0 and 2
        strategy = WeightedNodeSampling()
        points = graph.sample_points(100, strategy=strategy)
        ```

    Note:
        Requires nodes to have 'weight' attribute with positive values.

    See Also:
        - UniformNodeSampling: Unweighted discrete uniform sampling at nodes
        - UniformEdgeSampling: Continuous uniform sampling along edges
    """

    def sample(self, space: Space, n: int) -> list[Point]:
        """Sample n points from graph nodes weighted by node weights.

        Args:
            space: The quantum graph to sample from.
            n: Number of points to sample.

        Returns:
            List of n points sampled from nodes with probability proportional to weights.

        Raises:
            ValueError: If nodes lack 'weight' attribute or weights are invalid.
        """
        from ..quantum_graph.space import QGPoint

        node_weights = dict(nx.get_node_attributes(space, "weight"))

        if not node_weights:
            raise ValueError(
                "Nodes must have 'weight' attribute for weighted sampling. "
                "Use UniformNodeSampling for unweighted sampling."
            )

        # Validate weights are positive
        if any(w <= 0 for w in node_weights.values()):
            raise ValueError("All node weights must be positive")

        nx.set_node_attributes(space, 0, "nb_obs")
        points = []

        keys = list(node_weights.keys())
        values = list(node_weights.values())

        for _ in range(n):
            node = rd.choices(keys, weights=values, k=1)[0]
            nb_obs = nx.get_node_attributes(space, "nb_obs").get(node, 0) + 1
            nx.set_node_attributes(space, {node: {"nb_obs": nb_obs}})
            neighbor = rd.choice(list(space.neighbors(node)))
            points.append(QGPoint(space, (node, neighbor), 0))

        return points
