"""Generators for quantum graphs used in testing and benchmarking."""

from __future__ import annotations

import random as rd

import networkx as nx
import numpy as np

from .space import QuantumGraph


def generate_simple_graph(
    n_a: int = 5,
    n_aa: int = 3,
    bridge_length: float = 2.0,
    **attr,
) -> QuantumGraph:
    """Generate a symmetric two-cluster graph connected by a bridge.

    Creates a graph with two symmetric star-like clusters (A and B) connected
    by a single edge. Each cluster has a central node with n_a neighbors,
    and each neighbor has n_aa further neighbors.

    Args:
        n_a: Number of neighbors for each central node.
        n_aa: Number of second-level neighbors.
        bridge_length: Length of the edge connecting the two clusters.
        **attr: Additional graph attributes.

    Returns:
        A quantum graph with two symmetric clusters.

    Example:
        ```python
        graph = generate_simple_graph(n_a=5, n_aa=3, bridge_length=2.0)
        ```
    """
    graph = QuantumGraph(**attr)

    # Add central nodes
    graph.add_node("A0", weight=1)
    graph.add_node("B0", weight=1)
    graph.add_edge("A0", "B0", length=bridge_length)

    # Build cluster A
    for i in range(1, n_a + 1):
        node_a = f"A{i}"
        graph.add_node(node_a, weight=1)
        graph.add_edge("A0", node_a, length=1.0)

        for j in range(1, n_aa + 1):
            node_aa = f"{node_a}{j}"
            graph.add_node(node_aa, weight=1)
            graph.add_edge(node_a, node_aa, length=1.0)

    # Build cluster B (symmetric to A)
    for i in range(1, n_a + 1):
        node_b = f"B{i}"
        graph.add_node(node_b, weight=1)
        graph.add_edge("B0", node_b, length=1.0)

        for j in range(1, n_aa + 1):
            node_bb = f"{node_b}{j}"
            graph.add_node(node_bb, weight=1)
            graph.add_edge(node_b, node_bb, length=1.0)

    # Set edge weights and distributions
    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"weight": 1.0}})
        edge_length = graph.get_edge_data(*edge)["length"]
        # Proper closure to capture edge_length
        distrib = lambda L=edge_length: rd.uniform(0, L)
        nx.set_edge_attributes(graph, {edge: {"distribution": distrib}})

    graph.precomputing()
    return graph


def generate_simple_random_graph(
    n_a: int = 5,
    n_b: int = 5,
    lam_a: int = 0,
    lam_b: int = 0,
    bridge_length: float = 10.0,
    **attr,
) -> QuantumGraph:
    """Generate a random two-cluster graph with Poisson branching.

    Similar to generate_simple_graph but with:
    - Asymmetric clusters (different sizes)
    - Random edge lengths
    - Poisson-distributed third-level neighbors

    Args:
        n_a: Number of first-level neighbors of A0.
        n_b: Number of first-level neighbors of B0.
        lam_a: Poisson parameter for A cluster third-level branching.
        lam_b: Poisson parameter for B cluster third-level branching.
        bridge_length: Mean length of the bridge edge (actual length is uniform random).
        **attr: Additional graph attributes.

    Returns:
        A random quantum graph with two clusters.
    """
    graph = QuantumGraph(**attr)
    rng = np.random.default_rng()

    # Central nodes and bridge
    graph.add_node("A0", weight=5)
    graph.add_node("B0", weight=5)
    graph.add_edge("A0", "B0", length=rd.uniform(0.9 * bridge_length, 1.1 * bridge_length))

    # Build cluster A
    for i in range(1, n_a + 1):
        node_a = f"A{i}"
        graph.add_node(node_a, weight=3)
        graph.add_edge("A0", node_a, length=rd.uniform(0.9, 1.1))

        # Poisson-distributed third level
        num_children = rng.poisson(lam_a)
        for j in range(1, num_children + 1):
            node_aa = f"{node_a}{j}"
            graph.add_node(node_aa, weight=1)
            graph.add_edge(node_a, node_aa, length=rd.uniform(0.4, 0.6))

    # Build cluster B
    for i in range(1, n_b + 1):
        node_b = f"B{i}"
        graph.add_node(node_b, weight=3)
        graph.add_edge("B0", node_b, length=rd.uniform(0.9, 1.1))

        num_children = rng.poisson(lam_b)
        for j in range(1, num_children + 1):
            node_bb = f"{node_b}{j}"
            graph.add_node(node_bb, weight=1)
            graph.add_edge(node_b, node_bb, length=rd.uniform(0.4, 0.6))

    # Set edge weights and distributions
    node_weights = nx.get_node_attributes(graph, "weight")
    for edge in graph.edges:
        # Harmonic mean of node weights
        w = 0.5 / (1.0 / node_weights[edge[0]] + 1.0 / node_weights[edge[1]])
        nx.set_edge_attributes(graph, {edge: {"weight": w}})

        edge_length = graph.get_edge_data(*edge)["length"]
        distrib = lambda L=edge_length: rd.uniform(0, L)
        nx.set_edge_attributes(graph, {edge: {"distribution": distrib}})

    graph.precomputing()
    return graph


def generate_sbm(
    sizes: list[int] | None = None,
    p: list[list[float]] | None = None,
) -> QuantumGraph:
    """Generate a Stochastic Block Model quantum graph.

    Creates a quantum graph from a stochastic block model with uniform
    edge lengths and node weights.

    Args:
        sizes: Number of nodes in each block. Defaults to [50, 50].
        p: Matrix of edge probabilities. Element (r, s) gives the density
            of edges from block r to block s. Must be symmetric for undirected graphs.
            Defaults to [[0.7, 0.1], [0.1, 0.7]].

    Returns:
        A quantum graph representing the SBM.

    Example:
        ```python
        # Two balanced clusters with high intra-cluster, low inter-cluster edges
        graph = generate_sbm(sizes=[50, 50], p=[[0.7, 0.1], [0.1, 0.7]])
        ```
    """
    if sizes is None:
        sizes = [50, 50]
    if p is None:
        p = [[0.7, 0.1], [0.1, 0.7]]

    nx_graph = nx.stochastic_block_model(sizes=sizes, p=p)
    graph = QuantumGraph(nx_graph, attr=nx.get_node_attributes(nx_graph, name="block"))

    # Set uniform attributes
    nx.set_node_attributes(graph, 1, "weight")
    nx.set_edge_attributes(graph, 1, "length")

    for edge in graph.edges:
        nx.set_edge_attributes(graph, {edge: {"weight": 1}})
        edge_length = graph.get_edge_data(*edge)["length"]
        distrib = lambda L=edge_length: rd.uniform(0, L)
        nx.set_edge_attributes(graph, {edge: {"distribution": distrib}})

    graph.precomputing()
    return graph


def generate_random_sbm(
    sizes: list[int] | None = None,
    p: list[list[float]] | None = None,
    weights: list[float] | None = None,
    lengths: list[list[float]] | None = None,
) -> QuantumGraph:
    """Generate an SBM quantum graph with block-specific edge lengths and node weights.

    Args:
        sizes: Number of nodes in each block. Defaults to [50, 50].
        p: Matrix of edge probabilities. Defaults to [[0.7, 0.1], [0.1, 0.7]].
        weights: Node weight for each block. Defaults to [1, 1].
        lengths: Matrix of edge lengths. Element (i, j) gives the length
            for edges between blocks i and j. Defaults to [[1, 4], [4, 1]].

    Returns:
        A quantum graph with block-specific attributes.

    Example:
        ```python
        # Two clusters with different intra/inter-cluster distances
        graph = generate_random_sbm(
            sizes=[50, 50],
            p=[[0.7, 0.1], [0.1, 0.7]],
            weights=[1, 1],
            lengths=[[1, 4], [4, 1]]  # Longer inter-cluster edges
        )
        ```
    """
    if sizes is None:
        sizes = [50, 50]
    if p is None:
        p = [[0.7, 0.1], [0.1, 0.7]]
    if weights is None:
        weights = [1, 1]
    if lengths is None:
        lengths = [[1, 4], [4, 1]]

    nx_graph = nx.stochastic_block_model(sizes=sizes, p=p)
    graph = QuantumGraph(nx_graph)

    # Set node weights based on block
    for node in graph.nodes:
        block_idx = nx.get_node_attributes(graph, name="block")[node]
        w = weights[block_idx]
        nx.set_node_attributes(graph, {node: {"weight": w}})

    # Set edge lengths based on blocks
    for edge in graph.edges:
        block_i = nx.get_node_attributes(graph, name="block")[edge[0]]
        block_j = nx.get_node_attributes(graph, name="block")[edge[1]]
        edge_length = lengths[block_i][block_j]

        nx.set_edge_attributes(graph, {edge: {"length": edge_length, "weight": 1}})
        distrib = lambda L=edge_length: rd.uniform(0, L)
        nx.set_edge_attributes(graph, {edge: {"distribution": distrib}})

    graph.precomputing()
    return graph


def as_quantum_graph(
    graph: nx.Graph,
    node_weight: float = 1.0,
    edge_length: float = 1.0,
    edge_weight: float = 1.0,
) -> QuantumGraph:
    """Convert a NetworkX graph to a quantum graph with uniform attributes.

    Args:
        graph: The NetworkX graph to convert.
        node_weight: Uniform weight to assign to all nodes.
        edge_length: Uniform length to assign to all edges.
        edge_weight: Uniform weight to assign to all edges.

    Returns:
        The converted quantum graph.

    Example:
        ```python
        import networkx as nx
        G = nx.karate_club_graph()
        qg = as_quantum_graph(G, edge_length=1.0)
        ```
    """
    qg = QuantumGraph(graph)
    nx.set_node_attributes(qg, node_weight, "weight")
    nx.set_edge_attributes(qg, edge_length, "length")
    nx.set_edge_attributes(qg, edge_weight, "weight")

    distrib = lambda L=edge_length: rd.uniform(0, L)
    for edge in qg.edges:
        nx.set_edge_attributes(qg, {edge: {"distribution": distrib}})

    return qg


def complete_quantum_graph(
    objects: list,
    similarities: np.ndarray | None = None,
    true_labels: list | None = None,
) -> QuantumGraph:
    """Create a complete quantum graph from objects with optional similarity matrix.

    Useful for clustering when you have a pairwise distance/similarity matrix.

    Args:
        objects: List of objects (nodes will be indexed by position).
        similarities: Optional n×n matrix of similarities/distances. If None, all edges
            have length 1.
        true_labels: Optional true cluster labels for each object.

    Returns:
        A complete quantum graph where edge lengths are given by the similarity matrix.

    Example:
        ```python
        objects = [1, 2, 3, 4, 5]
        similarities = np.array([...])  # 5x5 matrix
        labels = [0, 0, 1, 1, 1]
        graph = complete_quantum_graph(objects, similarities, labels)
        ```
    """
    graph = QuantumGraph()

    # Add nodes
    for i, _ in enumerate(objects):
        if true_labels is not None:
            graph.add_node(i, weight=1, group=true_labels[i])
        else:
            graph.add_node(i, weight=1)

    # Add all edges (complete graph)
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if similarities is not None:
                edge_length = similarities[i][j]
            else:
                edge_length = 1.0

            distrib = lambda L=edge_length: rd.uniform(0, L)
            graph.add_edge(i, j, weight=1, length=edge_length, distribution=distrib)

    graph.precomputing()
    return graph
