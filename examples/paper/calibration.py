"""Temperature calibration: the critical depth c* of the k=2 energy landscape.

The annealing drift is set to b = 0.3 / c*, where c* is the largest energy barrier
that must be crossed to move two centers between any pair of node configurations.
"""

import numpy as np
import networkx as nx


def potential_matrix(distances, nu):
    """U[i, j] = sum_v nu[v] * min(d(i,v)^2, d(j,v)^2): energy of the centers {i, j}."""
    squared = distances**2
    n = len(nu)
    U = np.empty((n, n))
    for i in range(n):
        U[i] = (nu * np.minimum(squared[i][None, :], squared)).sum(axis=1)
    return U


def critical_depth(U, graph, nodes, node_to_index):
    """Critical depth c* via the minimum spanning tree of the k=2 state space."""
    state_graph = _state_graph(U, graph, nodes, node_to_index)
    mst = nx.minimum_spanning_tree(state_graph)
    return _largest_barrier(mst, U, len(nodes))


def _state_graph(U, graph, nodes, node_to_index):
    """Graph over states (i, j) = pair of center nodes.

    Two states are adjacent when a single center hops to a neighbouring node; the
    edge weight is the higher of the two states' energies (the saddle to cross).
    """
    n = len(nodes)
    states = nx.Graph()
    states.add_nodes_from(range(n * n))
    for i in range(n):
        for j in range(n):
            state = n * i + j
            for neighbour in graph.neighbors(nodes[i]):  # move the first center
                i2 = node_to_index[neighbour]
                other = n * i2 + j
                if state < other:
                    states.add_edge(state, other, weight=max(U[i, j], U[i2, j]))
            for neighbour in graph.neighbors(nodes[j]):  # move the second center
                j2 = node_to_index[neighbour]
                other = n * i + j2
                if state < other:
                    states.add_edge(state, other, weight=max(U[i, j], U[i, j2]))
    return states


def _largest_barrier(mst, U, n):
    """Largest barrier needed to merge all basins, scanning MST edges by weight."""
    parent = list(range(n * n))
    basin_min = [U[s // n, s % n] for s in range(n * n)]

    def basin(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    barrier = max(-U[i, j] for i in range(n) for j in range(n))
    for a, b, data in sorted(mst.edges(data=True), key=lambda e: e[2]["weight"]):
        ra, rb = basin(a), basin(b)
        barrier = max(barrier, data["weight"] - basin_min[ra] - basin_min[rb])
        if ra != rb:
            parent[ra] = rb
            basin_min[rb] = min(basin_min[ra], basin_min[rb])
    return barrier + float(U.min())
