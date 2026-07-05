"""Approximate a Riemannian manifold by a quantum graph built on an epsilon-net.

The construction places a quasi-uniform net (see :mod:`epsilon_net`), estimates
its covering radius epsilon, and connects points whose geodesic distance is at
most l(epsilon) = sqrt(epsilon) -- the approximation regime of the online
k-means theory. The result is a :class:`QuantumGraph` ready for clustering.
"""

from __future__ import annotations

import numpy as np
import networkx as nx
from sklearn.neighbors import NearestNeighbors

from ..quantum_graph.generators import UniformDistribution
from ..quantum_graph.space import QuantumGraph
from .epsilon_net import EpsilonNetStrategy, RepulsionNet
from .space import RiemannianManifold


def _geodesic(manifold: RiemannianManifold, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Elementwise geodesic distance between paired points ``a[i]``, ``b[i]``."""
    return manifold.norm(a, manifold.log(a, b))


def _geodesic_matrix(
    manifold: RiemannianManifold,
    sources: np.ndarray,
    targets: np.ndarray,
    *,
    buffer: int = 1_000_000,
) -> np.ndarray:
    """All-pairs geodesic distances ``(len(sources), len(targets))`` via log/norm.

    Uses only the manifold's intrinsic operations -- no embedding -- so it is
    correct on spaces (e.g. quotients) that have no faithful Euclidean embedding.
    Evaluated in row chunks to bound memory; O(len(sources) * len(targets)).
    """
    m, n = len(sources), len(targets)
    out = np.empty((m, n))
    chunk = max(1, buffer // max(n, 1))
    for lo in range(0, m, chunk):
        hi = min(lo + chunk, m)
        base = np.repeat(sources[lo:hi], n, axis=0)
        tiled = np.tile(targets, (hi - lo,) + (1,) * (targets.ndim - 1))
        out[lo:hi] = _geodesic(manifold, base, tiled).reshape(hi - lo, n)
    return out


def estimate_covering_radius(
    manifold: RiemannianManifold,
    points: np.ndarray,
    n_test: int = 10000,
    random_state: int | np.random.Generator | None = None,
    *,
    intrinsic: bool = False,
) -> float:
    """Estimate the covering radius max_z min_i d(z, x_i) by dense sampling.

    Nearest net points are found in ambient coordinates (exact when the ambient
    distance is monotone in the geodesic one, e.g. on the sphere), then the
    geodesic distance to that nearest point is measured exactly. With
    ``intrinsic=True`` the nearest net point is instead found by exhaustive
    geodesic distance -- for spaces without a faithful embedding (e.g. quotients),
    at O(n_test * n) cost.
    """
    rng = np.random.default_rng(random_state)
    test = manifold.random_uniform(n_test, rng)
    if intrinsic:
        return float(_geodesic_matrix(manifold, test, points).min(axis=1).max())
    _, idx = (
        NearestNeighbors(n_neighbors=1)
        .fit(manifold.embed(points))
        .kneighbors(manifold.embed(test))
    )
    nearest = points[idx[:, 0]]
    return float(_geodesic(manifold, test, nearest).max())


def build_epsilon_net_graph(
    manifold: RiemannianManifold,
    points: np.ndarray,
    ell: float | None = None,
    *,
    precompute: bool = True,
    covering_test: int = 10000,
    random_state: int | np.random.Generator | None = None,
    intrinsic: bool = False,
) -> QuantumGraph:
    """Connect an epsilon-net into a QuantumGraph (edges within ``ell``).

    Args:
        manifold: The manifold the points live on.
        points: An ``(n, d)`` epsilon-net (see :mod:`epsilon_net`).
        ell: Connection radius l(epsilon). Defaults to sqrt(covering radius).
        precompute: Precompute pairwise shortest-path distances on the graph.
        covering_test: Sample size used to estimate the covering radius.
        random_state: Seed for the covering-radius estimate.
        intrinsic: Build edges from exhaustive geodesic distances instead of an
            ambient nearest-neighbour prefilter. Required for spaces without a
            faithful embedding (e.g. quotients like the Bolza surface); O(n^2).

    Returns:
        A QuantumGraph with unit node/edge weights and geodesic edge lengths.

    Raises:
        ValueError: If the resulting graph is disconnected (net too sparse for
            ``ell``; use more points or a larger ``ell``).
    """
    n = len(points)
    if ell is None:
        eps = estimate_covering_radius(
            manifold, points, covering_test, random_state, intrinsic=intrinsic
        )
        ell = float(np.sqrt(eps))

    if intrinsic:
        # No embedding: filter on the exact all-pairs geodesic (quotient-aware).
        dmat = _geodesic_matrix(manifold, points, points)
        ii, jj = np.triu_indices(n, k=1)
        lengths = dmat[ii, jj]
        keep = lengths <= ell
        ii, jj, lengths = ii[keep], jj[keep], lengths[keep]
    else:
        # Ambient radius ell captures every pair within geodesic distance ell
        # (ambient distance <= geodesic distance); the exact geodesic filter follows.
        embedded = manifold.embed(points)
        adjacency = (
            NearestNeighbors(radius=ell)
            .fit(embedded)
            .radius_neighbors_graph(embedded, radius=ell, mode="connectivity")
            .tocoo()
        )
        upper = adjacency.row < adjacency.col
        ii, jj = adjacency.row[upper], adjacency.col[upper]
        lengths = _geodesic(manifold, points[ii], points[jj])
        keep = lengths <= ell
        ii, jj, lengths = ii[keep], jj[keep], lengths[keep]

    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_weighted_edges_from(
        zip(ii.tolist(), jj.tolist(), lengths.tolist()), weight="length"
    )
    if not nx.is_connected(graph):
        raise ValueError(
            "The epsilon-net graph is disconnected; use more points or a larger ell."
        )

    qg = QuantumGraph(graph, precompute=False)
    nx.set_node_attributes(qg, 1.0, "weight")
    for edge in qg.edges:
        length = qg.get_edge_data(*edge)["length"]
        nx.set_edge_attributes(
            qg, {edge: {"weight": 1.0, "distribution": UniformDistribution(length)}}
        )
    if precompute:
        qg.precomputing()
    return qg


def approximate_geodesic_space(
    manifold: RiemannianManifold,
    n: int,
    *,
    net: EpsilonNetStrategy | None = None,
    ell: float | None = None,
    random_state: int | np.random.Generator | None = None,
    intrinsic: bool = False,
) -> QuantumGraph:
    """Approximate ``manifold`` by a quantum graph on an ``n``-point epsilon-net.

    Args:
        manifold: The geodesic space to approximate.
        n: Number of net points.
        net: Placement strategy (defaults to :class:`RepulsionNet`).
        ell: Connection radius (defaults to sqrt of the covering radius).
        random_state: Seed for the net and the covering-radius estimate.
        intrinsic: Build the net and graph without an ambient embedding (required
            for quotient spaces such as the Bolza surface).
    """
    if net is None:
        net = RepulsionNet(random_state=random_state)
    points = net.build(manifold, n)
    return build_epsilon_net_graph(
        manifold, points, ell, random_state=random_state, intrinsic=intrinsic
    )
