"""Grid 10x10 experiment: a two-component generative density; ARI at the data level."""

import _env  # noqa: F401  -- pins BLAS threads; must precede numpy
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import networkx as nx

from kmeanssa_ng import as_quantum_graph, QGPoint
import baselines as B
from calibration import potential_matrix, critical_depth
from multistart import (
    annealings,
    method_entropy,
    methods_from_raw,
    summarize,
    run_seeds,
)

MODES = [(1, 1), (8, 8)]
SIGMA = 2.0
PKL = "results/grid_multi.pkl"


@dataclass
class Space:
    graph: object
    nodes: list
    neighbour: dict
    distances: np.ndarray
    densities: list
    nu: np.ndarray
    b: float


def build_space():
    """Build the grid graph, its mixture density nu, neighbour map and drift b."""
    graph = as_quantum_graph(nx.grid_2d_graph(10, 10), edge_length=1.0)
    graph.precomputing()
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    distances = np.array(
        [[nx.shortest_path_length(graph, u, v) for v in nodes] for u in nodes], float
    )
    neighbour = {node: next(iter(graph.neighbors(node))) for node in nodes}
    densities = [np.exp(-distances[:, index[m]] / SIGMA) for m in MODES]
    densities = [d / d.sum() for d in densities]
    nu = 0.5 * densities[0] + 0.5 * densities[1]
    for i, node in enumerate(nodes):
        graph.nodes[node]["weight"] = float(nu[i])
    b = 0.3 / critical_depth(potential_matrix(distances, nu), graph, nodes, index)
    return Space(graph, nodes, neighbour, distances, densities, nu, b)


def sample_data(seed, densities, n_nodes, n_data, n_obs):
    """Draw latent components and their node positions for one seed."""
    rng = np.random.default_rng(seed)
    comps = rng.integers(0, 2, n_data)
    data_nodes = np.array([rng.choice(n_nodes, p=densities[c]) for c in comps])
    if n_obs == n_data:
        obs_idx = data_nodes
    else:
        obs_idx = data_nodes[rng.integers(0, n_data, n_obs)]
    return comps, data_nodes, obs_idx


def run_seed(space, seed, comps, data_nodes, obs_idx, n_runs, track):
    """Multi-start SA plus baselines for one seed; return (methods, timings, conv)."""
    graph, nu = space.graph, space.nu
    for node, w in zip(space.nodes, nu):
        graph.nodes[node]["nb_obs"] = float(w)
    obs = [
        QGPoint(graph, (space.nodes[v], space.neighbour[space.nodes[v]]), 0)
        for v in obs_idx
    ]

    timings = {}
    t = time.perf_counter()
    labels, energies, convergence = [], [], None
    for r, centers, sa in annealings(
        lambda _rng: obs, 2, space.b, n_runs, "grid", seed, track_first=track
    ):
        if track and r == 0:
            convergence = {"time": sa.time_history, "energy": sa.energy_history}
        node_label = np.argmin(graph.node_center_distances(centers), axis=1)
        labels.append(node_label[data_nodes])
        energies.append(graph.node_energy(centers, weights=nu))
    timings["SA"] = time.perf_counter() - t
    raw = {"SA": (labels, np.array(energies))}

    # Baselines run on the nodes (weighted by nu), read out on the data. The node
    # distance matrix (space.distances) is the k-medoids input; it is built once
    # in build_space (see the store's setup timing).
    dist = space.distances
    t = time.perf_counter()
    lk, ek = B.weighted_kmedoids(
        dist, nu, 2, n_runs, method_entropy("grid", seed, "k-medoids")
    )
    timings["k-medoids"] = time.perf_counter() - t
    raw["k-medoids"] = ([lbl[data_nodes] for lbl in lk], ek)
    t = time.perf_counter()
    ls, _ = B.spectral_baseline(
        B.rbf_affinity(dist), 2, 20, method_entropy("grid", seed, "spectral")
    )
    timings["spectral"] = time.perf_counter() - t
    raw["spectral"] = ([lbl[data_nodes] for lbl in ls], None)
    return methods_from_raw(raw, comps), timings, convergence


def run(seeds=(42, 43, 44, 45, 46), n_runs=30, n_data=1000, n_obs=1000, n_jobs=1):
    t0 = time.perf_counter()
    space = build_space()  # one-time; builds the k-medoids node distance matrix
    build_s = time.perf_counter() - t0
    print(f"[grid] b={space.b:.4f}", flush=True)

    def fn(i, seed):
        comps, data_nodes, obs_idx = sample_data(
            seed, space.densities, len(space.nodes), n_data, n_obs
        )
        methods, timings, conv = run_seed(
            space, seed, comps, data_nodes, obs_idx, n_runs, track=(i == 0)
        )
        return {"methods": methods, "timings": timings}, conv

    per_seed, convergence = run_seeds(seeds, fn, n_jobs=n_jobs, tag="grid")

    store = {
        "name": "grid",
        "n_runs": n_runs,
        "seeds": list(seeds),
        "n_jobs": n_jobs,
        "setup": {"k-medoids matrix": build_s},  # one-time (fixed graph)
        "per_seed": per_seed,
        "convergence": convergence,
    }
    os.makedirs("results", exist_ok=True)
    pickle.dump(store, open(PKL, "wb"))
    print(f"saved {PKL}", flush=True)
    summarize(store)
    return store


if __name__ == "__main__":
    run()
