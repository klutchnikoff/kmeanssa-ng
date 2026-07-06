"""Stochastic block model experiment: a fresh weighted random graph per seed."""

import _env  # noqa: F401  -- pins BLAS threads; must precede numpy
import os
import pickle
import time
from dataclasses import dataclass

import numpy as np
import networkx as nx

from kmeanssa_ng import generate_random_sbm
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
import baselines as B
from calibration import potential_matrix, critical_depth
from multistart import (
    annealings,
    method_entropy,
    methods_from_raw,
    summarize,
    run_seeds,
)

PKL = "results/sbm_multi.pkl"


@dataclass
class Space:
    graph: object
    nodes: list
    true_labels: np.ndarray
    distances: np.ndarray
    nu: np.ndarray
    b: float
    dist_build_s: float = 0.0  # time to build the k-medoids distance matrix


def build_space(seed):
    """Generate one SBM graph with its distances, node measure nu and drift b."""
    graph = generate_random_sbm(
        sizes=[50, 50],
        p=[[0.8, 0.1], [0.1, 0.8]],
        weights=[1, 1],
        lengths=[[1, 4], [4, 1]],
        random_state=seed,
    )
    graph.precomputing()
    nodes = list(graph.nodes())
    index = {node: i for i, node in enumerate(nodes)}
    true_labels = np.array([graph.nodes[n].get("block", 0) for n in nodes])
    t = time.perf_counter()  # the node distance matrix is the k-medoids input
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
    distances = np.array([[lengths[u][v] for v in nodes] for u in nodes], float)
    dist_build_s = time.perf_counter() - t
    nu = np.ones(len(nodes)) / len(nodes)
    b = 0.3 / critical_depth(potential_matrix(distances, nu), graph, nodes, index)
    return Space(graph, nodes, true_labels, distances, nu, b, dist_build_s)


def run_seed(space, seed, n_runs, track):
    """Multi-start SA plus baselines for one seed; return (methods, timings, conv)."""
    graph = space.graph
    timings = {"k-medoids matrix": space.dist_build_s}  # built per seed in build_space
    t = time.perf_counter()
    labels, energies, convergence = [], [], None
    for r, centers, sa in annealings(
        lambda rng: graph.sample_points(
            100, strategy=UniformNodeSampling(random_state=rng)
        ),
        2,
        space.b,
        n_runs,
        "sbm",
        seed,
        track_first=track,
    ):
        if track and r == 0:
            convergence = {"time": sa.time_history, "energy": sa.energy_history}
        labels.append(np.argmin(graph.node_center_distances(centers), axis=1))
        energies.append(sa.calculate_energy(centers))
    timings["SA"] = time.perf_counter() - t
    raw = {"SA": (labels, np.array(energies))}

    t = time.perf_counter()
    lk, ek = B.weighted_kmedoids(
        space.distances, space.nu, 2, n_runs, method_entropy("sbm", seed, "k-medoids")
    )
    timings["k-medoids"] = time.perf_counter() - t
    raw["k-medoids"] = (lk, ek)
    t = time.perf_counter()
    adjacency = nx.to_numpy_array(graph, nodelist=space.nodes, weight=None)
    ls, _ = B.spectral_baseline(
        adjacency, 2, 20, method_entropy("sbm", seed, "spectral")
    )
    timings["spectral"] = time.perf_counter() - t
    raw["spectral"] = (ls, None)
    return methods_from_raw(raw, space.true_labels), timings, convergence


def run(seeds=(42, 43, 44, 45, 46), n_runs=50, n_jobs=1):
    def fn(i, seed):
        space = build_space(seed)
        methods, timings, conv = run_seed(space, seed, n_runs, track=(i == 0))
        return {"methods": methods, "timings": timings}, conv

    per_seed, convergence = run_seeds(seeds, fn, n_jobs=n_jobs, tag="sbm")

    store = {
        "name": "sbm",
        "n_runs": n_runs,
        "seeds": list(seeds),
        "n_jobs": n_jobs,
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
