"""Stochastic block model experiment: a fresh weighted random graph per seed."""

import os
import pickle

import numpy as np
import networkx as nx

from kmeanssa_ng import generate_random_sbm
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
import baselines as B
from calibration import potential_matrix, critical_depth
from multistart import annealings, methods_from_raw

PKL = "results/sbm_multi.pkl"


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
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight="length"))
    distances = np.array([[lengths[u][v] for v in nodes] for u in nodes], float)
    nu = np.ones(len(nodes)) / len(nodes)
    U = potential_matrix(distances, nu)
    b = 0.3 / critical_depth(U, graph, nodes, index)
    return graph, nodes, true_labels, distances, nu, b


def run_seed(graph, nodes, true_labels, distances, nu, b, seed, n_runs, track):
    """Multi-start SA plus k-medoids and spectral baselines for one seed."""
    labels, energies, history = [], [], None
    for r, centers, sa in annealings(
        lambda rng: graph.sample_points(
            100, strategy=UniformNodeSampling(random_state=rng)
        ),
        2,
        b,
        n_runs,
        seed + 100,
        track_first=track,
    ):
        if track and r == 0:
            history = sa.energy_history
        labels.append(np.argmin(graph.node_center_distances(centers), axis=1))
        energies.append(sa.calculate_energy(centers))
    raw = {"SA": (labels, np.array(energies))}

    lk, ek = B.weighted_kmedoids(distances, nu, 2, n_runs, seed + 2000)
    raw["k-medoids"] = (lk, ek)
    adjacency = nx.to_numpy_array(graph, nodelist=nodes, weight=None)
    ls, _ = B.spectral_baseline(adjacency, 2, 20, seed + 2000)
    raw["spectral"] = (ls, None)
    return methods_from_raw(raw, true_labels), history


def run(seeds=(42, 43, 44, 45, 46), n_runs=50):
    per_seed = {}
    fig_data = None
    for i, seed in enumerate(seeds):
        graph, nodes, true_labels, distances, nu, b = build_space(seed)
        methods, history = run_seed(
            graph, nodes, true_labels, distances, nu, b, seed, n_runs, track=(i == 0)
        )
        per_seed[seed] = {"methods": methods, "energy_history": history}
        if i == 0:
            fig_data = {
                "positions": nx.spring_layout(graph, seed=seed),
                "edges": list(graph.edges()),
                "true_labels": true_labels,
                "nodes": nodes,
            }
        print(f"[sbm] seed {seed} done", flush=True)

    store = {
        "name": "sbm",
        "n_runs": n_runs,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "fig_data": fig_data,
    }
    os.makedirs("results", exist_ok=True)
    pickle.dump(store, open(PKL, "wb"))
    print(f"saved {PKL}", flush=True)
    return store


if __name__ == "__main__":
    run()
