"""Grid 10x10 experiment: a two-component generative density; ARI at the data level."""

import os
import pickle

import numpy as np
import networkx as nx

from kmeanssa_ng import as_quantum_graph, QGPoint
import baselines as B
from calibration import potential_matrix, critical_depth
from multistart import annealings, methods_from_raw

MODES = [(1, 1), (8, 8)]
SIGMA = 2.0
PKL = "results/grid_multi.pkl"


def build_space():
    """Build the grid graph, its mixture density nu, neighbour map and drift b."""
    graph = as_quantum_graph(nx.grid_2d_graph(10, 10), edge_length=1.0)
    graph.precomputing()
    graph._node_position = {node: node for node in graph.nodes()}
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
    U = potential_matrix(distances, nu)
    return {
        "graph": graph,
        "nodes": nodes,
        "neighbour": neighbour,
        "distances": distances,
        "densities": densities,
        "nu": nu,
        "b": 0.3 / critical_depth(U, graph, nodes, index),
        "min_U": float(U.min()),
    }


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
    """Multi-start SA plus k-medoids and spectral baselines for one seed."""
    graph, nodes, nu = space["graph"], space["nodes"], space["nu"]
    obs = [QGPoint(graph, (nodes[v], space["neighbour"][nodes[v]]), 0) for v in obs_idx]

    labels, energies, history = [], [], None
    for r, centers, sa in annealings(
        lambda _rng: obs, 2, space["b"], n_runs, seed + 100, track_first=track
    ):
        if track and r == 0:
            history = sa.energy_history
        node_label = np.argmin(graph.node_center_distances(centers), axis=1)
        labels.append(node_label[data_nodes])
        energies.append(graph.node_energy(centers, weights=nu))
    raw = {"SA": (labels, np.array(energies))}

    # Baselines run on the nodes (weighted by nu), read out on the data.
    dist = space["distances"]
    lk, ek = B.weighted_kmedoids(dist, nu, 2, n_runs, seed + 1000)
    raw["k-medoids"] = ([lbl[data_nodes] for lbl in lk], ek)
    ls, _ = B.spectral_baseline(B.rbf_affinity(dist), 2, 20, seed + 1000)
    raw["spectral"] = ([lbl[data_nodes] for lbl in ls], None)
    return methods_from_raw(raw, comps), history


def run(seeds=(42, 43, 44, 45, 46), n_runs=30, n_data=1000, n_obs=1000):
    space = build_space()
    print(f"[grid] b={space['b']:.4f}", flush=True)
    per_seed = {}
    for i, seed in enumerate(seeds):
        comps, data_nodes, obs_idx = sample_data(
            seed, space["densities"], len(space["nodes"]), n_data, n_obs
        )
        methods, history = run_seed(
            space, seed, comps, data_nodes, obs_idx, n_runs, track=(i == 0)
        )
        per_seed[seed] = {
            "methods": methods,
            "energy_history": history,
            "comps": comps,
            "data_nodes": data_nodes,
        }
        print(f"[grid] seed {seed} done", flush=True)

    store = {
        "name": "grid",
        "nodes": space["nodes"],
        "positions": dict(space["graph"]._node_position),
        "nu": space["nu"],
        "edges": list(space["graph"].edges()),
        "min_U": space["min_U"],
        "b": space["b"],
        "n_runs": n_runs,
        "n_data": n_data,
        "n_obs": n_obs,
        "seeds": list(seeds),
        "per_seed": per_seed,
    }
    os.makedirs("results", exist_ok=True)
    pickle.dump(store, open(PKL, "wb"))
    print(f"saved {PKL}", flush=True)
    return store


if __name__ == "__main__":
    run()
