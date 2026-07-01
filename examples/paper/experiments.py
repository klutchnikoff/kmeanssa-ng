"""Grid and SBM experiments, saving raw per-run data to pickles.

Every run is seeded through explicit numpy Generators, so the pickled ARIs,
energies and labels are reproducible across machines:

  results/grid_multi.pkl
  results/sbm_multi.pkl
"""

import os
import pickle

import numpy as np
import networkx as nx

from kmeanssa_ng import (
    as_quantum_graph,
    generate_random_sbm,
    SimulatedAnnealing,
    KMeansPlusPlus,
    MinimizeEnergy,
    QGPoint,
)
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling
from kmeanssa_ng.core.metrics import adjusted_rand_index

import baselines as B


# k=2 centroids on nodes: state-space potential used to calibrate the temperature.
def potential_matrix(dist_matrix, nu):
    """U[u1,u2] = sum_v nu[v] * min(d[u1,v]^2, d[u2,v]^2)."""
    d2 = dist_matrix**2
    n = len(nu)
    U = np.empty((n, n))
    for u1 in range(n):
        U[u1] = (nu * np.minimum(d2[u1][None, :], d2)).sum(axis=1)
    return U


def compute_cstar(U, qg, nodes, node_to_idx):
    """Critical depth c* via the minimum-spanning-tree heuristic over the k=2 state space."""
    n = len(nodes)
    S = nx.Graph()
    S.add_nodes_from(range(n * n))
    for u1 in range(n):
        for u2 in range(n):
            si = n * u1 + u2
            for w_node in qg.neighbors(nodes[u1]):
                w1 = node_to_idx[w_node]
                sj = n * w1 + u2
                if si < sj:
                    S.add_edge(si, sj, weight=max(U[u1, u2], U[w1, u2]))
            for w_node in qg.neighbors(nodes[u2]):
                w2 = node_to_idx[w_node]
                sj = n * u1 + w2
                if si < sj:
                    S.add_edge(si, sj, weight=max(U[u1, u2], U[u1, w2]))
    T = nx.minimum_spanning_tree(S)
    parent = list(range(n * n))
    comp_min = [U[i // n, i % n] for i in range(n * n)]

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    max_h = max(-U[u1, u2] for u1 in range(n) for u2 in range(n))
    for u, v, d in sorted(T.edges(data=True), key=lambda x: x[2]["weight"]):
        ru, rv = find(u), find(v)
        max_h = max(max_h, d["weight"] - comp_min[ru] - comp_min[rv])
        if ru != rv:
            parent[ru] = rv
            comp_min[rv] = min(comp_min[ru], comp_min[rv])
    return max_h + float(U.min())


def sa_runs(qg, nodes, sampler, n_obs, k, b, n_runs, seed, tracking=False):
    """Run the SA n_runs times, sampling fresh observations each run.

    Returns (labels_per_run, energies, tracking_history_or_None).
    """
    labs, ens = [], []
    history = None
    for r in range(n_runs):
        # Independent streams for the sampler and the annealer, from one run seed.
        sample_ss, sa_ss = np.random.SeedSequence(seed + r + 100).spawn(2)
        sampler.random_state = np.random.default_rng(sample_ss)
        pts = qg.sample_points(n_obs, strategy=sampler)
        sa = SimulatedAnnealing(
            observations=pts,
            k=k,
            lambda0=1.0,
            beta0=b,
            step_size=0.01,
            energy_mode="obs",
            random_state=np.random.default_rng(sa_ss),
        )
        track = tracking and r == 0
        centers = sa.run(
            KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1, record_energy=track
        )
        if track:
            history = sa.energy_history
        labs.append(np.argmin(qg.node_center_distances(centers), axis=1))
        ens.append(sa.calculate_energy(centers))
    return labs, np.array(ens), history


def _methods_dict(raw, true_labels):
    out = {}
    for name, (labs, ens) in raw.items():
        aris = np.array([adjusted_rand_index(lbl, true_labels) for lbl in labs])
        out[name] = {
            "aris": aris,
            "energies": (np.asarray(ens) if ens is not None else None),
            "labels": [np.asarray(lbl) for lbl in labs],
        }
    return out


def run_grid_multi(
    seeds=(42, 43, 44, 45, 46), n_runs=30, n_data=1000, n_obs=1000, sigma=2.0
):
    """Grid 10x10 with a two-component generative density; ARI at the data level."""
    G = nx.grid_2d_graph(10, 10)
    qg = as_quantum_graph(G, edge_length=1.0)
    qg.precomputing()
    qg._node_position = {node: node for node in qg.nodes()}
    nodes = list(qg.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}
    N = len(nodes)
    dist = np.array(
        [[nx.shortest_path_length(qg, u, v) for v in nodes] for u in nodes], float
    )
    nbr = {nd: next(iter(qg.neighbors(nd))) for nd in qg.nodes()}
    v1, v2 = (1, 1), (8, 8)
    # Mixture density nu = 1/2 (f_1 + f_2), each f_c(v) ~ exp(-d(v, mu_c)/sigma).
    f = [np.exp(-dist[:, idx[m]] / sigma) for m in (v1, v2)]
    f = [fi / fi.sum() for fi in f]
    nu = 0.5 * f[0] + 0.5 * f[1]
    for i, nd in enumerate(nodes):
        qg.nodes[nd]["weight"] = float(nu[i])
    nu_row = nu  # nu is already in nodes() order
    U = potential_matrix(dist, nu)
    min_U = float(U.min())
    c_star = compute_cstar(U, qg, nodes, idx)
    b = 0.3 / c_star
    print(f"[grid] c*={c_star:.3f} b*={1 / c_star:.4f} b={b:.4f}", flush=True)

    per_seed = {}
    for i, sd in enumerate(seeds):
        rng = np.random.default_rng(sd)
        comps = rng.integers(0, 2, n_data)  # latent component (ground truth)
        data_nodes = np.array([rng.choice(N, p=f[c]) for c in comps])
        obs_idx = (
            data_nodes
            if n_obs == n_data
            else data_nodes[rng.integers(0, n_data, n_obs)]
        )
        obs = [QGPoint(qg, (nodes[v], nbr[nodes[v]]), 0) for v in obs_idx]

        # SA runs: energy computed exactly, data labels via projection of node clusters.
        labs, ens, hist = [], [], None
        for r in range(n_runs):
            sa = SimulatedAnnealing(
                observations=obs,
                k=2,
                lambda0=1.0,
                beta0=b,
                step_size=0.01,
                energy_mode="obs",
                random_state=np.random.default_rng(
                    np.random.SeedSequence(sd + r + 100)
                ),
            )
            track = i == 0 and r == 0
            centers = sa.run(
                KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1, record_energy=track
            )
            if track:
                hist = sa.energy_history
            node_lab = np.argmin(qg.node_center_distances(centers), axis=1)
            labs.append(node_lab[data_nodes])
            ens.append(qg.node_energy(centers, weights=nu_row))
        raw = {"SA": (labs, np.array(ens))}

        # Baselines on nodes (weighted by nu), evaluated on the data via data_nodes.
        lk, ek = B.weighted_kmedoids(dist, nu, 2, n_runs, sd + 1000)
        raw["k-medoids"] = ([lbl[data_nodes] for lbl in lk], ek)
        ls, _ = B.spectral_baseline(B.rbf_affinity(dist), 2, 20, sd + 1000)
        raw["spectral"] = ([lbl[data_nodes] for lbl in ls], None)

        per_seed[sd] = {
            "methods": _methods_dict(raw, comps),
            "energy_history": hist,
            "comps": comps,
            "data_nodes": data_nodes,
        }
        print(f"[grid] seed {sd} done", flush=True)

    store = {
        "name": "grid",
        "nodes": nodes,
        "positions": dict(qg._node_position),
        "nu": nu,
        "edges": list(qg.edges()),
        "min_U": min_U,
        "c_star": c_star,
        "b": b,
        "k": 2,
        "n_runs": n_runs,
        "n_data": n_data,
        "n_obs": n_obs,
        "sigma": sigma,
        "seeds": list(seeds),
        "per_seed": per_seed,
    }
    os.makedirs("results", exist_ok=True)
    pickle.dump(store, open("results/grid_multi.pkl", "wb"))
    print("saved results/grid_multi.pkl", flush=True)
    return store


def run_sbm_multi(seeds=(42, 43, 44, 45, 46), n_runs=50):
    """Stochastic block model with a fresh random graph per seed."""
    per_seed = {}
    fig_data = None
    for i, sd in enumerate(seeds):
        qg = generate_random_sbm(
            sizes=[50, 50],
            p=[[0.8, 0.1], [0.1, 0.8]],
            weights=[1, 1],
            lengths=[[1, 4], [4, 1]],
            random_state=sd,
        )
        qg.precomputing()
        nodes = list(qg.nodes())
        idx = {nd: j for j, nd in enumerate(nodes)}
        N = len(nodes)
        true_labels = np.array([qg.nodes[n].get("block", 0) for n in nodes])
        dmap = dict(nx.all_pairs_dijkstra_path_length(qg, weight="length"))
        dist = np.array([[dmap[u][v] for v in nodes] for u in nodes], float)
        nu = np.ones(N) / N
        U = potential_matrix(dist, nu)
        min_U = float(U.min())
        c_star = compute_cstar(U, qg, nodes, idx)
        b = 0.3 / c_star

        labs, ens, hist = sa_runs(
            qg, nodes, UniformNodeSampling(), 100, 2, b, n_runs, sd, tracking=(i == 0)
        )
        raw = {"SA": (labs, ens)}
        lk, ek = B.weighted_kmedoids(dist, nu, 2, n_runs, sd + 2000)
        raw["k-medoids"] = (lk, ek)
        adj = nx.to_numpy_array(qg, nodelist=nodes, weight=None)
        ls, _ = B.spectral_baseline(adj, 2, 20, sd + 2000)
        raw["spectral"] = (ls, None)
        per_seed[sd] = {
            "methods": _methods_dict(raw, true_labels),
            "energy_history": hist,
            "min_U": min_U,
            "c_star": c_star,
        }
        if i == 0:
            fig_data = {
                "positions": nx.spring_layout(qg, seed=sd),
                "edges": list(qg.edges()),
                "true_labels": true_labels,
                "nodes": nodes,
            }
        print(f"[sbm] seed {sd} c*={c_star:.3f} done", flush=True)

    store = {
        "name": "sbm",
        "k": 2,
        "n_runs": n_runs,
        "seeds": list(seeds),
        "per_seed": per_seed,
        "fig_data": fig_data,
    }
    os.makedirs("results", exist_ok=True)
    pickle.dump(store, open("results/sbm_multi.pkl", "wb"))
    print("saved results/sbm_multi.pkl", flush=True)
    return store


if __name__ == "__main__":
    run_grid_multi()
    run_sbm_multi()
