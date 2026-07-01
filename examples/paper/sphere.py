"""Sphere S^2 experiment on an epsilon-net graph, saving raw data.

Four methods are compared on von Mises-Fisher data: SA on the approximating
graph, SA directly on the manifold, CLVQ and k-medoids. All per-run data is
pickled to results/sphere_multi.pkl so tables and figures rebuild without the
~20 min recomputation. The net and its distances are independent of the seed,
so the graph is built once and reused across seeds.
"""

import os
import time
import pickle

import numpy as np

from kmeanssa_ng import (
    QGPoint,
    create_sphere,
    RiemannianPoint,
    FibonacciNet,
    build_epsilon_net_graph,
    estimate_covering_radius,
)
from kmeanssa_ng.core.metrics import compute_labels, adjusted_rand_index

import baselines as B
from multistart import annealings, methods_from_raw

MODES = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
KAPPA = 10.0
PKL = "results/sphere_multi.pkl"


def sample_vmf(mu, kappa, n_samples, rng):
    """Sample from a von Mises-Fisher distribution on S^2 using ``rng``.

    geomstats' vMF sampler is only seedable through the global numpy RNG, so a
    small reproducible sampler is kept here to stay Generator-driven.
    """
    mu = np.array(mu, dtype=float)
    mu = mu / np.linalg.norm(mu)
    u = rng.uniform(0, 1, n_samples)
    z = 1.0 + np.log(u + (1.0 - u) * np.exp(-2.0 * kappa)) / kappa
    phi = rng.uniform(0, 2.0 * np.pi, n_samples)
    sin_theta = np.sqrt(1.0 - z**2)
    canonical = np.stack([sin_theta * np.cos(phi), sin_theta * np.sin(phi), z], axis=1)
    # Rotate the north-pole frame onto mu.
    if abs(mu[2]) > 0.9999:
        rotation = np.eye(3) if mu[2] > 0 else -np.eye(3)
    else:
        u3 = mu
        u1 = np.array([-mu[1], mu[0], 0.0])
        u1 /= np.linalg.norm(u1)
        u2 = np.cross(u3, u1)
        rotation = np.column_stack((u1, u2, u3))
    return canonical @ rotation.T


def build_graph(n_net):
    """Fibonacci epsilon-net on S^2, connected within l(eps) = sqrt(eps)."""
    sphere = create_sphere(2)
    V = FibonacciNet().build(sphere, n_net)
    eps = estimate_covering_radius(sphere, V, random_state=0)
    t = time.time()
    qg = build_epsilon_net_graph(sphere, V, ell=float(np.sqrt(eps)))
    print(
        f"graph n_net={n_net} eps={eps:.4f} l_eps={np.sqrt(eps):.4f} "
        f"edges={qg.number_of_edges()} build={time.time() - t:.0f}s",
        flush=True,
    )
    nbr = {v: next(iter(qg.neighbors(v))) for v in qg.nodes}
    node_list = np.array(list(qg.nodes()))
    return qg, V, nbr, node_list, eps, float(np.sqrt(eps))


def make_data(seed, n_data):
    """Sample n_data vMF points over the three modes; return (points, labels)."""
    rng = np.random.default_rng(seed)
    comps = rng.integers(3, size=n_data)
    data = np.empty((n_data, 3))
    for c in range(3):
        mask = comps == c
        count = int(mask.sum())
        if count:
            data[mask] = sample_vmf(MODES[c], KAPPA, count, rng)
    return data, comps


def _sa_graph_runs(qg, V, nbr, node_list, nu_row, proj, n_data, n_obs, b, n_runs, seed):
    """SA on the approximating graph; return (data_labels, energies, centroids)."""

    def observations_for(rng):
        on = proj[rng.integers(0, n_data, size=n_obs)]
        return [QGPoint(qg, (int(v), nbr[int(v)]), 0) for v in on]

    node_label = np.empty(len(V), dtype=int)
    labels, energies, centroids = [], [], []
    for _, centers, sa in annealings(observations_for, 3, b, n_runs, seed + 300):
        node_label[node_list] = np.argmin(qg.node_center_distances(centers), axis=1)
        labels.append(node_label[proj].copy())
        energies.append(qg.node_energy(centers, weights=nu_row))
        centroids.append(np.array([V[c.closest_node()] for c in centers]))
    return labels, np.array(energies), centroids


def _sa_sphere_runs(data, n_data, n_obs, b, n_runs, seed):
    """SA directly on S^2 (great-circle geodesics); return (data_labels, energies)."""
    sphere = create_sphere(2)
    all_points = [RiemannianPoint(sphere, x) for x in data]

    def observations_for(rng):
        idx = rng.integers(0, n_data, size=n_obs)
        return [RiemannianPoint(sphere, data[i]) for i in idx]

    labels, energies = [], []
    for _, centers, sa in annealings(observations_for, 3, b, n_runs, seed + 300):
        labels.append(np.array(compute_labels(sphere, all_points, centers)))
        energies.append(sa.calculate_energy(centers))
    return labels, np.array(energies)


def _print_diag(seed, labels, energies, dtrue):
    """Print SA-graph runs sorted by energy (the first is the selected one)."""
    print(f"  [diag seed {seed}] SA-graph runs by energy:", flush=True)
    for o in np.argsort(energies):
        sizes = tuple(int(x) for x in np.bincount(labels[o], minlength=3))
        ari = adjusted_rand_index(labels[o], dtrue)
        print(f"    E={energies[o]:.3f}  ARI={ari:.3f}  sizes={sizes}", flush=True)


def eval_seed(
    qg, V, nbr, node_list, seed, n_data=2000, n_obs=2000, b=0.2, n_runs=20, diag=False
):
    """Run all four methods for one seed; return raw per-run data at the data level."""
    data, dtrue = make_data(seed, n_data)
    proj = np.argmin(np.arccos(np.clip(data @ V.T, -1, 1)), axis=1)
    nu = np.bincount(proj, minlength=len(V)).astype(float)
    nu /= nu.sum()
    nu_row = nu[node_list]

    graph_labels, graph_energies, centroids = _sa_graph_runs(
        qg, V, nbr, node_list, nu_row, proj, n_data, n_obs, b, n_runs, seed
    )
    if diag:
        _print_diag(seed, graph_labels, graph_energies, dtrue)
    sphere_labels, sphere_energies = _sa_sphere_runs(
        data, n_data, n_obs, b, n_runs, seed
    )

    raw = {
        "SA-graph": (graph_labels, graph_energies),
        "SA-sphere": (sphere_labels, sphere_energies),
    }
    labc, enc = B.clvq_sphere(data, k=3, n_runs=n_runs, base_seed=seed + 300)
    raw["CLVQ"] = (labc, np.array(enc))
    geodesic = np.arccos(np.clip(data @ data.T, -1, 1))
    labm, enm = B.weighted_kmedoids(
        geodesic, np.ones(n_data) / n_data, k=3, n_runs=n_runs, base_seed=seed + 300
    )
    raw["k-medoids"] = (labm, np.array(enm))

    methods = methods_from_raw(raw, dtrue)
    methods["SA-graph"]["centroids"] = centroids
    return {"data": data, "dtrue": dtrue, "proj": proj, "methods": methods}


def agg(aris, energies):
    aris = np.asarray(aris)
    sel = float(aris[int(np.argmin(energies))])
    return float(aris.mean()), float(aris.std()), float(aris.max()), sel


def summarize(store):
    """Print per-seed and aggregated ARIs from a saved/loaded store."""
    names = list(next(iter(store["per_seed"].values()))["methods"].keys())
    agg_by = {n: [] for n in names}
    for sd, sres in store["per_seed"].items():
        line = []
        for n in names:
            m = sres["methods"][n]
            a = agg(m["aris"], m["energies"])
            agg_by[n].append(a)
            line.append(f"{n} sel={a[3]:.3f} best={a[2]:.3f}")
        print(f"seed {sd}: " + "  ".join(line))
    print("\n=== aggregated over seeds (mean of per-seed values) ===")
    print(f"{'method':12s}  mean   best   selected  (std sel)")
    for n in names:
        rows = np.array(agg_by[n])
        print(
            f"{n:12s}  {rows[:, 0].mean():.3f}  {rows[:, 2].mean():.3f}  "
            f"{rows[:, 3].mean():.3f}     ({rows[:, 3].std():.3f})"
        )


def run(
    seeds=(42, 43, 44, 45, 46), n_net=5000, n_data=2000, n_obs=2000, b=0.2, n_runs=20
):
    qg, V, nbr, node_list, eps, l_eps = build_graph(n_net)
    per_seed = {}
    for i, sd in enumerate(seeds):
        per_seed[sd] = eval_seed(
            qg, V, nbr, node_list, sd, n_data, n_obs, b, n_runs, diag=(i == 0)
        )
        print(f"seed {sd} done", flush=True)
    store = {
        "config": {
            "n_net": n_net,
            "n_data": n_data,
            "n_obs": n_obs,
            "b": b,
            "n_runs": n_runs,
            "seeds": list(seeds),
            "kappa": KAPPA,
            "modes": MODES,
        },
        "V": V,
        "eps": eps,
        "l_eps": l_eps,
        "per_seed": per_seed,
    }
    os.makedirs("results", exist_ok=True)
    with open(PKL, "wb") as f:
        pickle.dump(store, f)
    print(f"\nsaved raw data to {PKL}\n", flush=True)
    summarize(store)
    return store


if __name__ == "__main__":
    run()
