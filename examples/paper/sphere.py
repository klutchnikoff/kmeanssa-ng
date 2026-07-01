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
    SimulatedAnnealing,
    KMeansPlusPlus,
    MinimizeEnergy,
    create_sphere,
    RiemannianPoint,
    FibonacciNet,
    build_epsilon_net_graph,
    estimate_covering_radius,
)
from kmeanssa_ng.core.metrics import compute_labels, adjusted_rand_index

import baselines as B

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


def eval_seed(
    qg, V, nbr, node_list, seed, n_data=2000, n_obs=2000, b=0.2, n_runs=20, diag=False
):
    """Run all four methods for one seed; return raw per-run data at the data level."""
    data, dtrue = make_data(seed, n_data)
    proj = np.argmin(np.arccos(np.clip(data @ V.T, -1, 1)), axis=1)
    n_net = len(V)
    nu = np.bincount(proj, minlength=n_net).astype(float)
    nu /= nu.sum()
    nu_row = nu[node_list]
    lbn = np.empty(n_net, dtype=int)
    raw = {}

    # SA on the graph (shortest-path distances, no exponential map).
    labs, ens, sizes, cents = [], [], [], []
    for r in range(n_runs):
        idx_ss, sa_ss = np.random.SeedSequence(seed + r + 300).spawn(2)
        on = proj[np.random.default_rng(idx_ss).integers(0, n_data, size=n_obs)]
        obs = [QGPoint(qg, (int(v), nbr[int(v)]), 0) for v in on]
        sa = SimulatedAnnealing(
            observations=obs,
            k=3,
            lambda0=1.0,
            beta0=b,
            step_size=0.01,
            energy_mode="obs",
            random_state=np.random.default_rng(sa_ss),
        )
        centers = sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1)
        lbn[node_list] = np.argmin(qg.node_center_distances(centers), axis=1)
        dl = lbn[proj].copy()
        labs.append(dl)
        ens.append(qg.node_energy(centers, weights=nu_row))
        sizes.append(tuple(int(x) for x in np.bincount(dl, minlength=3)))
        cents.append(np.array([V[c.closest_node()] for c in centers]))
    raw["SA-graph"] = (labs, np.array(ens))
    if diag:
        order = np.argsort(ens)
        print(
            f"  [diag seed {seed}] SA-graph runs by energy (selected = first):",
            flush=True,
        )
        for o in order:
            print(
                f"    E={ens[o]:.3f}  ARI={adjusted_rand_index(labs[o], dtrue):.3f}  "
                f"sizes={sizes[o]}",
                flush=True,
            )

    # SA directly on the sphere (great-circle geodesics, exponential map).
    sp = create_sphere(2)
    allp = [RiemannianPoint(sp, x) for x in data]
    labs, ens = [], []
    for r in range(n_runs):
        idx_ss, sa_ss = np.random.SeedSequence(seed + r + 300).spawn(2)
        idx = np.random.default_rng(idx_ss).integers(0, n_data, size=n_obs)
        obs = [RiemannianPoint(sp, data[i]) for i in idx]
        sa = SimulatedAnnealing(
            observations=obs,
            k=3,
            lambda0=1.0,
            beta0=b,
            step_size=0.01,
            energy_mode="obs",
            random_state=np.random.default_rng(sa_ss),
        )
        centers = sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1)
        labs.append(np.array(compute_labels(sp, allp, centers)))
        ens.append(sa.calculate_energy(centers))
    raw["SA-sphere"] = (labs, np.array(ens))

    # CLVQ and k-medoids on the exact geodesic distances of the data.
    labc, enc = B.clvq_sphere(data, k=3, n_runs=n_runs, base_seed=seed + 300)
    raw["CLVQ"] = (labc, np.array(enc))
    Dd = np.arccos(np.clip(data @ data.T, -1, 1))
    labm, enm = B.weighted_kmedoids(
        Dd, np.ones(n_data) / n_data, k=3, n_runs=n_runs, base_seed=seed + 300
    )
    raw["k-medoids"] = (labm, np.array(enm))

    methods = {}
    for name, (labs, ens) in raw.items():
        aris = np.array([adjusted_rand_index(lbl, dtrue) for lbl in labs])
        methods[name] = {
            "aris": aris,
            "energies": ens,
            "labels": [np.asarray(lbl) for lbl in labs],
        }
    methods["SA-graph"]["centroids"] = cents
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


def run_multi(
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
    run_multi()
