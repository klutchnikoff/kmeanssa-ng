"""Rate-theorem illustration on a 3-node path toy A--B--C, k=2.

The three benchmark setups have a large critical depth c*, so the polynomial
convergence rate of the annealing is invisible on any feasible horizon. Here we
build a deliberately shallow landscape (small c*) on which the rate

    P(U(X_t) - U* > eta) = O(t^{-rho}),   rho < min(b*eta, (1 - b*c*) / 2),

is directly observable: the log-log slope of the empirical exceedance matches
the predicted exponent. A--B is short (l_ab), B--C long (l_bc); nu is uniform on
the three nodes, so the "uniform" energy mode records the true potential U(X_t).

``run()`` performs ``n_runs`` independent annealings, interpolates each energy
trajectory onto a common annealing-time grid, and caches the ensemble to
``results/rate_toy.pkl``; ``make_figures.figure_rate()`` turns it into figure_5.
"""

import os
import pickle

import numpy as np
import networkx as nx

from kmeanssa_ng import (
    as_quantum_graph,
    QGPoint,
    SimulatedAnnealing,
    KMeansPlusPlus,
    MinimizeEnergy,
)
from kmeanssa_ng.quantum_graph.generators import UniformDistribution
from calibration import potential_matrix, critical_depth
from multistart import code_stamp, method_entropy

PKL = "results/rate_toy.pkl"


def build_toy(l_ab=1.0, l_bc=2.0):
    """3-node path A--B--C with the given edge lengths; nu uniform on the nodes."""
    g = nx.Graph()
    g.add_edge("A", "B")
    g.add_edge("B", "C")
    qg = as_quantum_graph(g, edge_length=1.0)
    for (u, v), length in [(("A", "B"), l_ab), (("B", "C"), l_bc)]:
        qg[u][v]["length"] = float(length)
        qg[u][v]["distribution"] = UniformDistribution(float(length))
    qg.precomputing()
    nodes = list(qg.nodes())
    index = {n: i for i, n in enumerate(nodes)}
    lengths = dict(nx.all_pairs_dijkstra_path_length(qg, weight="length"))
    distances = np.array([[lengths[u][v] for v in nodes] for u in nodes], float)
    nu = np.ones(len(nodes)) / len(nodes)
    for i, n in enumerate(nodes):
        qg.nodes[n]["weight"] = float(nu[i])
    return qg, nodes, index, distances, nu


def true_min_energy(l_ab=1.0, l_bc=2.0, grid=4000):
    """Global minimum of U over continuous center positions on the path.

    The graph is a path, so arc-length ``s`` in ``[0, l_ab + l_bc]`` parametrizes
    every point and the geodesic distance is ``|s - s'|``; the nodes sit at 0,
    ``l_ab`` and ``l_ab + l_bc``. We minimize over all pairs of positions.
    """
    s_nodes = np.array([0.0, l_ab, l_ab + l_bc])
    s = np.linspace(0.0, l_ab + l_bc, grid)
    d2 = (s[:, None] - s_nodes[None, :]) ** 2
    return min(np.minimum(d2[i][None, :], d2).mean(axis=1).min() for i in range(grid))


def run(n_runs=300, n_obs=100000, b=0.3, l_ab=1.0, l_bc=2.0, seed=7, n_grid=800):
    """Multi-start annealing on the toy; cache interpolated energy trajectories.

    Defaults are the light pipeline setting (~15 min). The paper figure uses
    ``n_runs=700, n_obs=250000`` (~70 min), which is what ``reproduce.py``
    runs without ``--quick``.
    """
    qg, nodes, index, distances, nu = build_toy(l_ab, l_bc)
    cstar = critical_depth(potential_matrix(distances, nu), qg, nodes, index)
    ustar = true_min_energy(l_ab, l_bc)
    neigh = {n: next(iter(qg.neighbors(n))) for n in nodes}
    grid = np.linspace(0.0, 0.95 * np.sqrt(n_obs), n_grid)  # t_max ~ sqrt(n_obs)
    stacked = np.empty((n_runs, n_grid))
    run_entropy = method_entropy("rate", seed).spawn(n_runs)
    for r in range(n_runs):
        obs_seed, sa_seed = run_entropy[r].spawn(2)
        idx = np.random.default_rng(obs_seed).integers(0, len(nodes), n_obs)
        obs = [QGPoint(qg, (nodes[i], neigh[nodes[i]]), 0) for i in idx]
        sa = SimulatedAnnealing(
            observations=obs,
            k=2,
            lambda0=1.0,
            beta0=b,
            step_size=0.01,
            energy_mode="uniform",  # nu uniform -> records the true potential U
            random_state=np.random.default_rng(sa_seed),
        )
        sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.0, record_energy=True)
        stacked[r] = np.interp(grid, sa.time_history, sa.energy_history)
        if (r + 1) % 50 == 0:
            print(f"[rate] {r + 1}/{n_runs} runs", flush=True)

    store = {
        "name": "rate",
        "code": code_stamp(),
        "grid": grid,
        "stacked": stacked,
        "ustar": ustar,
        "cstar": cstar,
        "b": b,
        "l_ab": l_ab,
        "l_bc": l_bc,
        "n_obs": n_obs,
        "n_runs": n_runs,
    }
    os.makedirs("results", exist_ok=True)
    pickle.dump(store, open(PKL, "wb"))
    print(f"saved {PKL}", flush=True)
    summarize(store)
    return store


def fitted_slopes(store, mults=(1, 2, 4, 6)):
    """Log-log slope of the exceedance over the last decade, per eta = mult*U*."""
    grid, stacked = store["grid"], store["stacked"]
    ustar, cstar, b = store["ustar"], store["cstar"], store["b"]
    excess = stacked - ustar
    win = grid > grid[-1] / 10
    out = {}
    for m in mults:
        eta = m * ustar
        p = (excess > eta).mean(axis=0)
        ok = win & (p > 0)
        slope = (
            np.polyfit(np.log(grid[ok]), np.log(p[ok]), 1)[0]
            if ok.sum() > 5
            else np.nan
        )
        out[m] = (-slope, min(b * eta, 0.5 * (1 - b * cstar)))
    return out


def summarize(store):
    """Print measured vs predicted rate exponents."""
    print(
        f"{store['name']}: c*={store['cstar']:.3f}  U*={store['ustar']:.4f}  "
        f"b={store['b']} (b*={1 / store['cstar']:.3f})  "
        f"{store['n_runs']} runs x {store['n_obs']} obs"
    )
    print(f"{'eta':>10} {'rho_hat':>9} {'rho_pred':>10}")
    for m, (rho_hat, pred) in fitted_slopes(store).items():
        print(
            f"{m}*U*={m * store['ustar']:>5.3f} {rho_hat:>9.3f} {'<' + f'{pred:.3f}':>10}"
        )


if __name__ == "__main__":
    run()
