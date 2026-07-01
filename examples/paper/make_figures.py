"""Rebuild the three partition figures from saved data and reproducible re-runs.

Sphere: plotted directly from results/sphere_multi.pkl. Grid and SBM: the
lowest-energy SA run of seed 42 is regenerated (same seeding as experiments.py),
giving the node labels and centroids of the partition plot.

  figures/figure_{1,2,3}.{pdf,png}
"""

import os
import pickle

import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kmeanssa_ng import (  # noqa: E402
    as_quantum_graph,
    generate_random_sbm,
    QGPoint,
    SimulatedAnnealing,
    KMeansPlusPlus,
    MinimizeEnergy,
)
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling  # noqa: E402

import experiments as EXP  # noqa: E402

os.makedirs("figures", exist_ok=True)
SEED = 42
CMAP = plt.cm.coolwarm


def _sa_selected(qg, nodes, obs, k, b, n_runs, seed, nu_row, N):
    """Return (node_labels, centroids, energy) of the lowest-energy run over n_runs."""
    best = None
    for r in range(n_runs):
        sa_rng = np.random.default_rng(np.random.SeedSequence(seed + r + 100))
        sa = SimulatedAnnealing(
            observations=obs,
            k=k,
            lambda0=1.0,
            beta0=b,
            step_size=0.01,
            energy_mode="obs",
            random_state=sa_rng,
        )
        centers = sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1)
        lab = np.argmin(qg.node_center_distances(centers), axis=1)
        e = qg.node_energy(centers, weights=nu_row)
        if best is None or e < best[2]:
            best = (lab, [c.closest_node() for c in centers], e)
    return best


def figure_grid():
    G = nx.grid_2d_graph(10, 10)
    qg = as_quantum_graph(G, edge_length=1.0)
    qg.precomputing()
    pos = {nd: nd for nd in qg.nodes()}
    nodes = list(qg.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}
    N = len(nodes)
    dist = np.array(
        [[nx.shortest_path_length(qg, u, v) for v in nodes] for u in nodes], float
    )
    nbr = {nd: next(iter(qg.neighbors(nd))) for nd in qg.nodes()}
    f = [np.exp(-dist[:, idx[m]] / 2.0) for m in ((1, 1), (8, 8))]
    f = [fi / fi.sum() for fi in f]
    nu = 0.5 * f[0] + 0.5 * f[1]
    for i, nd in enumerate(nodes):
        qg.nodes[nd]["weight"] = float(nu[i])
    U = EXP.potential_matrix(dist, nu)
    b = 0.3 / EXP.compute_cstar(U, qg, nodes, idx)
    rng = np.random.default_rng(SEED)
    comps = rng.integers(0, 2, 1000)
    data_nodes = np.array([rng.choice(N, p=f[c]) for c in comps])
    obs = [QGPoint(qg, (nodes[v], nbr[nodes[v]]), 0) for v in data_nodes]
    lab, cent, _ = _sa_selected(qg, nodes, obs, 2, b, 30, SEED, nu, N)
    gen = (f[1] > f[0]).astype(int)  # dominant generative component per node

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    sizes = [nu[i] * 9000 + 40 for i in range(N)]
    for a, col, title in [
        (ax[0], gen, "(a) Generative components"),
        (ax[1], lab, "(b) SA partition & centroids ($\\star$)"),
    ]:
        nx.draw(
            qg,
            pos=pos,
            ax=a,
            node_color=col,
            cmap=CMAP,
            node_size=sizes,
            edge_color="gainsboro",
            width=1.0,
            with_labels=False,
        )
        a.set_title(title, fontsize=12)
        a.axis("off")
    for c in cent:
        ax[1].plot(
            pos[c][0],
            pos[c][1],
            marker="*",
            color="black",
            markersize=18,
            markeredgecolor="white",
            markeredgewidth=1.6,
            zorder=10,
        )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"figures/figure_1.{ext}", bbox_inches="tight", dpi=200)
    plt.close()
    print("figure_1 (grid) done")


def figure_sbm():
    qg = generate_random_sbm(
        sizes=[50, 50],
        p=[[0.8, 0.1], [0.1, 0.8]],
        weights=[1, 1],
        lengths=[[1, 4], [4, 1]],
        random_state=SEED,
    )
    qg.precomputing()
    nodes = list(qg.nodes())
    idx = {nd: i for i, nd in enumerate(nodes)}
    N = len(nodes)
    true = np.array([qg.nodes[n].get("block", 0) for n in nodes])
    dmap = dict(nx.all_pairs_dijkstra_path_length(qg, weight="length"))
    dist = np.array([[dmap[u][v] for v in nodes] for u in nodes], float)
    nu = np.ones(N) / N
    U = EXP.potential_matrix(dist, nu)
    b = 0.3 / EXP.compute_cstar(U, qg, nodes, idx)
    pos = nx.spring_layout(qg, seed=SEED)

    best = None
    for r in range(50):
        sample_ss, sa_ss = np.random.SeedSequence(SEED + r + 100).spawn(2)
        sampler = UniformNodeSampling(random_state=np.random.default_rng(sample_ss))
        pts = qg.sample_points(100, strategy=sampler)
        sa = SimulatedAnnealing(
            observations=pts,
            k=2,
            lambda0=1.0,
            beta0=b,
            step_size=0.01,
            energy_mode="obs",
            random_state=np.random.default_rng(sa_ss),
        )
        centers = sa.run(KMeansPlusPlus(), MinimizeEnergy(), robust_prop=0.1)
        e = sa.calculate_energy(centers)
        if best is None or e < best[1]:
            best = (
                np.argmin(qg.node_center_distances(centers), axis=1),
                e,
                [c.closest_node() for c in centers],
            )
    lab, _, cent = best

    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    for a, col, title in [
        (ax[0], true, "(a) True communities"),
        (ax[1], lab, "(b) SA partition & centroids ($\\star$)"),
    ]:
        nx.draw(
            qg,
            pos=pos,
            ax=a,
            node_color=col,
            cmap=CMAP,
            node_size=140,
            edge_color="gainsboro",
            width=0.6,
            with_labels=False,
        )
        a.set_title(title, fontsize=12)
        a.axis("off")
    for c in cent:
        ax[1].plot(
            pos[c][0],
            pos[c][1],
            marker="*",
            color="black",
            markersize=18,
            markeredgecolor="white",
            markeredgewidth=1.6,
            zorder=10,
        )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"figures/figure_2.{ext}", bbox_inches="tight", dpi=200)
    plt.close()
    print("figure_2 (sbm) done")


def figure_sphere():
    store = pickle.load(open("results/sphere_multi.pkl", "rb"))
    sres = store["per_seed"][SEED]
    data, dtrue = sres["data"], sres["dtrue"]
    m = sres["methods"]["SA-graph"]
    sel = int(np.argmin(m["energies"]))
    lab = m["labels"][sel]

    def sphere_bg(a):
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        a.plot_surface(
            np.outer(np.cos(u), np.sin(v)),
            np.outer(np.sin(u), np.sin(v)),
            np.outer(np.ones_like(u), np.cos(v)),
            color="whitesmoke",
            alpha=0.12,
            edgecolor="gainsboro",
            linewidth=0.2,
            zorder=1,
        )

    palette = np.array(["#e41a1c", "#377eb8", "#4daf4a"])
    fig = plt.figure(figsize=(12, 5.5))
    for k, (col, title) in enumerate(
        [(dtrue, "(a) True vMF components"), (lab, "(b) SA (graph) clusters")]
    ):
        a = fig.add_subplot(1, 2, k + 1, projection="3d")
        sphere_bg(a)
        # A dense 3D cloud hides surface-level markers under matplotlib's depth
        # sorting, so cluster centers are not drawn on the sphere.
        a.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            c=palette[np.asarray(col)],
            s=16,
            zorder=2,
            depthshade=True,
        )
        a.set_title(title, fontsize=12)
        a.view_init(elev=22, azim=45)
        a.set_xlim([-1, 1])
        a.set_ylim([-1, 1])
        a.set_zlim([-1, 1])
        a.axis("off")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(f"figures/figure_3.{ext}", bbox_inches="tight", dpi=200)
    plt.close()
    print("figure_3 (sphere) done")


if __name__ == "__main__":
    figure_grid()
    figure_sbm()
    figure_sphere()
