"""Rebuild the three partition figures from saved data and reproducible re-runs.

Sphere: plotted directly from results/sphere_multi.pkl. Grid and SBM: the
lowest-energy SA run of seed 42 is regenerated (same seeding as the experiments),
giving the node labels and centroids of the partition plot.

  figures/figure_{1,2,3}.{pdf,png}
"""

import pickle

import numpy as np
import networkx as nx
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from kmeanssa_ng import QGPoint  # noqa: E402
from kmeanssa_ng.quantum_graph.sampling import UniformNodeSampling  # noqa: E402

import grid  # noqa: E402
import sbm  # noqa: E402
from multistart import annealings  # noqa: E402

SEED = 42
CMAP = plt.cm.coolwarm


def _lowest_energy_run(graph, observations_for, b, n_runs, seed, energy_of):
    """Return (node_labels, centroid_nodes) of the lowest-energy run over n_runs."""
    best_energy, best = np.inf, None
    for _, centers, sa in annealings(observations_for, 2, b, n_runs, seed + 100):
        energy = energy_of(sa, centers)
        if energy < best_energy:
            best_energy = energy
            best = (
                np.argmin(graph.node_center_distances(centers), axis=1),
                [c.closest_node() for c in centers],
            )
    return best


def _draw_partition(graph, pos, panels, node_size, width, centroid_nodes, path):
    """Two-panel node plot (reference vs SA partition) with centroid stars."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 5.5))
    for axis, (colors, title) in zip(ax, panels):
        nx.draw(
            graph,
            pos=pos,
            ax=axis,
            node_color=colors,
            cmap=CMAP,
            node_size=node_size,
            edge_color="gainsboro",
            width=width,
            with_labels=False,
        )
        axis.set_title(title, fontsize=12)
        axis.axis("off")
    for node in centroid_nodes:
        ax[1].plot(
            pos[node][0],
            pos[node][1],
            marker="*",
            color="black",
            markersize=18,
            markeredgecolor="white",
            markeredgewidth=1.6,
            zorder=10,
        )
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{path}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)


def figure_grid():
    space = grid.build_space()
    graph, nodes, nu = space.graph, space.nodes, space.nu
    _, data_nodes, obs_idx = grid.sample_data(
        SEED, space.densities, len(nodes), 1000, 1000
    )
    obs = [QGPoint(graph, (nodes[v], space.neighbour[nodes[v]]), 0) for v in obs_idx]
    lab, cent = _lowest_energy_run(
        graph,
        lambda _rng: obs,
        space.b,
        30,
        SEED,
        energy_of=lambda sa, centers: graph.node_energy(centers, weights=nu),
    )
    dominant = (space.densities[1] > space.densities[0]).astype(int)
    pos = {node: node for node in nodes}
    sizes = [nu[i] * 9000 + 40 for i in range(len(nodes))]
    _draw_partition(
        graph,
        pos,
        [
            (dominant, "(a) Generative components"),
            (lab, "(b) SA partition & centroids ($\\star$)"),
        ],
        sizes,
        1.0,
        cent,
        "figures/figure_1",
    )
    print("figure_1 (grid) done")


def figure_sbm():
    space = sbm.build_space(SEED)
    graph, true = space.graph, space.true_labels
    lab, cent = _lowest_energy_run(
        graph,
        lambda rng: graph.sample_points(
            100, strategy=UniformNodeSampling(random_state=rng)
        ),
        space.b,
        50,
        SEED,
        energy_of=lambda sa, centers: sa.calculate_energy(centers),
    )
    pos = nx.spring_layout(graph, seed=SEED)
    _draw_partition(
        graph,
        pos,
        [
            (true, "(a) True communities"),
            (lab, "(b) SA partition & centroids ($\\star$)"),
        ],
        140,
        0.6,
        cent,
        "figures/figure_2",
    )
    print("figure_2 (sbm) done")


def figure_sphere():
    store = pickle.load(open("results/sphere_multi.pkl", "rb"))
    sres = store["per_seed"][SEED]
    data, dtrue = sres["data"], sres["dtrue"]
    method = sres["methods"]["SA-graph"]
    lab = method["labels"][int(np.argmin(method["energies"]))]

    def sphere_bg(axis):
        u = np.linspace(0, 2 * np.pi, 40)
        v = np.linspace(0, np.pi, 40)
        axis.plot_surface(
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
    panels = [(dtrue, "(a) True vMF components"), (lab, "(b) SA (graph) clusters")]
    for k, (colors, title) in enumerate(panels):
        axis = fig.add_subplot(1, 2, k + 1, projection="3d")
        sphere_bg(axis)
        # A dense 3D cloud hides surface-level markers under matplotlib's depth
        # sorting, so cluster centers are not drawn on the sphere.
        axis.scatter(
            data[:, 0],
            data[:, 1],
            data[:, 2],
            c=palette[np.asarray(colors)],
            s=16,
            zorder=2,
            depthshade=True,
        )
        axis.set_title(title, fontsize=12)
        axis.view_init(elev=22, azim=45)
        axis.set_xlim([-1, 1])
        axis.set_ylim([-1, 1])
        axis.set_zlim([-1, 1])
        axis.axis("off")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/figure_3.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("figure_3 (sphere) done")


def figure_convergence():
    """Energy along the annealing time for the tracked run of each experiment."""
    experiments = [
        ("Grid $10\\times10$", "grid_multi"),
        ("SBM", "sbm_multi"),
        ("Sphere $\\mathbb{S}^2$", "sphere_multi"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for axis, (title, name) in zip(axes, experiments):
        conv = pickle.load(open(f"results/{name}.pkl", "rb"))["convergence"]
        axis.plot(conv["time"], conv["energy"], color="#377eb8")
        axis.set_title(title, fontsize=12)
        axis.set_xlabel("annealing time")
        axis.set_ylabel("energy $U$")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/figure_4.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("figure_4 (convergence) done")


def figure_rate():
    """Two panels on the toy: exceedance rate (log-log) and energy trajectories."""
    store = pickle.load(open("results/rate.pkl", "rb"))
    grid, stacked = store["grid"], store["stacked"]
    ustar, cstar, b = store["ustar"], store["cstar"], store["b"]
    excess = stacked - ustar

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    # (a) log-log exceedance with fitted slope vs predicted rate exponent.
    win = grid > grid[-1] / 10
    for mult in (1, 2, 4, 6):
        eta = mult * ustar
        p = (excess > eta).mean(axis=0)
        (line,) = axA.plot(grid[1:], p[1:], lw=1.6, label=rf"$\eta={mult}U^\downarrow$")
        ok = win & (p > 0)
        slope = np.polyfit(np.log(grid[ok]), np.log(p[ok]), 1)[0]
        pred = min(b * eta, 0.5 * (1 - b * cstar))
        axA.text(
            grid[-1] * 1.03,
            p[win][-1],
            rf"$\hat\rho={-slope:.2f}$" + "\n" + rf"$(\rho<{pred:.2f})$",
            fontsize=7,
            color=line.get_color(),
            va="center",
        )
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel("annealing time $t$")
    axA.set_ylabel(r"$\hat{\mathbb{P}}(U(X_t) - U^\downarrow > \eta)$")
    axA.set_title(f"(a) Exceedance rate ($b={b}$, $c^\\star={cstar:.2f}$)", fontsize=12)
    axA.legend(fontsize=8, loc="lower left")

    # (b) a few energy trajectories with quantile bands concentrating on U*.
    sample = np.random.default_rng(0).choice(
        stacked.shape[0], size=min(5, stacked.shape[0]), replace=False
    )
    for r in sample:
        axB.plot(grid[1:], stacked[r, 1:], color="0.5", lw=0.4, alpha=0.22)
    q05, q25, med, q75, q95 = np.percentile(stacked, [5, 25, 50, 75, 95], axis=0)
    axB.fill_between(grid[1:], q05[1:], q95[1:], color="C0", alpha=0.20, label="5-95%")
    axB.fill_between(grid[1:], q25[1:], q75[1:], color="C0", alpha=0.40, label="25-75%")
    axB.plot(grid[1:], med[1:], color="C3", lw=2, label="median")
    axB.axhline(ustar, ls="--", color="k", lw=1, label=r"$U^\downarrow$")
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel("annealing time $t$")
    axB.set_ylabel(r"energy $U(X_t)$")
    axB.set_title("(b) Energy trajectories: escape and concentration", fontsize=12)
    axB.legend(fontsize=8, loc="upper left")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/figure_5.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("figure_5 (rate) done")


def figure_memory():
    """Energy of the memory-augmented estimator: partial (1%) and full memory."""
    store = pickle.load(open("results/rate.pkl", "rb"))
    grid, stacked = store["grid"], store["stacked"]
    ustar = store["ustar"]

    # The time grid is linearly spaced, so a memory window of the last p of the
    # physical annealing time is exactly the last p of the indices up to t.
    n_grid = len(grid)
    stacked_10 = np.minimum.accumulate(stacked, axis=1)

    stacked_01 = np.empty_like(stacked)
    win_01 = max(1, int(0.01 * n_grid))
    for i in range(n_grid):
        start = max(0, i - win_01 + 1)
        stacked_01[:, i] = np.min(stacked[:, start : i + 1], axis=1)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 4.6))

    def plot_memory(ax, data, title):
        q05, q25, med, q75, q95 = np.percentile(data, [5, 25, 50, 75, 95], axis=0)
        ax.fill_between(
            grid[1:], q05[1:], q95[1:], color="C0", alpha=0.20, label="5-95%"
        )
        ax.fill_between(
            grid[1:], q25[1:], q75[1:], color="C0", alpha=0.40, label="25-75%"
        )
        ax.plot(grid[1:], med[1:], color="C3", lw=2, label="median")
        ax.axhline(ustar, ls="--", color="k", lw=1, label=r"$U^\downarrow$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        # Share figure_5's vertical range so the two panels compare directly.
        ymin = ustar * 0.99 if ustar > 0 else 0.1
        ymax = np.max(stacked[:, 1:]) * 1.05
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel("annealing time $t$")
        ax.set_ylabel(r"memory energy $V^p_{\mathcal{G}}(t)$")
        ax.set_title(title, fontsize=12)
        ax.legend(fontsize=8, loc="upper right")

    plot_memory(axA, stacked_01, r"(a) Partial memory ($p = 0.01$)")
    plot_memory(axB, stacked_10, r"(b) Full memory ($p = 1.0$)")

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"figures/figure_6.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print("figure_6 (memory) done")


if __name__ == "__main__":
    figure_grid()
    figure_sbm()
    figure_sphere()
    figure_convergence()
    figure_rate()
    figure_memory()
