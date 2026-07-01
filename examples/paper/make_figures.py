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
    graph, nodes, nu = space["graph"], space["nodes"], space["nu"]
    _, data_nodes, obs_idx = grid.sample_data(
        SEED, space["densities"], len(nodes), 1000, 1000
    )
    obs = [QGPoint(graph, (nodes[v], space["neighbour"][nodes[v]]), 0) for v in obs_idx]
    lab, cent = _lowest_energy_run(
        graph,
        lambda _rng: obs,
        space["b"],
        30,
        SEED,
        energy_of=lambda sa, centers: graph.node_energy(centers, weights=nu),
    )
    dominant = (space["densities"][1] > space["densities"][0]).astype(int)
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
    graph, nodes, true, _dist, _nu, b = sbm.build_space(SEED)
    lab, cent = _lowest_energy_run(
        graph,
        lambda rng: graph.sample_points(
            100, strategy=UniformNodeSampling(random_state=rng)
        ),
        b,
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


if __name__ == "__main__":
    figure_grid()
    figure_sbm()
    figure_sphere()
