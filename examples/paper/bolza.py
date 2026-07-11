"""Clustering on the Bolza surface: a compact genus-2 hyperbolic example.

A negatively curved counterpart to the sphere experiment. We approximate the
Bolza surface (Poincare disk / Fuchsian octagon group) by an intrinsic epsilon-net
quantum graph, draw three hyperbolic-Gaussian blobs on it, and recover them with
simulated-annealing k-means on the graph. The figure shows the fundamental
octagon, the data coloured by true mode, and the net coloured by recovered cluster
with the centroids.

Run: ``python bolza.py`` (caches results/bolza.pkl, then writes
figures/figure_bolza.{pdf,png}; ``reproduce.py`` runs it as its Part 4). The net
is cached in data/ since the intrinsic (quotient-aware) repulsion is the build's
bottleneck. The experiment is a qualitative illustration: one documented seed,
with every random stream derived from the shared structured entropy.
"""

import _env  # noqa: F401  -- pins BLAS threads; must precede numpy
import os
import pickle

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

from kmeanssa_ng import create_bolza_surface, QGPoint
from kmeanssa_ng.riemannian_manifold import (
    RepulsionNet,
    build_epsilon_net_graph,
    estimate_covering_radius,
)
from kmeanssa_ng.riemannian_manifold import bolza as bz

from multistart import annealings, code_stamp, data_entropy

N_NET = 400  # epsilon-net points
N_ITER = 120  # repulsion relaxation steps
K = 3  # clusters
N_DATA = 900  # observations
BETA0 = 0.5  # drift strength
STEP_SIZE = 0.05  # SDE time-discretization step
SIGMA = 0.22  # blob spread, as a hyperbolic-distance scale (surface diameter ~2.09)
# Three well-separated modes inside the fundamental octagon.
MODES = np.array([0.45 * np.exp(1j * (0.4 + t * 2 * np.pi / 3)) for t in range(K)])
COLORS = np.array(["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"])
NET_CACHE = "data/bolza_net_{n}.npy"
PKL = "results/bolza.pkl"


# --------------------------------------------------------------------------- #
# Net + graph
# --------------------------------------------------------------------------- #
def build_or_load_net(surface):
    """Deterministic epsilon-net, cached (the intrinsic repulsion is the cost)."""
    path = NET_CACHE.format(n=N_NET)
    if os.path.exists(path):
        return np.load(path)
    os.makedirs("data", exist_ok=True)
    net = RepulsionNet(n_iter=N_ITER, random_state=0).build(surface, N_NET)
    np.save(path, net)
    return net


def nearest_node(points, targets):
    """Index of the net point nearest each target, by quotient distance."""
    zp = points[:, 0] + 1j * points[:, 1]
    zt = np.asarray(targets)
    return np.argmin(bz.quotient_distance(zt[:, None], zp[None, :]), axis=1)


# --------------------------------------------------------------------------- #
# Data: three hyperbolic-Gaussian blobs on the surface
# --------------------------------------------------------------------------- #
def generate_data(surface, rng, n_data):
    """Three isotropic hyperbolic-Gaussian blobs, one per mode."""
    labels = rng.integers(0, K, n_data)
    base = MODES[labels]
    g = rng.standard_normal(n_data) + 1j * rng.standard_normal(n_data)
    # Ambient tangent whose conformal (hyperbolic) norm is SIGMA*|g|: scaling by
    # (1 - |base|^2) / 2 measures the spread in hyperbolic distance, independent of
    # where the mode sits in the disk.
    tangent = 0.5 * SIGMA * (1.0 - np.abs(base) ** 2) * g
    points = bz.fold_to_domain(bz.exp_map(base, tangent))
    return points, labels


# --------------------------------------------------------------------------- #
# Simulated annealing on the graph (multi-start, keep the lowest energy)
# --------------------------------------------------------------------------- #
def cluster(qg, data_nodes, n_runs, seed):
    """Multi-start SA on the net graph; return the lowest-energy partition.

    The reference measure is the empirical node measure of the observations
    (set on ``obs_weight`` for the annealer, and used as the selection weights),
    as in the sphere experiment.
    """
    nu = np.bincount(data_nodes, minlength=qg.number_of_nodes()).astype(float)
    nu /= nu.sum()
    for v, w in zip(qg.nodes(), nu):
        qg.nodes[v]["obs_weight"] = float(w)
    nbr = {v: next(iter(qg.neighbors(v))) for v in qg.nodes()}
    obs = [QGPoint(qg, (int(v), nbr[int(v)]), 0) for v in data_nodes]

    node_labels, energies, centroids = [], [], []
    for _, centers, _ in annealings(
        lambda _rng: obs, K, BETA0, n_runs, "bolza", seed, step_size=STEP_SIZE
    ):
        node_labels.append(np.argmin(qg.node_center_distances(centers), axis=1))
        energies.append(qg.node_energy(centers, weights=nu))
        centroids.append(np.array([c.closest_node() for c in centers]))
    best = int(np.argmin(energies))
    return node_labels[best], centroids[best], np.array(energies)


# --------------------------------------------------------------------------- #
# Figure
# --------------------------------------------------------------------------- #
def _geodesic_arc(z1, z2, n=60):
    """Sample the hyperbolic geodesic between two disk points (a circular arc)."""
    w = bz._mobius_to_origin(z2, z1)  # send z1 -> 0; geodesic to w is the segment [0,w]
    seg = np.linspace(0, 1, n) * w
    return bz._mobius_from_origin(seg, z1)


def _draw_octagon(ax):
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), color="0.8", lw=1)  # unit circle (boundary)
    V = bz.VERTICES
    for k in range(bz.N_SIDES):
        arc = _geodesic_arc(V[k], V[(k + 1) % bz.N_SIDES])
        ax.plot(arc.real, arc.imag, color="0.35", lw=1.6)
    ax.set_aspect("equal")
    ax.set_xlim(-1.02, 1.02)
    ax.set_ylim(-1.02, 1.02)
    ax.axis("off")


def _nearest_center(grid_z, centers_z):
    """Index of the nearest of ``centers_z`` for each grid point, by quotient
    distance -- so the Voronoi cells wrap correctly across the identified boundary."""
    flat = grid_z.ravel()
    d = bz.quotient_distance(flat[:, None], np.asarray(centers_z)[None, :])
    return d.argmin(axis=1).reshape(grid_z.shape)


def _fill_regions(ax, X, Y, region, cmap):
    ax.pcolormesh(
        X,
        Y,
        np.ma.masked_invalid(region),
        cmap=cmap,
        vmin=-0.5,
        vmax=K - 0.5,
        alpha=0.40,
        shading="nearest",
        rasterized=True,
        zorder=0,
    )


def make_figure(net, data, true_labels, data_label, centroids, ari, stem):
    from matplotlib.colors import ListedColormap

    cmap = ListedColormap(COLORS[:K])
    centre_z = net[centroids, 0] + 1j * net[centroids, 1]
    # Align recovered cluster ids to the true modes (majority vote) so both panels
    # share one colour scheme and are directly comparable.
    perm = np.array(
        [
            np.bincount(true_labels[data_label == c], minlength=K).argmax()
            if np.any(data_label == c)
            else c
            for c in range(K)
        ]
    )

    res = 400  # raster grid: quotient-distance Voronoi cells ("tiles")
    g = np.linspace(-0.9, 0.9, res)
    X, Y = np.meshgrid(g, g)
    Z = X + 1j * Y
    # |z| < 0.9 drops the grid corners (|z| >= 1 is outside the Poincaré disk, where
    # the hyperbolic distance is undefined); in_fundamental_domain shapes the octagon.
    inside = bz.in_fundamental_domain(Z.ravel()).reshape(Z.shape) & (np.abs(Z) < 0.9)
    reg_true = np.where(inside, _nearest_center(Z, MODES), np.nan)
    reg_rec = np.where(inside, perm[_nearest_center(Z, centre_z)], np.nan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5.6))

    _fill_regions(ax1, X, Y, reg_true, cmap)
    _draw_octagon(ax1)
    ax1.scatter(data.real, data.imag, s=7, c=COLORS[true_labels], alpha=0.5, lw=0)
    ax1.scatter(
        MODES.real,
        MODES.imag,
        s=200,
        marker="*",
        c="k",
        edgecolors="w",
        linewidths=1.2,
        zorder=5,
    )
    ax1.set_title("True partition and data (3 modes)")

    _fill_regions(ax2, X, Y, reg_rec, cmap)
    _draw_octagon(ax2)
    ax2.scatter(
        MODES.real,
        MODES.imag,
        s=70,
        marker="o",
        facecolors="none",
        edgecolors="k",
        linewidths=1.1,
        zorder=4,
    )  # true modes, for reference
    ax2.scatter(
        centre_z.real,
        centre_z.imag,
        s=200,
        marker="*",
        c="k",
        edgecolors="w",
        linewidths=1.2,
        zorder=5,
    )  # recovered centres
    ax2.set_title(f"Recovered partition (ARI = {ari:.3f})")

    fig.suptitle(
        "Simulated-annealing k-means on the Bolza surface (genus-2, curvature -1)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs("figures", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", dpi=200)
    print(f"saved {stem}.{{pdf,png}}", flush=True)


def run(seed=42, n_runs=8, n_data=N_DATA):
    """Run the Bolza experiment and cache its raw results to ``results/bolza.pkl``.

    The figure is rendered separately (``render``, called by
    ``make_figures.figure_bolza``), so it is a pure view of these results.
    """
    surface = create_bolza_surface()

    net = build_or_load_net(surface)
    eps = estimate_covering_radius(
        surface, net, n_test=5000, random_state=0, intrinsic=True
    )
    ell = float(np.sqrt(eps))
    qg = build_epsilon_net_graph(surface, net, ell=ell, intrinsic=True)
    print(
        f"[bolza] net={len(net)} ell={ell:.3f} "
        f"nodes={qg.number_of_nodes()} edges={qg.number_of_edges()}",
        flush=True,
    )

    data, true_labels = generate_data(
        surface, np.random.default_rng(data_entropy("bolza", seed)), n_data
    )
    data_nodes = nearest_node(net, data)
    node_label, centroids, energies = cluster(qg, data_nodes, n_runs, seed)
    data_label = node_label[data_nodes]
    ari = adjusted_rand_score(true_labels, data_label)
    print(f"[bolza] ARI = {ari:.3f}", flush=True)

    store = {
        "name": "bolza",
        "code": code_stamp(),
        "config": {
            "seed": seed,
            "n_runs": n_runs,
            "n_data": n_data,
            "n_net": N_NET,
            "sigma": SIGMA,
            "beta0": BETA0,
            "step_size": STEP_SIZE,
        },
        "net": net,
        "ell": ell,
        "data": data,
        "true_labels": true_labels,
        "data_label": data_label,
        "node_label": node_label,
        "centroids": centroids,
        "energies": energies,
        "ari": ari,
    }
    os.makedirs("results", exist_ok=True)
    with open(PKL, "wb") as f:
        pickle.dump(store, f)
    print(f"saved {PKL}", flush=True)
    return store


def render(store, stem="figures/figure_bolza"):
    """Render the partition figure from a cached result store."""
    make_figure(
        store["net"],
        store["data"],
        store["true_labels"],
        store["data_label"],
        store["centroids"],
        store["ari"],
        stem,
    )


if __name__ == "__main__":
    render(run())
