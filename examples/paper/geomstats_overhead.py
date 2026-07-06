"""Wall-time comparison of the clustering methods, and the cost of manifold
geometry without closed forms.

This justifies the epsilon-net for the paper. Annealing on the graph (``SA-graph``)
calls *no* manifold geometry per step -- graph distances are precomputed once --
so it stays cheap on any geodesic space. Methods that operate directly on the
manifold are only cheap where closed-form exp/log/distance exist (as on S^2); a
``NaiveSphere`` that keeps geomstats stands in for a manifold without closed forms.

Timed on one seed at publication settings, ``n_runs`` restarts each:

  SA-graph     annealing on the epsilon-net graph        (the paper's method)
  SA-sphere    annealing on S^2, closed-form geometry
  SA-naive     annealing on S^2, geomstats geometry      (no closed forms)
  CLVQ         streaming CLVQ, closed-form geometry       (baseline)
  CLVQ-naive   streaming CLVQ, geomstats geometry         (no closed forms)
  k-medoids    weighted k-medoids on the geodesic matrix  (baseline)

    python geomstats_overhead.py
"""

import _env  # noqa: F401  -- pins BLAS threads; must precede numpy
import csv
import os
import time

import numpy as np
from geomstats.geometry.hypersphere import Hypersphere

from kmeanssa_ng import RiemannianPoint
from kmeanssa_ng.riemannian_manifold import RiemannianManifold, Sphere
from kmeanssa_ng.core.metrics import compute_labels, adjusted_rand_index

import sphere as S
import baselines as B
from multistart import annealings, method_entropy


class NaiveSphere(RiemannianManifold):
    """A sphere that overrides nothing, so exp/log/distance fall back to geomstats.

    It stands in for a general manifold on which closed-form geometry is unavailable.
    """


def _best_ari(labels, dtrue):
    return max(adjusted_rand_index(np.asarray(lab), dtrue) for lab in labels)


def _sa_graph(qg, V, nbr, node_list, nu_row, proj, data, dtrue, n_obs, b, n_runs, seed):
    labels, _, _, _ = S._sa_graph_runs(
        qg, V, nbr, node_list, nu_row, proj, len(data), n_obs, b, n_runs, seed, False
    )
    return _best_ari(labels, dtrue)


def _sa_manifold(manifold, data, dtrue, n_obs, b, n_runs, seed):
    all_points = [RiemannianPoint(manifold, x) for x in data]
    n = len(data)

    def obs_for(rng):
        idx = rng.integers(0, n, size=n_obs)
        return [RiemannianPoint(manifold, data[i]) for i in idx]

    # The closed-form and naive spheres share the same entropy on purpose:
    # identical streams mean identical trajectories, so the timing ratio
    # isolates the cost of the geometry backend.
    labels = [
        np.array(compute_labels(manifold, all_points, centers))
        for _, centers, _ in annealings(
            obs_for, 3, b, n_runs, "overhead", seed, method="sa-manifold"
        )
    ]
    return _best_ari(labels, dtrue)


def _clvq_naive(
    data, dtrue, manifold, n_runs, entropy, n_epochs=10, gamma0=1.0, decay=0.6
):
    """CLVQ with every geometric quantity (distance, log, exp) via geomstats."""
    n = len(data)
    best = []
    for child in entropy.spawn(n_runs):
        rng = np.random.default_rng(child)
        stream = rng.integers(0, n, size=n_epochs * n)
        c = data[rng.choice(n, size=3, replace=False)].astype(float).copy()
        for t, idx in enumerate(stream):
            z = data[idx]
            i = int(np.argmin([manifold.norm(ci, manifold.log(ci, z)) for ci in c]))
            c[i] = manifold.exp(c[i], gamma0 / (t + 1) ** decay * manifold.log(c[i], z))
        dist = np.array(
            [[manifold.norm(ci, manifold.log(ci, x)) for ci in c] for x in data]
        )
        best.append(adjusted_rand_index(np.argmin(dist, axis=1), dtrue))
    return max(best)


def _timed(fn):
    t = time.perf_counter()
    ari = fn()
    return time.perf_counter() - t, ari


def main(seed=42, n_data=2000, n_obs=2000, n_net=5000, n_runs=3, b=0.2):
    data, dtrue = S.make_data(seed, n_data)

    # epsilon-net + the projection/weights the graph annealing needs
    qg, V, nbr, node_list, _, _ = S.build_graph(n_net)
    proj = np.argmin(np.arccos(np.clip(data @ V.T, -1, 1)), axis=1)
    nu = np.bincount(proj, minlength=len(V)).astype(float)
    nu /= nu.sum()
    nu_row = nu[node_list]
    closed, naive = Sphere(Hypersphere(2)), NaiveSphere(Hypersphere(2))
    geodesic = np.arccos(np.clip(data @ data.T, -1.0, 1.0))
    w = np.ones(n_data) / n_data

    # warm up the JIT (graph) and geomstats paths on tiny runs, excluded from timing
    _sa_graph(qg, V, nbr, node_list, nu_row, proj, data, dtrue, 50, b, 1, seed)
    _sa_manifold(naive, data[:50], dtrue[:50], 50, b, 1, seed)

    t = {}
    a = {}
    t["SA-graph"], a["SA-graph"] = _timed(
        lambda: _sa_graph(
            qg, V, nbr, node_list, nu_row, proj, data, dtrue, n_obs, b, n_runs, seed
        )
    )
    t["SA-sphere"], a["SA-sphere"] = _timed(
        lambda: _sa_manifold(closed, data, dtrue, n_obs, b, n_runs, seed)
    )
    t["SA-naive"], a["SA-naive"] = _timed(
        lambda: _sa_manifold(naive, data, dtrue, n_obs, b, n_runs, seed)
    )
    # CLVQ and CLVQ-naive share the same entropy on purpose (identical streams,
    # so the ratio isolates the geometry backend).
    t["CLVQ"], a["CLVQ"] = _timed(
        lambda: _best_ari(
            B.clvq_sphere(data, 3, n_runs, method_entropy("overhead", seed, "clvq"))[0],
            dtrue,
        )
    )
    t["CLVQ-naive"], a["CLVQ-naive"] = _timed(
        lambda: _clvq_naive(
            data, dtrue, naive, n_runs, method_entropy("overhead", seed, "clvq")
        )
    )
    t["k-medoids"], a["k-medoids"] = _timed(
        lambda: _best_ari(
            B.weighted_kmedoids(
                geodesic, w, 3, n_runs, method_entropy("overhead", seed, "k-medoids")
            )[0],
            dtrue,
        )
    )

    print(f"one seed, k=3, n_obs={n_obs}, n_runs={n_runs}\n")
    print(f"{'method':12s}  {'time':>8}  {'best ARI':>9}")
    for m in ["SA-graph", "SA-sphere", "SA-naive", "CLVQ", "CLVQ-naive", "k-medoids"]:
        print(f"{m:12s}  {t[m]:7.1f}s  {a[m]:9.3f}")

    def ratio(x, y):
        return f"{t[x] / t[y]:.1f}x" if t[y] else "n/a"

    print("\ncomparisons:")
    print(
        f"  (1) SA-graph vs SA-sphere vs SA-naive : "
        f"{t['SA-graph']:.1f}s / {t['SA-sphere']:.1f}s / {t['SA-naive']:.1f}s "
        f"(naive is {ratio('SA-naive', 'SA-graph')} SA-graph)"
    )
    print(
        f"  (2) CLVQ vs CLVQ-naive (geomstats)    : "
        f"{t['CLVQ']:.1f}s / {t['CLVQ-naive']:.1f}s ({ratio('CLVQ-naive', 'CLVQ')} slower)"
    )
    print(
        f"  (3) SA-graph vs CLVQ                  : "
        f"{t['SA-graph']:.1f}s / {t['CLVQ']:.1f}s"
    )
    print(
        f"  (4) SA-graph vs k-medoids             : "
        f"{t['SA-graph']:.1f}s / {t['k-medoids']:.1f}s"
    )

    # Save a CSV with the time ratio relative to SA-graph (the baseline method).
    order = ["SA-graph", "SA-sphere", "SA-naive", "CLVQ", "CLVQ-naive", "k-medoids"]
    base = t["SA-graph"]
    os.makedirs("results", exist_ok=True)
    with open("results/geomstats_overhead.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["method", "time_s", "best_ari", "ratio_vs_sa_graph"])
        for m in order:
            writer.writerow([m, f"{t[m]:.3f}", f"{a[m]:.4f}", f"{t[m] / base:.3f}"])
    print("\nsaved results/geomstats_overhead.csv")


if __name__ == "__main__":
    main()
