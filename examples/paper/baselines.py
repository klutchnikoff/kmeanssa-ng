"""Baseline clustering methods for the k-means-on-metric-graphs experiments.

All baselines produce, for each run, a vector of per-node labels comparable to the
ground truth, so that the Adjusted Rand Index can be computed exactly as for the
simulated-annealing method. Aggregation (mean/std/best/energy-selected) is done by
`aggregate`.

- weighted_kmedoids: k-medoids on a precomputed (shortest-path / geodesic) distance
  matrix, minimizing the weighted within-cluster sum of squared distances --- i.e. the
  same discrete energy U_G as the paper, with centroids restricted to the vertex set.
- spectral_baseline: scikit-learn SpectralClustering on a precomputed affinity.
- clvq_sphere: Competitive Learning Vector Quantization (Le Brigant), the online
  geodesic stochastic-gradient method on S^2 using the exponential/logarithm maps.
"""

import numpy as np
from sklearn.cluster import SpectralClustering


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def aggregate(labels_per_run, energies, true_labels, ari_fn):
    """Return (ari_mean, ari_std, ari_best, ari_selected).

    ari_selected is the ARI of the run with the lowest energy (unsupervised selection,
    as in the paper). If `energies` is None, ari_selected falls back to the mean.
    """
    aris = np.array([ari_fn(lbl, true_labels) for lbl in labels_per_run])
    ari_mean, ari_std, ari_best = (
        float(aris.mean()),
        float(aris.std()),
        float(aris.max()),
    )
    if energies is None:
        ari_selected = ari_mean
    else:
        ari_selected = float(aris[int(np.argmin(energies))])
    return ari_mean, ari_std, ari_best, ari_selected


# --------------------------------------------------------------------------------------
# Weighted k-medoids on a distance matrix (graph "Lloyd" baseline)
# --------------------------------------------------------------------------------------
def _kpp_init(D2, weights, k, rng):
    """k-means++ style seeding on squared distances, weighted by node weights."""
    n = D2.shape[0]
    first = rng.choice(n, p=weights / weights.sum())
    medoids = [first]
    closest = D2[first].copy()
    for _ in range(1, k):
        p = weights * closest
        s = p.sum()
        if s <= 0:
            cand = rng.choice(n)
        else:
            cand = rng.choice(n, p=p / s)
        medoids.append(cand)
        closest = np.minimum(closest, D2[cand])
    return medoids


def _kmedoids_once(D2, weights, k, rng, max_iter=100):
    """One weighted k-medoids run. Returns (labels, energy)."""
    medoids = _kpp_init(D2, weights, k, rng)
    labels = np.argmin(D2[:, medoids], axis=1)
    for _ in range(max_iter):
        new_medoids = []
        for i in range(k):
            members = np.where(labels == i)[0]
            if len(members) == 0:  # empty cluster: reseed on worst-served node
                members = np.array([int(np.argmax(np.min(D2[:, medoids], axis=1)))])
            # medoid = member minimizing weighted sum of squared distances to the cluster
            costs = (weights[members][:, None] * D2[np.ix_(members, members)]).sum(
                axis=0
            )
            new_medoids.append(int(members[np.argmin(costs)]))
        new_labels = np.argmin(D2[:, new_medoids], axis=1)
        if new_medoids == medoids and np.array_equal(new_labels, labels):
            medoids, labels = new_medoids, new_labels
            break
        medoids, labels = new_medoids, new_labels
    energy = float((weights * D2[:, medoids].min(axis=1)).sum())
    return labels, energy


def weighted_kmedoids(D, weights, k, n_runs, base_seed):
    """Multi-start weighted k-medoids on distance matrix D (N x N), node weights (N,).

    Returns (labels_per_run, energies). Minimizes sum_v weights[v] * min_i D[v, m_i]^2,
    i.e. the discrete k-means energy with centroids in the vertex set.
    """
    D2 = np.asarray(D, dtype=float) ** 2
    weights = np.asarray(weights, dtype=float)
    labels_per_run, energies = [], []
    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        labels, energy = _kmedoids_once(D2, weights, k, rng)
        labels_per_run.append(labels)
        energies.append(energy)
    return labels_per_run, energies


# --------------------------------------------------------------------------------------
# Spectral clustering on a precomputed affinity
# --------------------------------------------------------------------------------------
def spectral_baseline(affinity, k, n_runs, base_seed):
    """SpectralClustering on a precomputed affinity (similarity) matrix.

    Returns (labels_per_run, None). Randomness comes from the final k-means on the
    spectral embedding; we restart n_runs times with distinct seeds.
    """
    affinity = np.asarray(affinity, dtype=float)
    labels_per_run = []
    for r in range(n_runs):
        sc = SpectralClustering(
            n_clusters=k,
            affinity="precomputed",
            assign_labels="kmeans",
            random_state=base_seed + r,
        )
        labels_per_run.append(sc.fit_predict(affinity))
    return labels_per_run, None


def rbf_affinity(D, scale=None):
    """Gaussian affinity exp(-D^2 / (2 sigma^2)) from a distance matrix."""
    D = np.asarray(D, dtype=float)
    if scale is None:
        off = D[~np.eye(D.shape[0], dtype=bool)]
        scale = np.median(off[off > 0])
    return np.exp(-(D**2) / (2.0 * scale**2))


# --------------------------------------------------------------------------------------
# CLVQ on the sphere S^2 (Le Brigant): online geodesic stochastic gradient.
# The sphere exp/log/distance are inlined (three-line numpy) rather than taken from
# geomstats: this baseline streams tens of thousands of points online, so per-call
# library overhead would dominate its runtime.
# --------------------------------------------------------------------------------------
def _sphere_geodesic_dist(a, B):
    """Geodesic distance from unit vector a to rows of B (unit vectors)."""
    return np.arccos(np.clip(B @ a, -1.0, 1.0))


def _sphere_log(a, b):
    """Logarithm map Log_a(b) on the unit sphere."""
    inner = float(np.clip(a @ b, -1.0, 1.0))
    theta = np.arccos(inner)
    if theta < 1e-9:
        return np.zeros_like(a)
    u = b - inner * a
    nu = np.linalg.norm(u)
    if nu < 1e-12:
        return np.zeros_like(a)
    return theta * u / nu


def _sphere_exp(a, v):
    """Exponential map Exp_a(v) on the unit sphere."""
    nv = np.linalg.norm(v)
    if nv < 1e-12:
        return a / np.linalg.norm(a)
    p = np.cos(nv) * a + np.sin(nv) * v / nv
    return p / np.linalg.norm(p)


def _clvq_once(node_coords, stream_idx, k, rng, gamma0=1.0, decay=0.6):
    """One CLVQ run. Streams node_coords[stream_idx] online; returns (labels, energy).

    Gain sequence gamma_t = gamma0 / (t+1)^decay satisfies the Robbins-Monro conditions
    for decay in (1/2, 1].
    """
    n = node_coords.shape[0]
    centroids = node_coords[rng.choice(n, size=k, replace=False)].astype(float).copy()
    for t, idx in enumerate(stream_idx):
        z = node_coords[idx]
        i = int(np.argmin(_sphere_geodesic_dist_centroids(centroids, z)))
        gamma = gamma0 / (t + 1) ** decay
        centroids[i] = _sphere_exp(centroids[i], gamma * _sphere_log(centroids[i], z))
    # assign every node to its nearest final centroid
    dists = np.arccos(np.clip(node_coords @ centroids.T, -1.0, 1.0))  # (n, k)
    labels = np.argmin(dists, axis=1)
    energy = float((dists.min(axis=1) ** 2).mean())
    return labels, energy


def _sphere_geodesic_dist_centroids(centroids, z):
    """Geodesic distances from point z to each centroid (rows of `centroids`)."""
    return np.arccos(np.clip(centroids @ z, -1.0, 1.0))


def clvq_sphere(node_coords, k, n_runs, base_seed, n_epochs=10, gamma0=1.0, decay=0.6):
    """Multi-start CLVQ on S^2. Streams uniformly-sampled nodes (n_epochs passes).

    Returns (labels_per_run, energies). `node_coords` is (N, 3) unit vectors.
    """
    node_coords = np.asarray(node_coords, dtype=float)
    n = node_coords.shape[0]
    labels_per_run, energies = [], []
    for r in range(n_runs):
        rng = np.random.default_rng(base_seed + r)
        stream = rng.integers(0, n, size=n_epochs * n)  # uniform on nodes, like nu
        labels, energy = _clvq_once(
            node_coords, stream, k, rng, gamma0=gamma0, decay=decay
        )
        labels_per_run.append(labels)
        energies.append(energy)
    return labels_per_run, energies
