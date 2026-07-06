"""Shared multi-start driver for the simulated-annealing experiments.

Every experiment runs the annealer many times from independent seeds and keeps the
per-run labels and energies; the lowest-energy run is selected downstream. This
module factors out that seeded loop and the per-method ARI aggregation.
"""

import numpy as np
import networkx as nx

from kmeanssa_ng import SimulatedAnnealing, KMeansPlusPlus, MinimizeEnergy
from kmeanssa_ng.core.metrics import adjusted_rand_index

# Structured entropy for every random stream of the pipeline. Seeding by
# arithmetic (seed + r) makes streams collide as soon as s + r = s' + r'
# (run 1 of seed 42 is run 0 of seed 43); SeedSequence entropy lists have no
# such collisions, and distinct experiments and methods get distinct streams.
_EXPERIMENT_IDS = {
    "grid": 1,
    "sbm": 2,
    "sphere": 3,
    "rate": 4,
    "bolza": 5,
    "overhead": 6,
}
_METHOD_IDS = {"sa": 0, "sa-manifold": 1, "k-medoids": 2, "spectral": 3, "clvq": 4}


def method_entropy(experiment, seed, method="sa"):
    """Root ``SeedSequence`` for one (experiment, seed, method) family of runs.

    Spawn one child per run: children of a ``SeedSequence`` are mutually
    independent, so every (experiment, seed, method, run) tuple gets its own
    stream.
    """
    return np.random.SeedSequence(
        [_EXPERIMENT_IDS[experiment], seed, _METHOD_IDS[method]]
    )


def _register_obs_counts(observations):
    """Populate the graph's per-node ``nb_obs`` from these observations.

    ``energy_history`` is recorded in "obs" energy mode, which weights each node
    by its observed-point count (``nb_obs``). Only the graph samplers set that
    attribute, so experiments that build observations by hand (grid, sphere)
    must register them too -- otherwise the recorded energy is identically zero.
    Mirrors what ``QuantumGraph.sample_points`` does: reset, then count.
    """
    graph = observations[0].space
    nx.set_node_attributes(graph, 0, "nb_obs")
    counts = {}
    for point in observations:
        node = point.closest_node()
        counts[node] = counts.get(node, 0) + 1
    nx.set_node_attributes(graph, counts, "nb_obs")


def annealings(
    observations_for, k, beta0, n_runs, experiment, seed, method="sa", track_first=False
):
    """Yield (run_index, centers, sa) for ``n_runs`` independently-seeded SA runs.

    Args:
        observations_for: callback ``rng -> observations`` building a run's data from
            a dedicated Generator, kept separate from the annealing stream.
        k, beta0: number of clusters and drift strength.
        n_runs: number of restarts.
        experiment, seed, method: entropy of the run streams (``method_entropy``).
        track_first: record the energy history of the first run (``sa.energy_history``).
    """
    run_entropy = method_entropy(experiment, seed, method).spawn(n_runs)
    for r in range(n_runs):
        obs_seed, sa_seed = run_entropy[r].spawn(2)
        observations = observations_for(np.random.default_rng(obs_seed))
        if track_first and r == 0:
            _register_obs_counts(observations)
        sa = SimulatedAnnealing(
            observations=observations,
            k=k,
            lambda0=1.0,
            beta0=beta0,
            step_size=0.01,
            energy_mode="obs",
            random_state=np.random.default_rng(sa_seed),
        )
        centers = sa.run(
            KMeansPlusPlus(),
            MinimizeEnergy(),
            robust_prop=0.1,
            record_energy=track_first and r == 0,
        )
        yield r, centers, sa


def run_seeds(seeds, fn, n_jobs=1, tag=""):
    """Run ``fn(i, seed) -> (per_seed_value, convergence)`` over all seeds.

    Seeds are independent, so they run in parallel when ``n_jobs != 1`` (joblib's
    loky backend: each worker gets its own copy of the space, so per-node ``nb_obs``
    mutations never race). The result is order-independent -- all randomness flows
    through explicit seed-derived generators -- so the merged output is identical
    whatever ``n_jobs`` is. ``convergence`` is the diagnostic recorded by the one
    tracked seed (``i == 0``); the others return ``None``.
    """
    from joblib import Parallel, delayed

    def one(i, seed):
        value, conv = fn(i, seed)
        print(f"[{tag}] seed {seed} done", flush=True)
        return seed, value, conv

    results = Parallel(n_jobs=n_jobs)(
        delayed(one)(i, seed) for i, seed in enumerate(seeds)
    )
    per_seed, convergence = {}, None
    for seed, value, conv in results:
        per_seed[seed] = value
        convergence = conv or convergence
    return per_seed, convergence


def methods_from_raw(raw, true_labels):
    """Turn ``{method: (labels_per_run, energies)}`` into ARI/energy/label records."""
    methods = {}
    for name, (labels, energies) in raw.items():
        aris = np.array([adjusted_rand_index(lbl, true_labels) for lbl in labels])
        methods[name] = {
            "aris": aris,
            "energies": np.asarray(energies) if energies is not None else None,
            "labels": [np.asarray(lbl) for lbl in labels],
        }
    return methods


def _selected_ari(method_record):
    """ARI of the lowest-energy run, or the mean when there is no energy signal."""
    aris = method_record["aris"]
    energies = method_record["energies"]
    if energies is not None and len(energies):
        return float(aris[int(np.argmin(energies))])
    return float(np.mean(aris))


def summarize(store):
    """Print, per method, the ARI averaged over seeds (mean / best / selected)."""
    per_seed = list(store["per_seed"].values())
    methods = per_seed[0]["methods"].keys()
    print(f"{'method':12s}  mean   best   selected")
    for method in methods:
        records = [sres["methods"][method] for sres in per_seed]
        mean = np.mean([r["aris"].mean() for r in records])
        best = np.mean([r["aris"].max() for r in records])
        selected = np.mean([_selected_ari(r) for r in records])
        print(f"{method:12s}  {mean:.3f}  {best:.3f}  {selected:.3f}")
