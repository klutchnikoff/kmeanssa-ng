"""Shared multi-start driver for the simulated-annealing experiments.

Every experiment runs the annealer many times from independent seeds and keeps the
per-run labels and energies; the lowest-energy run is selected downstream. This
module factors out that seeded loop and the per-method ARI aggregation.
"""

import os
import pickle

import numpy as np

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


def annealings(
    observations_for,
    k,
    beta0,
    n_runs,
    experiment,
    seed,
    method="sa",
    step_size=0.01,
    track_first=False,
):
    """Yield (run_index, centers, sa) for ``n_runs`` independently-seeded SA runs.

    The annealer runs in "obs" energy mode, which reads the reference measure
    from the graph's per-node ``obs_weight``. The experiment owns that measure: it
    must be in place before (or set by) ``observations_for``, and it is the
    same for every run and every seed -- so the tracked energy history, the
    internal ``MinimizeEnergy`` selection and the downstream selection all
    evaluate the same functional.

    Args:
        observations_for: callback ``rng -> observations`` building a run's data from
            a dedicated Generator, kept separate from the annealing stream.
        k, beta0: number of clusters and drift strength.
        n_runs: number of restarts.
        experiment, seed, method: entropy of the run streams (``method_entropy``).
        step_size: SDE time-discretization step of the annealer.
        track_first: record the energy history of the first run (``sa.energy_history``).
    """
    run_entropy = method_entropy(experiment, seed, method).spawn(n_runs)
    for r in range(n_runs):
        obs_seed, sa_seed = run_entropy[r].spawn(2)
        observations = observations_for(np.random.default_rng(obs_seed))
        sa = SimulatedAnnealing(
            observations=observations,
            k=k,
            lambda0=1.0,
            beta0=beta0,
            step_size=step_size,
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


def run_seeds(seeds, fn, n_jobs=1, tag="", checkpoint_dir=None, config=None):
    """Run ``fn(i, seed) -> (per_seed_value, convergence)`` over all seeds.

    Seeds are independent, so they run in parallel when ``n_jobs != 1`` (joblib's
    loky backend: each worker gets its own copy of the space, so per-node ``obs_weight``
    mutations never race). The result is order-independent -- all randomness flows
    through explicit seed-derived generators -- so the merged output is identical
    whatever ``n_jobs`` is. ``convergence`` is the diagnostic recorded by the one
    tracked seed (``i == 0``); the others return ``None``.

    When ``checkpoint_dir`` is set, each seed's result is written there on
    completion (``seed_<s>.pkl``, atomically) together with ``config``, and a
    later invocation with the same ``config`` resumes from those files instead
    of recomputing -- a crash at seed 99 no longer loses the first 98. A
    checkpoint whose config differs belongs to another protocol and is
    recomputed (then overwritten).
    """
    from joblib import Parallel, delayed

    def ckpt_path(seed):
        return os.path.join(checkpoint_dir, f"seed_{seed}.pkl")

    def load_checkpoint(seed):
        if checkpoint_dir is None or not os.path.exists(ckpt_path(seed)):
            return None
        with open(ckpt_path(seed), "rb") as f:
            saved = pickle.load(f)
        return saved if saved.get("config") == config else None

    def one(i, seed):
        saved = load_checkpoint(seed)
        if saved is not None:
            print(f"[{tag}] seed {seed} resumed from checkpoint", flush=True)
            return seed, saved["value"], saved["convergence"]
        value, conv = fn(i, seed)
        if checkpoint_dir is not None:
            tmp = ckpt_path(seed) + ".tmp"
            with open(tmp, "wb") as f:
                pickle.dump({"config": config, "value": value, "convergence": conv}, f)
            os.replace(tmp, ckpt_path(seed))
        print(f"[{tag}] seed {seed} done", flush=True)
        return seed, value, conv

    if checkpoint_dir is not None:
        os.makedirs(checkpoint_dir, exist_ok=True)
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
