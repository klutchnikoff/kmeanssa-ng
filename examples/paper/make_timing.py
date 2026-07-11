"""Aggregate per-method wall times into results/table_timing.csv.

Reads the experiment pickles, averages the per-seed timings, appends the one-time
setup costs (the k-medoids distance matrix where it is built once, the sphere
epsilon-net build), and writes a long-format CSV.

Wall times are meaningful only from a **sequential** run (``n_jobs=1``); under
parallelism per-seed times include worker contention, so this refuses to write.

    python make_timing.py
"""

import _env  # noqa: F401  -- pins BLAS threads; must precede numpy
import csv
import os
import pickle
import time

import numpy as np

EXPERIMENTS = [
    ("Grid $10\\times10$", "results/grid_multi.pkl"),
    ("SBM", "results/sbm_multi.pkl"),
    ("Sphere $\\mathbb{S}^2$", "results/sphere_multi.pkl"),
]
PKL = "results/table_timing.csv"


def _epsilon_net_build_seconds(n_net=5000):
    """Cold-build the sphere epsilon-net (without saving) and time it."""
    import sphere

    t = time.perf_counter()
    sphere._construct_net(n_net)  # builds fresh; leaves the frozen data/cache intact
    return time.perf_counter() - t


def main(measure_net=True):
    stores = []
    for name, path in EXPERIMENTS:
        if os.path.exists(path):
            stores.append((name, pickle.load(open(path, "rb"))))
        else:
            print(f"skip {name}: {path} missing", flush=True)

    jobs = {store.get("n_jobs", 1) for _, store in stores}
    if jobs != {1}:
        print(
            "WARNING: an experiment ran with n_jobs != 1; per-seed wall times "
            f"include worker contention. Not writing {PKL} -- re-run sequentially.",
            flush=True,
        )
        # A timing table from an earlier campaign must not survive next to the
        # fresh CSVs: it would silently describe another run of the pipeline.
        if os.path.exists(PKL):
            os.remove(PKL)
            print(f"removed stale {PKL} (from a previous campaign)", flush=True)
        return

    rows = []  # (experiment, item, kind, mean_s, std_s, n_seeds)
    for name, store in stores:
        # The tracked seed (the first of the campaign) runs the convergence
        # diagnostic, which recomputes the full energy after every observation
        # inside the SA timing bracket; that inflates only its SA row. Exclude
        # it so the reported per-seed times measure the algorithm alone.
        seeds = store.get("seeds") or store.get("config", {}).get("seeds", [])
        tracked = seeds[0] if seeds else None
        timings = [
            v["timings"]
            for s, v in store["per_seed"].items()
            if "timings" in v and s != tracked
        ]
        n = len(timings)
        for m in timings[0] if timings else []:
            ts = np.array([t[m] for t in timings])
            rows.append(
                (name, m, "per_seed", ts.mean(), ts.std(ddof=1 if n > 1 else 0), n)
            )
        for item, secs in store.get("setup", {}).items():
            rows.append((name, item, "one_time", secs, 0.0, 1))

    if measure_net:
        print("measuring epsilon-net cold build (~3 min)...", flush=True)
        rows.append(
            (
                "Sphere $\\mathbb{S}^2$",
                "epsilon-net build",
                "one_time",
                _epsilon_net_build_seconds(),
                0.0,
                1,
            )
        )

    os.makedirs("results", exist_ok=True)
    with open(PKL, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["experiment", "item", "kind", "time_mean_s", "time_std_s", "n_seeds"]
        )
        for name, item, kind, mean, std, n in rows:
            writer.writerow([name, item, kind, f"{mean:.4f}", f"{std:.4f}", n])
    print(f"wrote {PKL}", flush=True)


if __name__ == "__main__":
    main()
