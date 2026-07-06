"""Compute the experiment statistics and write them as CSV.

Presentation (the LaTeX table model) lives in the article, not here: this module
only emits the numbers at full precision. See the README for a \\csvreader
snippet that typesets these files.

  results/table_performance.csv   (one row per experiment: energy + ARI of the SA)
  results/table_comparison.csv    (one row per experiment x method: ARI)
"""

import csv
import pickle

import numpy as np

RESULTS = "results"

# On the sphere the reference SA is the graph route.
SA_KEY = {"grid": "SA", "sbm": "SA", "sphere": "SA-graph"}
LABEL = {"grid": "Grid $10\\times10$", "sbm": "SBM", "sphere": "Sphere $\\mathbb{S}^2$"}
V = {"grid": 100, "sbm": 100}  # sphere's |V| comes from its config
METHODS = {
    "grid": ["SA", "k-medoids", "spectral"],
    "sbm": ["SA", "k-medoids", "spectral"],
    "sphere": ["SA-graph", "SA-sphere", "CLVQ", "k-medoids"],
}
PRETTY = {
    "SA": "SA (graph)",
    "SA-graph": "SA (graph)",
    "SA-sphere": "SA (sphere)",
    "k-medoids": "$k$-medoids",
    "spectral": "spectral",
    "CLVQ": "CLVQ",
}


def load(name):
    return pickle.load(open(f"{RESULTS}/{name}.pkl", "rb"))


def method_stats(per_seed, method):
    """Pooled mean/std/best over all runs; selected = per-seed (argmin energy), averaged."""
    aris_all, sel = [], []
    for sres in per_seed.values():
        m = sres["methods"][method]
        aris_all.append(np.asarray(m["aris"]))
        energies = m["energies"]
        if energies is not None and len(energies):
            sel.append(float(m["aris"][int(np.argmin(energies))]))
        else:  # spectral has no energy -> report the mean as "selected"
            sel.append(float(np.mean(m["aris"])))
    aris = np.concatenate(aris_all)
    return {
        "mean": aris.mean(),
        "std": aris.std(),
        "best": aris.max(),
        "sel_mean": float(np.mean(sel)),
        "sel_std": float(np.std(sel)),
    }


def energy_stats(per_seed, method):
    energies = np.concatenate(
        [np.asarray(s["methods"][method]["energies"]) for s in per_seed.values()]
    )
    return energies.mean(), energies.std(), energies.min()


def _config(store, field):
    return store.get(field, store.get("config", {}).get(field))


def _check_same_campaign(stores):
    """Refuse to mix experiment pickles coming from different campaigns.

    The experiment scripts overwrite the same result files whatever their
    arguments, so a stray standalone run (e.g. the 5-seed default) would
    silently replace one experiment of a 100-seed campaign. Seed sets are the
    campaign signature: they must match across the three pickles.
    """
    seed_sets = {key: list(_config(store, "seeds")) for key, store in stores.items()}
    reference = next(iter(seed_sets.values()))
    mismatched = {k: v for k, v in seed_sets.items() if v != reference}
    if mismatched:
        detail = ", ".join(
            f"{k}: {len(v)} seeds {v[:3]}..." for k, v in seed_sets.items()
        )
        raise SystemExit(
            f"experiment pickles come from different campaigns ({detail}); "
            "re-run the missing experiments (see reproduce.py) before writing tables"
        )


def performance_rows(stores):
    """One row per experiment: SA final energy and ARI statistics."""
    rows = []
    for key in ("grid", "sbm", "sphere"):
        store = stores[key]
        per_seed = store["per_seed"]
        em, es, eb = energy_stats(per_seed, SA_KEY[key])
        ari = method_stats(per_seed, SA_KEY[key])
        rows.append(
            {
                "experiment": LABEL[key],
                "V": V.get(key, _config(store, "n_net")),
                "k": 3 if key == "sphere" else 2,
                "n_runs": _config(store, "n_runs"),
                "n_seeds": len(_config(store, "seeds")),
                "energy_mean": em,
                "energy_std": es,
                "energy_best": eb,
                "ari_mean": ari["mean"],
                "ari_std": ari["std"],
                "ari_best": ari["best"],
                "ari_sel": ari["sel_mean"],
            }
        )
    return rows


def comparison_rows(stores):
    """One row per (experiment, method): selected/best ARI against the baselines."""
    rows = []
    for key in ("grid", "sbm", "sphere"):
        per_seed = stores[key]["per_seed"]
        for method in METHODS[key]:
            ari = method_stats(per_seed, method)
            rows.append(
                {
                    "experiment": LABEL[key],
                    "method": PRETTY[method],
                    "ari_sel_mean": ari["sel_mean"],
                    "ari_sel_std": ari["sel_std"],
                    "ari_best": ari["best"],
                }
            )
    return rows


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    stores = {
        "grid": load("grid_multi"),
        "sbm": load("sbm_multi"),
        "sphere": load("sphere_multi"),
    }
    _check_same_campaign(stores)
    write_csv(f"{RESULTS}/table_performance.csv", performance_rows(stores))
    write_csv(f"{RESULTS}/table_comparison.csv", comparison_rows(stores))
    print(f"wrote {RESULTS}/table_performance.csv and {RESULTS}/table_comparison.csv")


if __name__ == "__main__":
    main()
