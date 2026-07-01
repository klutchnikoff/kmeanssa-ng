"""Build the LaTeX tables from the saved pickles (no recomputation).

  Table 1: SA performance per experiment (|V|, k, runs, final energy, ARI).
  Table 2: comparison with baselines (ARI selected, mean +/- std over seeds).

Writes results/tables.tex and prints it.
"""

import pickle

import numpy as np

RESULTS = "results"


def load(name):
    return pickle.load(open(f"{RESULTS}/{name}.pkl", "rb"))


def method_stats(per_seed, method):
    """Pooled mean/std/best over all runs; selected = per-seed (argmin energy), averaged."""
    aris_all, sel = [], []
    for sres in per_seed.values():
        m = sres["methods"][method]
        aris_all.append(np.asarray(m["aris"]))
        en = m["energies"]
        if en is not None and len(en):
            sel.append(float(m["aris"][int(np.argmin(en))]))
        else:  # spectral has no energy -> report the mean as "selected"
            sel.append(float(np.mean(m["aris"])))
    a = np.concatenate(aris_all)
    return {
        "mean": a.mean(),
        "std": a.std(),
        "best": a.max(),
        "sel_mean": float(np.mean(sel)),
        "sel_std": float(np.std(sel)),
    }


def energy_stats(per_seed, method="SA"):
    e = np.concatenate(
        [np.asarray(s["methods"][method]["energies"]) for s in per_seed.values()]
    )
    return e.mean(), e.std(), e.min()


def fmt(x):
    return f"{x:.3f}"


# On the sphere the reference SA is the graph route.
SA_KEY = {"grid": "SA", "sbm": "SA", "sphere": "SA-graph"}
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


def _performance_table(meta):
    """Table 1: per-experiment SA final energy and ARI."""
    lines = [
        "% ---- Table 1: SA performance ----",
        "\\begin{tabular}{lccccccc}",
        "\\hline",
        "Experiment & $|V|$ & $k$ & Runs & \\multicolumn{2}{c}{Final energy $U$} "
        "& \\multicolumn{2}{c}{ARI} \\\\",
        "\\cline{5-6}\\cline{7-8}",
        "& & & & Mean$\\pm\\sigma$ & Best & Mean$\\pm\\sigma$ & Best / Sel. \\\\",
        "\\hline",
    ]
    for key in ("grid", "sbm", "sphere"):
        label, V, k, store = meta[key]
        ps = store["per_seed"]
        n_runs = store.get("n_runs", store.get("config", {}).get("n_runs"))
        n_seeds = len(store.get("seeds", store.get("config", {}).get("seeds", [])))
        em, es, eb = energy_stats(ps, SA_KEY[key])
        s = method_stats(ps, SA_KEY[key])
        lines.append(
            f"{label} & {V} & {k} & {n_runs}$\\times${n_seeds} & "
            f"${em:.2f}\\pm{es:.2f}$ & {eb:.2f} & "
            f"${fmt(s['mean'])}\\pm{fmt(s['std'])}$ & {fmt(s['best'])} / {fmt(s['sel_mean'])} \\\\"
        )
    return lines + ["\\hline", "\\end{tabular}", ""]


def _comparison_table(meta):
    """Table 2: ARI comparison against the baselines."""
    lines = [
        "% ---- Table 2: comparison with baselines (ARI) ----",
        "\\begin{tabular}{llcc}",
        "\\hline",
        "Experiment & Method & ARI (sel., mean$\\pm\\sigma$) & ARI best \\\\",
        "\\hline",
    ]
    for key in ("grid", "sbm", "sphere"):
        label, _V, _k, store = meta[key]
        ps = store["per_seed"]
        for j, method in enumerate(METHODS[key]):
            s = method_stats(ps, method)
            exp_cell = label if j == 0 else ""
            lines.append(
                f"{exp_cell} & {PRETTY[method]} & "
                f"${fmt(s['sel_mean'])}\\pm{fmt(s['sel_std'])}$ & {fmt(s['best'])} \\\\"
            )
        lines.append("\\hline")
    return lines + ["\\end{tabular}"]


def main():
    sphere_store = load("sphere_multi")
    meta = {
        "grid": ("Grid $10\\times10$", 100, 2, load("grid_multi")),
        "sbm": ("SBM", 100, 2, load("sbm_multi")),
        "sphere": (
            "Sphere $\\mathbb{S}^2$",
            sphere_store["config"]["n_net"],
            3,
            sphere_store,
        ),
    }
    out = "\n".join(_performance_table(meta) + _comparison_table(meta))
    print(out)
    open(f"{RESULTS}/tables.tex", "w").write(out + "\n")
    print("\n% written to results/tables.tex")


if __name__ == "__main__":
    main()
