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


def main():
    grid = load("grid_multi")
    sbm = load("sbm_multi")
    sph = load("sphere_multi")

    # On the sphere the reference SA is the graph route.
    sa_key = {"grid": "SA", "sbm": "SA", "sphere": "SA-graph"}
    meta = {
        "grid": ("Grid $10\\times10$", 100, 2, grid),
        "sbm": ("SBM", 100, 2, sbm),
        "sphere": ("Sphere $\\mathbb{S}^2$", sph["config"]["n_net"], 3, sph),
    }

    lines = []
    lines.append("% ---- Table 1: SA performance ----")
    lines.append("\\begin{tabular}{lccccccc}")
    lines.append("\\hline")
    lines.append(
        "Experiment & $|V|$ & $k$ & Runs & \\multicolumn{2}{c}{Final energy $U$} "
        "& \\multicolumn{2}{c}{ARI} \\\\"
    )
    lines.append("\\cline{5-6}\\cline{7-8}")
    lines.append(
        "& & & & Mean$\\pm\\sigma$ & Best & Mean$\\pm\\sigma$ & Best / Sel. \\\\"
    )
    lines.append("\\hline")
    for key in ("grid", "sbm", "sphere"):
        label, V, k, store = meta[key]
        ps = store["per_seed"]
        n_runs = store.get("n_runs", store.get("config", {}).get("n_runs"))
        n_seeds = len(store.get("seeds", store.get("config", {}).get("seeds", [])))
        em, es, eb = energy_stats(ps, sa_key[key])
        s = method_stats(ps, sa_key[key])
        lines.append(
            f"{label} & {V} & {k} & {n_runs}$\\times${n_seeds} & "
            f"${em:.2f}\\pm{es:.2f}$ & {eb:.2f} & "
            f"${fmt(s['mean'])}\\pm{fmt(s['std'])}$ & {fmt(s['best'])} / {fmt(s['sel_mean'])} \\\\"
        )
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("")

    methods = {
        "grid": ["SA", "k-medoids", "spectral"],
        "sbm": ["SA", "k-medoids", "spectral"],
        "sphere": ["SA-graph", "SA-sphere", "CLVQ", "k-medoids"],
    }
    pretty = {
        "SA": "SA (graph)",
        "SA-graph": "SA (graph)",
        "SA-sphere": "SA (sphere)",
        "k-medoids": "$k$-medoids",
        "spectral": "spectral",
        "CLVQ": "CLVQ",
    }
    lines.append("% ---- Table 2: comparison with baselines (ARI) ----")
    lines.append("\\begin{tabular}{llcc}")
    lines.append("\\hline")
    lines.append("Experiment & Method & ARI (sel., mean$\\pm\\sigma$) & ARI best \\\\")
    lines.append("\\hline")
    for key in ("grid", "sbm", "sphere"):
        label, V, k, store = meta[key]
        ps = store["per_seed"]
        for j, mth in enumerate(methods[key]):
            s = method_stats(ps, mth)
            exp_cell = label if j == 0 else ""
            lines.append(
                f"{exp_cell} & {pretty[mth]} & "
                f"${fmt(s['sel_mean'])}\\pm{fmt(s['sel_std'])}$ & {fmt(s['best'])} \\\\"
            )
        lines.append("\\hline")
    lines.append("\\end{tabular}")

    out = "\n".join(lines)
    print(out)
    open(f"{RESULTS}/tables.tex", "w").write(out + "\n")
    print("\n% written to results/tables.tex")


if __name__ == "__main__":
    main()
