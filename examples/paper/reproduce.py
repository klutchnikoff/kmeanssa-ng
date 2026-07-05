"""Run every experiment, then build the tables and figures.

    python reproduce.py                    # publication numbers
    python reproduce.py --quick            # seconds-long smoke run of the pipeline
    python reproduce.py --paper            # full-resolution rate figure (~70 min extra)
    python reproduce.py --jobs -1          # fan the seed loops over every core
    python reproduce.py --seeds 100        # use 100 seeds (42..141) instead of each run's own
    python reproduce.py --no-rate          # skip the rate toy (rate.py) and its figures

The sphere experiment dominates the runtime (~20 min); grid and SBM take a
couple of minutes. The rate toy runs at a light setting (~15 min) by default;
``--paper`` reruns it at the resolution used for figure_5 in the article.

Seeds are independent, so ``--jobs N`` (or ``-1`` for all cores) runs them in
parallel; the results are identical to the sequential run whatever ``N`` is.
"""

import _env  # noqa: F401  -- pins BLAS threads; must precede numpy
import sys

import grid
import sbm
import sphere
import rate
import make_tables
import make_figures


def main(quick=False, paper=False, n_jobs=1, n_seeds=None, with_rate=True):
    # Seeds are independent; n_jobs>1 fans them out (n_jobs=-1 uses every core).
    # Results are identical whatever n_jobs is (see multistart.run_seeds).
    if quick:
        grid.run(seeds=(42, 43), n_runs=3, n_data=200, n_obs=200, n_jobs=n_jobs)
        sbm.run(seeds=(42, 43), n_runs=3, n_jobs=n_jobs)
        sphere.run(
            seeds=(42, 43), n_net=400, n_data=200, n_obs=200, n_runs=3, n_jobs=n_jobs
        )
    else:
        # --seeds N uses the N consecutive seeds 42..42+N-1 (default: each run's own).
        seed_kw = {} if n_seeds is None else {"seeds": tuple(range(42, 42 + n_seeds))}
        grid.run(n_jobs=n_jobs, **seed_kw)
        sbm.run(n_jobs=n_jobs, **seed_kw)
        sphere.run(n_jobs=n_jobs, **seed_kw)
    if with_rate:
        rate.run(n_runs=20, n_obs=3000) if quick else (
            rate.run(n_runs=700, n_obs=250000) if paper else rate.run()
        )
    make_tables.main()
    make_figures.figure_grid()
    make_figures.figure_sbm()
    make_figures.figure_sphere()
    make_figures.figure_convergence()
    if with_rate:
        make_figures.figure_rate()
        make_figures.figure_memory()


def _parse_int(argv, name, default):
    """Read ``--name N`` / ``--name=N`` from argv, else ``default``."""
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return int(argv[i + 1])
        if a.startswith(name + "="):
            return int(a.split("=", 1)[1])
    return default


if __name__ == "__main__":
    main(
        quick="--quick" in sys.argv,
        paper="--paper" in sys.argv,
        n_jobs=_parse_int(sys.argv, "--jobs", 1),
        n_seeds=_parse_int(sys.argv, "--seeds", None),
        with_rate="--no-rate" not in sys.argv,
    )
