"""Run every experiment, then build the tables and figures.

    python reproduce.py           # EXACTLY the article (100 seeds, full rate) -- hours
    python reproduce.py --quick   # seconds-long smoke test of the whole pipeline
    python reproduce.py --jobs -1 # same as no flags, seeds fanned over every core
    python reproduce.py --seeds 20  # a lighter run (20 seeds instead of 100)

With no flags this regenerates every figure and table of the article. It is slow
(the sphere experiment at 100 seeds dominates, then the full-resolution rate toy);
``--jobs -1`` gives byte-identical results faster by running the independent seeds
in parallel. Outputs land in figures/ and results/.
"""

import _env  # noqa: F401  -- pins BLAS threads; must precede numpy
import sys

import grid
import sbm
import sphere
import rate_toy
import bolza
import make_tables
import make_figures
import make_timing
import geomstats_overhead

PAPER_SEEDS = tuple(range(42, 142))  # the 100 seeds behind the article's numbers


def main(quick=False, n_jobs=1, n_seeds=None):
    """Regenerate every figure and table of the article, in the paper's four parts.

    With no arguments this is exactly the article's configuration: 100 seeds for
    grid/sbm/sphere, the full-resolution rate toy, and the Bolza illustration.
    ``--quick`` is a seconds-long smoke test; ``--jobs N`` (or -1) fans the seeds
    over cores with identical results; ``--seeds N`` overrides the seed count.
    """
    seeds = PAPER_SEEDS if n_seeds is None else tuple(range(42, 42 + n_seeds))

    # ── Part 1 — rate-theorem toy graph → figures 5 and 6 ──
    rate_toy.run(n_runs=20, n_obs=3000) if quick else rate_toy.run(
        n_runs=700, n_obs=250000
    )
    make_figures.figure_rate()  # figure_5
    make_figures.figure_memory()  # figure_6

    # ── Part 2 — main experiments (grid, SBM, sphere): our method vs baselines ──
    if quick:
        grid.run(seeds=(42, 43), n_runs=3, n_data=200, n_obs=200, n_jobs=n_jobs)
        sbm.run(seeds=(42, 43), n_runs=3, n_jobs=n_jobs)
        sphere.run(
            seeds=(42, 43), n_net=400, n_data=200, n_obs=200, n_runs=3, n_jobs=n_jobs
        )
    else:
        grid.run(seeds=seeds, n_jobs=n_jobs)
        sbm.run(seeds=seeds, n_jobs=n_jobs)
        sphere.run(seeds=seeds, n_jobs=n_jobs)
    make_tables.main()  # ARI: results/table_{performance,comparison}.csv
    make_figures.figure_grid()  # figure_1
    make_figures.figure_sbm()  # figure_2
    make_figures.figure_sphere()  # figure_3
    make_figures.figure_convergence()  # figure_4 (auxiliary convergence diagnostic)
    make_timing.main(measure_net=not quick)  # time: results/table_timing.csv

    # ── Part 3 — closed-form vs geomstats geometry → results/geomstats_overhead.csv ──
    if quick:
        geomstats_overhead.main(n_data=200, n_obs=200, n_net=400, n_runs=1)
    else:
        geomstats_overhead.main()

    # ── Part 4 — Bolza surface (genus 2, curvature -1) → figure_bolza ──
    bolza.run(n_runs=2, n_data=200) if quick else bolza.run()
    make_figures.figure_bolza()


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
        n_jobs=_parse_int(sys.argv, "--jobs", 1),
        n_seeds=_parse_int(sys.argv, "--seeds", None),
    )
