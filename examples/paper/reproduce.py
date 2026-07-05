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
import rate
import make_tables
import make_figures
import timing_comparison

PAPER_SEEDS = tuple(range(42, 142))  # the 100 seeds behind the article's numbers


def main(quick=False, n_jobs=1, n_seeds=None):
    """Regenerate every figure and table of the article.

    With no arguments this is exactly the article's configuration: 100 seeds for
    grid/sbm/sphere and the full-resolution rate toy. ``--quick`` is a seconds-long
    smoke test; ``--jobs N`` (or -1) fans the seeds over cores with identical
    results; ``--seeds N`` overrides the seed count.
    """
    if quick:
        grid.run(seeds=(42, 43), n_runs=3, n_data=200, n_obs=200, n_jobs=n_jobs)
        sbm.run(seeds=(42, 43), n_runs=3, n_jobs=n_jobs)
        sphere.run(
            seeds=(42, 43), n_net=400, n_data=200, n_obs=200, n_runs=3, n_jobs=n_jobs
        )
        rate.run(n_runs=20, n_obs=3000)
    else:
        seeds = PAPER_SEEDS if n_seeds is None else tuple(range(42, 42 + n_seeds))
        grid.run(seeds=seeds, n_jobs=n_jobs)
        sbm.run(seeds=seeds, n_jobs=n_jobs)
        sphere.run(seeds=seeds, n_jobs=n_jobs)
        rate.run(n_runs=700, n_obs=250000)  # full paper resolution

    make_tables.main()  # -> results/table_{performance,comparison}.csv
    make_figures.figure_grid()  # figure_1
    make_figures.figure_sbm()  # figure_2
    make_figures.figure_sphere()  # figure_3
    make_figures.figure_convergence()  # figure_4
    make_figures.figure_rate()  # figure_5
    make_figures.figure_memory()  # figure_6
    if quick:
        timing_comparison.main(n_data=200, n_obs=200, n_net=400, n_runs=1)
    else:
        timing_comparison.main()  # -> results/timing_comparison.csv


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
