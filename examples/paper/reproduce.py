"""Run every experiment, then build the tables and figures.

    python reproduce.py            # publication numbers (sphere ~20 min)
    python reproduce.py --quick    # seconds-long smoke run of the whole pipeline
    python reproduce.py --paper    # full-resolution rate figure (~70 min extra)

The sphere experiment dominates the runtime (~20 min); grid and SBM take a
couple of minutes. The rate toy runs at a light setting (~15 min) by default;
``--paper`` reruns it at the resolution used for figure_5 in the article.
"""

import sys

import grid
import sbm
import sphere
import rate
import make_tables
import make_figures


def main(quick=False, paper=False):
    if quick:
        grid.run(seeds=(42, 43), n_runs=3, n_data=200, n_obs=200)
        sbm.run(seeds=(42, 43), n_runs=3)
        sphere.run(seeds=(42, 43), n_net=400, n_data=200, n_obs=200, n_runs=3)
        rate.run(n_runs=20, n_obs=3000)
    else:
        grid.run()
        sbm.run()
        sphere.run()
        rate.run(n_runs=700, n_obs=250000) if paper else rate.run()
    make_tables.main()
    make_figures.figure_grid()
    make_figures.figure_sbm()
    make_figures.figure_sphere()
    make_figures.figure_convergence()
    make_figures.figure_rate()
    make_figures.figure_memory()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv, paper="--paper" in sys.argv)
