"""Run every experiment, then build the tables and figures.

    python reproduce.py

The sphere experiment dominates the runtime (~20 min); grid and SBM take a
couple of minutes. Pass --quick for a small smoke run that exercises the whole
pipeline in seconds without producing publication-grade numbers.
"""

import sys

import grid
import sbm
import sphere
import make_tables
import make_figures


def main(quick=False):
    if quick:
        grid.run(seeds=(42, 43), n_runs=3, n_data=200, n_obs=200)
        sbm.run(seeds=(42, 43), n_runs=3)
        sphere.run(seeds=(42, 43), n_net=400, n_data=200, n_obs=200, n_runs=3)
    else:
        grid.run()
        sbm.run()
        sphere.run()
    make_tables.main()
    make_figures.figure_grid()
    make_figures.figure_sbm()
    make_figures.figure_sphere()
    make_figures.figure_convergence()


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
