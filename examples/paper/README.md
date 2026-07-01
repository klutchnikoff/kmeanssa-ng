# Paper experiments

Reproduction code for the empirical section of *Online k-means Clustering on
Metric Graphs and Geodesic Spaces* (Brécheteau, Gavra, Klutchnikoff).

Three experiments compare the simulated-annealing clustering of `kmeanssa-ng`
against baseline methods (weighted k-medoids, spectral clustering, CLVQ):

| Experiment | Space | Baselines |
|---|---|---|
| Grid $10\times10$ | metric graph | k-medoids, spectral |
| Stochastic block model | weighted metric graph | k-medoids, spectral |
| Sphere $\mathbb{S}^2$ | geodesic space via $\varepsilon$-net graph | k-medoids, CLVQ |

## Layout

```
common.py        shared helpers (labels, energy, tracking SA, sphere geometry)
baselines.py     k-medoids, spectral, CLVQ
experiments.py   grid + SBM  -> results/grid_multi.pkl, results/sbm_multi.pkl
sphere.py        sphere      -> results/sphere_multi.pkl
make_tables.py   results/*.pkl -> results/tables.tex
make_figures.py  results/*.pkl -> figures/figure_{1,2,3}.pdf
reproduce.py     run everything end to end
```

Raw per-run results are cached as pickles so tables and figures can be rebuilt
without re-running the (slow) experiments.

## Running

```bash
pip install "kmeanssa-ng>=0.8.0" scikit-learn matplotlib
python reproduce.py            # experiments + tables + figures
# or step by step:
python experiments.py          # grid + SBM
python sphere.py               # sphere (~20 min)
python make_tables.py
python make_figures.py
```

Every experiment is seeded (`seeds = 42..46` by default), so results are
reproducible across machines.
