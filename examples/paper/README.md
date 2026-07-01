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
calibration.py   temperature calibration: the critical depth c*
multistart.py    the seeded multi-start SA loop, shared by every experiment
baselines.py     k-medoids, spectral, CLVQ
grid.py          grid experiment    -> results/grid_multi.pkl
sbm.py           SBM experiment     -> results/sbm_multi.pkl
sphere.py        sphere experiment  -> results/sphere_multi.pkl
make_tables.py   results/*.pkl -> results/table_{performance,comparison}.csv
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
python grid.py
python sbm.py
python sphere.py               # ~20 min
python make_tables.py
python make_figures.py
```

Every experiment is seeded (`seeds = 42..46` by default), so results are
reproducible across machines.

## Typesetting the tables

`make_tables.py` only emits the numbers, as CSV at full precision — the table
model stays in the article. With `csvsimple` and `siunitx`, rounding and layout
are controlled entirely in LaTeX:

```latex
\usepackage{csvsimple,siunitx}
\sisetup{round-mode=places, round-precision=3}

% Comparison with the baselines (results/table_comparison.csv)
\begin{tabular}{llcc}
\hline
Experiment & Method & ARI (sel., mean$\pm\sigma$) & ARI best \\
\hline
\csvreader[late after line=\\]{results/table_comparison.csv}%
  {experiment=\exper, method=\meth, ari_sel_mean=\selm,
   ari_sel_std=\sels, ari_best=\best}%
  {\exper & \meth & $\num{\selm} \pm \num{\sels}$ & \num{\best}}
\hline
\end{tabular}
```

Change the precision, columns or layout without touching Python. (`pgfplotstable`
is an alternative if you prefer automatic numeric formatting.)
