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
_env.py          pins BLAS to one thread (imported first, before numpy)
calibration.py   temperature calibration: the critical depth c*
multistart.py    the seeded multi-start SA loop + parallel seed driver
baselines.py     k-medoids, spectral, CLVQ
grid.py          grid experiment    -> results/grid_multi.pkl
sbm.py           SBM experiment     -> results/sbm_multi.pkl
sphere.py        sphere experiment  -> results/sphere_multi.pkl
make_tables.py   results/*.pkl -> results/table_{performance,comparison}.csv
make_figures.py  results/*.pkl -> figures/figure_{1,2,3,5,6}.pdf
reproduce.py     run everything end to end
timing_comparison.py  standalone: per-method wall time (SA-graph vs SA on the
                 sphere with/without closed forms, CLVQ, k-medoids) ->
                 results/timing_comparison.csv; justifies the epsilon-net
data/            frozen sphere epsilon-net definition (committed, ~1 MB)
cache/           its 191 MB pairwise-distance matrix (rebuilt on demand, gitignored)
```

Raw per-run results are cached as pickles so tables and figures can be rebuilt
without re-running the (slow) experiments. The sphere epsilon-net is frozen under
`data/`: it is built once (~3 min) and reloaded in ~0.2 s thereafter.

## Running

```bash
pip install "kmeanssa-ng>=0.8.0" scikit-learn matplotlib
python reproduce.py            # experiments + tables + figures
python reproduce.py --jobs -1  # same, seeds fanned over every core
# or step by step:
python grid.py
python sbm.py
python sphere.py               # ~20 min (first run also builds the net)
python make_tables.py
python make_figures.py
```

Every experiment is seeded (`seeds = 42..46` by default), so results are
reproducible across machines -- and, because BLAS is pinned to one thread, they
are identical whether the seeds run sequentially or in parallel (`--jobs`).

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

Or, with `pgfplotstable` driving `siunitx` `S` columns (automatic rounding and
decimal alignment; `string type` hands the raw CSV number to `siunitx`):

```latex
\usepackage{pgfplotstable,siunitx,booktabs}
\sisetup{round-mode=places, round-precision=3}

% Comparison with the baselines (results/table_comparison.csv)
\pgfplotstabletypeset[
  col sep=comma,
  columns={experiment, method, ari_sel_mean, ari_sel_std, ari_best},
  columns/experiment/.style={string type, column name=Experiment},
  columns/method/.style={string type, column name=Method},
  columns/ari_sel_mean/.style={string type, column type={S}, column name={ARI (sel.)}},
  columns/ari_sel_std/.style={string type, column type={S}, column name={$\sigma$}},
  columns/ari_best/.style={string type, column type={S}, column name={ARI best}},
  every head row/.style={before row=\toprule, after row=\midrule},
  every last row/.style={after row=\bottomrule},
]{results/table_comparison.csv}
```

Either way, the precision, columns and layout are changed in LaTeX without
re-running Python.
