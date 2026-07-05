# Paper experiments

Reproduction code for the empirical section of *Online k-means Clustering on
Metric Graphs and Geodesic Spaces* (Brécheteau, Gavra, Klutchnikoff).

**`python reproduce.py` regenerates every figure and table of the article in one
command** (no flags = the article's exact configuration).

## What it produces

Three benchmarks compare the simulated-annealing clustering of `kmeanssa-ng`
against baseline methods (weighted k-medoids, spectral clustering, CLVQ), plus a
toy graph that makes the convergence-rate theorem visible:

| Experiment | Space | Baselines |
|---|---|---|
| Grid $10\times10$ | metric graph | k-medoids, spectral |
| Stochastic block model | weighted metric graph | k-medoids, spectral |
| Sphere $\mathbb{S}^2$ | geodesic space via $\varepsilon$-net graph | k-medoids, CLVQ |
| Rate toy (3-node path) | shallow metric graph | — |

Figures land in `figures/`, tables in `results/`:

| Output | Produced by | Content |
|---|---|---|
| `figure_1` | `figure_grid` | grid partition (reference vs SA) |
| `figure_2` | `figure_sbm` | SBM partition |
| `figure_3` | `figure_sphere` | sphere partition |
| `figure_4` | `figure_convergence` | energy along the annealing time (auxiliary diagnostic; not necessarily used in the article) |
| `figure_5` | `figure_rate` | rate-theorem exceedance (log--log) + energy trajectories |
| `figure_6` | `figure_memory` | memory-augmented estimator (partial / full memory) |
| `table_performance.csv` | `make_tables` | per-experiment energy/ARI summary |
| `table_comparison.csv` | `make_tables` | ARI vs baselines (mean$\pm\sigma$, best) |
| `sphere_timing.csv` | `sphere.run` | per-method wall time on the sphere (mean$\pm\sigma$ over seeds) |
| `timing_comparison.csv` | `timing_comparison` | per-method time with vs without closed-form geometry (justifies the $\varepsilon$-net) |

## Layout

```
_env.py           pins BLAS to one thread (imported first, before numpy)
calibration.py    temperature calibration: the critical depth c*
multistart.py     the seeded multi-start SA loop + parallel seed driver
baselines.py      k-medoids, spectral, CLVQ
grid.py sbm.py sphere.py   the three benchmarks -> results/*_multi.pkl
rate.py           rate-theorem toy               -> results/rate.pkl
make_tables.py    results/*.pkl -> results/table_{performance,comparison}.csv
make_figures.py   results/*.pkl -> figures/figure_{1..6}.pdf
timing_comparison.py  standalone method-timing benchmark -> results/timing_comparison.csv
reproduce.py      regenerate everything in the article
data/             frozen sphere epsilon-net definition (committed, ~1 MB)
cache/            its 191 MB pairwise-distance matrix (rebuilt on demand, gitignored)
```

Raw per-run results are cached as pickles so tables and figures can be rebuilt
without re-running the (slow) experiments. The sphere epsilon-net is frozen under
`data/`: it is built once (~3 min) and reloaded in ~0.2 s thereafter.

## Running

```bash
pip install "kmeanssa-ng>=0.8.0" scikit-learn matplotlib

python reproduce.py            # EXACTLY the article (100 seeds, full rate) -- a few hours
python reproduce.py --jobs -1  # same, byte-identical, seeds fanned over every core
python reproduce.py --quick    # seconds-long smoke test of the whole pipeline
python reproduce.py --seeds 20 # a lighter run (20 seeds instead of 100)
```

With no flags, `reproduce.py` runs the article's configuration: **100 seeds**
(42..141) for grid/sbm/sphere and the full-resolution rate toy (700 runs). The
sphere experiment dominates the runtime (a few hours sequential); `--jobs -1`
runs the independent seeds in parallel for the same results. Individual scripts
(`python sphere.py`, etc.) can also be run on their own defaults.

Reproducibility: every experiment is seeded, and BLAS is pinned to one thread, so
results are byte-identical across machines and whatever `--jobs` is. The
per-method wall times in `sphere_timing.csv` are clean only from a **sequential**
run (`--jobs 1`, the default); under parallelism they include worker contention.

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
