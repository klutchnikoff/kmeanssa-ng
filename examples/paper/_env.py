"""Pin BLAS/OpenMP to a single thread -- import this before numpy.

Two reasons, both about the seed loop being parallelised (see ``multistart.run_seeds``):

* Reproducibility. scikit-learn's spectral baseline runs an eigendecomposition
  whose LAPACK rounding depends on the BLAS thread count, so its discrete labels
  can flip near cluster boundaries when the thread environment changes. Parallel
  workers use a different thread count than the sequential parent, which would
  make results depend on ``n_jobs``. One thread everywhere removes that.
* Speed. Under joblib each worker would otherwise spawn its own BLAS thread pool,
  oversubscribing the cores; one BLAS thread per worker is what we actually want.

Values are set with ``setdefault`` so an explicit environment override still wins.
"""

import os

for _var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_var, "1")
