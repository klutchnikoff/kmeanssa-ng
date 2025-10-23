# Performance Benchmarks

This directory contains performance benchmarks for `kmeanssa-ng` to detect performance regressions.

## Running Benchmarks

### Quick benchmarks (fast tests only)

```bash
pdm run benchmark
```

This runs all benchmarks except those marked as `@pytest.mark.slow` (~7 seconds).

### All benchmarks (including slow tests)

```bash
pdm run benchmark-all
```

This includes larger graph benchmarks (~20-30 seconds).

### Compare with baseline

After making changes, compare performance with the saved baseline:

```bash
pdm run benchmark-compare
```

This will show a comparison table highlighting performance changes.

## Benchmark Coverage

The benchmarks test the following critical operations:

### 1. Graph Precomputing (`precomputing()`)
- **Small graph** (40 nodes): ~65 µs
- **Medium graph** (100 nodes): ~370 µs
- Tests all-pairs shortest path caching (Dijkstra)

### 2. Batch Distance Computation (`batch_distances_from_centers()`)
- **5 centers**: ~8 µs
- **10 centers**: ~8 µs
- Tests Numba-accelerated distance computation

### 3. K-means++ Initialization (`sample_kpp_centers()`)
- **Small graph** (k=3): ~280 µs
- **Medium graph** (k=5): ~1,285 µs
- Tests initial center selection algorithm

### 4. Simulated Annealing (Interleaved)
- **Small** (50 points, k=2): ~2.2 ms
- **Medium** (150 points, k=3): ~7-10 ms (marked slow)
- Tests main clustering algorithm

### 5. Simulated Annealing (Sequential)
- **Small** (50 points, k=2): ~2.4 ms
- **Medium** (150 points, k=3): ~7-10 ms (marked slow)
- Tests alternative algorithm variant

## Interpreting Results

The output shows:
- **Min/Max/Mean**: Time statistics in microseconds (µs)
- **StdDev**: Standard deviation (lower is more stable)
- **Median**: Middle value (less affected by outliers)
- **OPS**: Operations per second (higher is better)
- **Outliers**: Number of unusual measurements

### What to watch for

- **Mean** should stay within ±10% between runs
- Large **StdDev** indicates unstable performance
- **Outliers** > 10% suggests inconsistent behavior

## Saving & Comparing Baselines

### Save current performance as baseline

```bash
pdm run pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=my_baseline
```

### Compare against a specific baseline

```bash
pdm run pytest tests/test_benchmarks.py --benchmark-only --benchmark-compare=my_baseline
```

### View all saved baselines

```bash
ls .benchmarks/Darwin-CPython-3.12-64bit/
```

## CI Integration

Benchmarks are NOT run in CI by default (too variable). To add them:

```yaml
# .gitlab-ci.yml
benchmark:
  stage: test
  script:
    - pdm run benchmark
  allow_failure: true  # Don't fail CI on performance variance
```

## Adding New Benchmarks

1. Add test to `test_benchmarks.py`
2. Use `benchmark` fixture: `def test_my_benchmark(benchmark): ...`
3. Mark slow tests: `@pytest.mark.slow`
4. Document expected performance in this file

## References

- pytest-benchmark docs: https://pytest-benchmark.readthedocs.io/
- Best practices: Use warmup runs, avoid I/O, focus on hot paths
