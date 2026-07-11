---
title: Benchmark Developer Guide
---


This document provides technical instructions for developers on how to
run, manage, and contribute to the performance benchmarks for
`kmeanssa-ng`.

## Running Benchmarks

### Quick benchmarks (fast tests only)

``` bash
pdm run benchmark
```

This runs all benchmarks except those marked as `@pytest.mark.slow`.

### All benchmarks (including slow tests)

``` bash
pdm run benchmark-all
```

### Compare with a baseline

After making changes, compare performance with the saved baseline:

``` bash
pdm run benchmark-compare
```

This will show a comparison table highlighting performance changes.

## Saving & Comparing Baselines

### Save current performance as a new baseline

``` bash
pdm run pytest tests/test_benchmarks.py --benchmark-only --benchmark-save=my_baseline
```

### Compare against a specific baseline

``` bash
pdm run pytest tests/test_benchmarks.py --benchmark-only --benchmark-compare=my_baseline
```

## CI Integration

Benchmarks are not run in CI by default due to performance variability
on shared runners. To enable them, you can add a job like the following
to `.gitlab-ci.yml`:

``` yaml
benchmark:
  stage: test
  script:
    - pdm run benchmark
  allow_failure: true  # Don't fail CI on performance variance
```

## Adding New Benchmarks

1.  Add a new test function to `tests/test_benchmarks.py`.
2.  Use the `benchmark` fixture provided by `pytest-benchmark`.
3.  Mark slow tests with `@pytest.mark.slow`.
4.  Ensure the new benchmark is covered by the documentation if it
    represents a new feature.

For more details, refer to the [pytest-benchmark
documentation](https://pytest-benchmark.readthedocs.io/).
