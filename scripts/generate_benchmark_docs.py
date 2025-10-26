#!/usr/bin/env python

import json
import os
from pathlib import Path


def main():
    """Generate the performance documentation.

    This script reads the latest benchmark results and generates a markdown table
    in the performance documentation.
    """
    print("Generating benchmark documentation...")

    # Find the latest benchmark file
    benchmark_dir = Path(".benchmarks")
    if not benchmark_dir.exists():
        print("Benchmark directory not found. Skipping.")
        return

    # Find the subdirectory (e.g., Darwin-CPython-3.12-64bit)
    subdirs = [d for d in benchmark_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("No benchmark subdirectory found. Skipping.")
        return
    latest_subdir = max(subdirs, key=os.path.getmtime)

    # Find the latest JSON file in the subdirectory
    json_files = list(latest_subdir.glob("*.json"))
    if not json_files:
        print("No benchmark JSON files found. Skipping.")
        return
    latest_file = max(json_files, key=os.path.getmtime)

    # Read the benchmark data
    with open(latest_file) as f:
        data = json.load(f)

    # Extract the relevant benchmark results
    results = {}
    for bench in data["benchmarks"]:
        name = bench["name"]
        mean_time = bench["stats"]["mean"] * 1e6  # convert to microseconds
        results[name] = mean_time

    # Generate the markdown table
    python_uniform = results.get("test_benchmark_energy_python_uniform", 0)
    numba_uniform = results.get("test_benchmark_energy_numba_uniform", 0)
    python_obs = results.get("test_benchmark_energy_python_obs", 0)
    numba_obs = results.get("test_benchmark_energy_numba_obs", 0)

    speedup_uniform = python_uniform / numba_uniform if numba_uniform else 0
    speedup_obs = python_obs / numba_obs if numba_obs else 0

    table = f"""| Method | Mode      | Mean Time (µs) | Speedup vs. Python |
| :------- | :-------- | :--------------- | :----------------- |
| Python | `uniform` | {python_uniform:.1f}           | 1x                 |
| Numba  | `uniform` | {numba_uniform:.1f}             | {speedup_uniform:.1f}x             |
| Python | `obs`     | {python_obs:.1f}           | 1x                 |
| Numba  | `obs`     | {numba_obs:.1f}             | {speedup_obs:.1f}x             |"""

    # Write the performance documentation
    doc = f"""---
title: "Performance"
---

This page presents some performance benchmarks for critical operations in `kmeanssa-ng`.

## Energy Calculation

The following table shows the performance of the k-means energy calculation for different implementations and modes. The benchmarks were run on a medium-sized graph with 100 nodes and 10 centers.

{table}

As you can see, the Numba-accelerated versions are significantly faster than the pure Python implementations.
"""

    with open("docs-src/performance.qmd", "w") as f:
        f.write(doc)

    print("Benchmark documentation generated successfully.")


if __name__ == "__main__":
    main()
