#!/usr/bin/env python

import json
import os
from collections import defaultdict
from pathlib import Path


def format_time(seconds: float) -> tuple[float, str]:
    """Format time in microseconds for consistency."""
    return seconds * 1e6, "µs"


def categorize_benchmarks(results: dict) -> dict:
    """Categorize benchmarks by their type."""
    categories = defaultdict(list)

    for name, bench_data in results.items():
        if "robustification" in name:
            categories["Robustification Strategies"].append((name, bench_data))
        elif "energy" in name:
            categories["Energy Calculation"].append((name, bench_data))
        elif "precomputing" in name:
            categories["Graph Precomputing"].append((name, bench_data))
        elif "batch_distances" in name:
            categories["Batch Distance Computation"].append((name, bench_data))
        elif "kpp_initialization" in name:
            categories["K-means++ Initialization"].append((name, bench_data))
        elif "sa_" in name:
            categories["Simulated Annealing"].append((name, bench_data))
        else:
            categories["Other"].append((name, bench_data))

    return dict(categories)


def extract_benchmark_results(latest_file: Path) -> dict:
    """Extract and structure benchmark results from pytest-benchmark JSON.

    Returns a structured dict with all benchmark data organized by category.
    """
    with open(latest_file) as f:
        data = json.load(f)

    # Extract benchmark results with all stats
    results = {}
    for bench in data["benchmarks"]:
        name = bench["name"]
        results[name] = {
            "mean": bench["stats"]["mean"],
            "min": bench["stats"]["min"],
            "max": bench["stats"]["max"],
            "stddev": bench["stats"]["stddev"],
            "rounds": bench["stats"]["rounds"],
            "median": bench["stats"]["median"],
            "q1": bench["stats"]["q1"],
            "q3": bench["stats"]["q3"],
        }

    return results


def generate_structured_data(results: dict, source_file: str) -> dict:
    """Generate structured benchmark data for consumption by docs.

    Returns a dict with:
    - metadata: source file, timestamp
    - categories: organized benchmark data
    - comparisons: pre-computed speedups and ratios
    """
    categories = categorize_benchmarks(results)

    # Build structured output
    structured = {
        "metadata": {
            "source_file": source_file,
            "generated_at": os.path.getmtime(
                Path(".benchmarks") / "Darwin-CPython-3.12-64bit" / source_file
            )
            if (
                Path(".benchmarks") / "Darwin-CPython-3.12-64bit" / source_file
            ).exists()
            else None,
        },
        "categories": {},
        "comparisons": {},
    }

    # Add categorized benchmarks
    for category, benchmarks in categories.items():
        structured["categories"][category] = [
            {
                "name": name.replace("test_benchmark_", "").replace("_", " ").title(),
                "test_name": name,
                "mean_us": format_time(data["mean"])[0],
                "min_us": format_time(data["min"])[0],
                "max_us": format_time(data["max"])[0],
                "median_us": format_time(data["median"])[0],
                "q1_us": format_time(data["q1"])[0],
                "q3_us": format_time(data["q3"])[0],
                "rounds": data["rounds"],
            }
            for name, data in sorted(benchmarks, key=lambda x: x[1]["mean"])
        ]

    # Add energy calculation comparison
    if "Energy Calculation" in categories:
        py_uniform = results.get("test_benchmark_energy_python_uniform")
        nb_uniform = results.get("test_benchmark_energy_numba_uniform")
        py_obs = results.get("test_benchmark_energy_python_obs")
        obs_weight = results.get("test_benchmark_energy_numba_obs")

        if all([py_uniform, nb_uniform, py_obs, obs_weight]):
            structured["comparisons"]["energy_calculation"] = {
                "python_uniform_us": format_time(py_uniform["mean"])[0],
                "numba_uniform_us": format_time(nb_uniform["mean"])[0],
                "speedup_uniform": py_uniform["mean"] / nb_uniform["mean"],
                "python_obs_us": format_time(py_obs["mean"])[0],
                "numba_obs_us": format_time(obs_weight["mean"])[0],
                "speedup_obs": py_obs["mean"] / obs_weight["mean"],
            }

    # Add robustification comparison
    if "Robustification Strategies" in categories:
        mfn = results.get("test_benchmark_robustification_mostfrequentnode")
        me_uniform = results.get(
            "test_benchmark_robustification_minimize_energy_uniform"
        )
        me_obs = results.get("test_benchmark_robustification_minimize_energy_obs")
        me_uniform_py = results.get(
            "test_benchmark_robustification_minimize_energy_uniform_python"
        )
        me_obs_py = results.get(
            "test_benchmark_robustification_minimize_energy_obs_python"
        )

        if mfn:
            robust_comparison = {
                "baseline": {
                    "name": "MostFrequentNode",
                    "mean_us": format_time(mfn["mean"])[0],
                },
                "strategies": [],
            }

            if me_uniform:
                robust_comparison["strategies"].append(
                    {
                        "name": "MinimizeEnergy",
                        "implementation": "Numba",
                        "mode": "uniform",
                        "mean_us": format_time(me_uniform["mean"])[0],
                        "ratio_vs_baseline": me_uniform["mean"] / mfn["mean"],
                    }
                )

            if me_obs:
                robust_comparison["strategies"].append(
                    {
                        "name": "MinimizeEnergy",
                        "implementation": "Numba",
                        "mode": "obs",
                        "mean_us": format_time(me_obs["mean"])[0],
                        "ratio_vs_baseline": me_obs["mean"] / mfn["mean"],
                    }
                )

            if me_uniform_py:
                robust_comparison["strategies"].append(
                    {
                        "name": "MinimizeEnergy",
                        "implementation": "Python",
                        "mode": "uniform",
                        "mean_us": format_time(me_uniform_py["mean"])[0],
                        "ratio_vs_baseline": me_uniform_py["mean"] / mfn["mean"],
                    }
                )

            if me_obs_py:
                robust_comparison["strategies"].append(
                    {
                        "name": "MinimizeEnergy",
                        "implementation": "Python",
                        "mode": "obs",
                        "mean_us": format_time(me_obs_py["mean"])[0],
                        "ratio_vs_baseline": me_obs_py["mean"] / mfn["mean"],
                    }
                )

            # Calculate Numba speedups
            if me_uniform and me_uniform_py and me_obs and me_obs_py:
                speedup_uniform = me_uniform_py["mean"] / me_uniform["mean"]
                speedup_obs = me_obs_py["mean"] / me_obs["mean"]
                speedups = sorted([speedup_uniform, speedup_obs])
                robust_comparison["numba_speedup_range"] = {
                    "min": speedups[0],
                    "max": speedups[1],
                }

            structured["comparisons"]["robustification"] = robust_comparison

    return structured


def main():
    """Generate structured benchmark data as JSON.

    This script reads the latest benchmark results and generates a structured
    JSON file that can be consumed by documentation (Quarto, etc.).
    """
    print("Generating benchmark data...")

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

    print(f"Using benchmark data from: {latest_file.name}")

    # Extract and structure benchmark results
    results = extract_benchmark_results(latest_file)
    structured_data = generate_structured_data(results, latest_file.name)

    # Write structured data as JSON
    output_file = Path("docs-src/benchmark_data.json")
    with open(output_file, "w") as f:
        json.dump(structured_data, f, indent=2)

    print(f"✓ Structured benchmark data generated: {output_file}")
    print(f"  Total benchmarks: {len(results)}")
    print(f"  Categories: {len(structured_data['categories'])}")
    print(f"  Comparisons: {len(structured_data['comparisons'])}")


if __name__ == "__main__":
    main()
