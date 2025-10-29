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


def generate_category_table(benchmarks: list) -> str:
    """Generate a markdown table for a category of benchmarks."""
    if not benchmarks:
        return ""

    # Sort by mean time
    benchmarks = sorted(benchmarks, key=lambda x: x[1]["mean"])

    lines = [
        "| Benchmark | Mean Time | Min Time | Max Time | Rounds |",
        "| :-------- | :-------- | :------- | :------- | -----: |",
    ]

    for name, data in benchmarks:
        # Clean up the test name for display
        display_name = name.replace("test_benchmark_", "").replace("_", " ").title()

        mean_val, mean_unit = format_time(data["mean"])
        min_val, min_unit = format_time(data["min"])
        max_val, max_unit = format_time(data["max"])
        rounds = data["rounds"]

        lines.append(
            f"| {display_name} | {mean_val:.2f} {mean_unit} | "
            f"{min_val:.2f} {min_unit} | {max_val:.2f} {max_unit} | {rounds} |"
        )

    return "\n".join(lines)


def generate_energy_comparison_table(results: dict) -> str:
    """Generate a comparison table for energy benchmarks (Python vs Numba)."""
    python_uniform = results.get("test_benchmark_energy_python_uniform")
    numba_uniform = results.get("test_benchmark_energy_numba_uniform")
    python_obs = results.get("test_benchmark_energy_python_obs")
    numba_obs = results.get("test_benchmark_energy_numba_obs")

    if not all([python_uniform, numba_uniform, python_obs, numba_obs]):
        return ""

    speedup_uniform = python_uniform["mean"] / numba_uniform["mean"]
    speedup_obs = python_obs["mean"] / numba_obs["mean"]

    py_uni_val, py_uni_unit = format_time(python_uniform["mean"])
    nb_uni_val, nb_uni_unit = format_time(numba_uniform["mean"])
    py_obs_val, py_obs_unit = format_time(python_obs["mean"])
    nb_obs_val, nb_obs_unit = format_time(numba_obs["mean"])

    return f"""### Python vs Numba Comparison

The following table compares the performance of pure Python vs Numba-accelerated implementations:

| Method | Mode | Mean Time | Speedup vs. Python |
| :----- | :--- | :-------- | :----------------- |
| Python | `uniform` | {py_uni_val:.2f} {py_uni_unit} | 1.0x |
| Numba | `uniform` | {nb_uni_val:.2f} {nb_uni_unit} | **{speedup_uniform:.1f}x** |
| Python | `obs` | {py_obs_val:.2f} {py_obs_unit} | 1.0x |
| Numba | `obs` | {nb_obs_val:.2f} {nb_obs_unit} | **{speedup_obs:.1f}x** |
"""


def generate_robustification_comparison_table(results: dict) -> str:
    """Generate a comparison table for robustification strategies."""
    mostfrequentnode = results.get("test_benchmark_robustification_mostfrequentnode")
    minimize_uniform = results.get(
        "test_benchmark_robustification_minimize_energy_uniform"
    )
    minimize_obs = results.get("test_benchmark_robustification_minimize_energy_obs")
    minimize_uniform_python = results.get(
        "test_benchmark_robustification_minimize_energy_uniform_python"
    )
    minimize_obs_python = results.get(
        "test_benchmark_robustification_minimize_energy_obs_python"
    )

    if not mostfrequentnode:
        return ""

    mfn_val, mfn_unit = format_time(mostfrequentnode["mean"])

    lines = [
        "The following table compares different robustification strategies. `MostFrequentNode` is used as the baseline reference (1.0x).",
        "",
        "| Strategy | Implementation | Mode | Mean Time | Ratio vs. MostFrequentNode |",
        "| :------- | :------------- | :--- | :-------- | :------------------------- |",
        f"| MostFrequentNode | - | - | {mfn_val:.2f} {mfn_unit} | 1.0x |",
    ]

    # MinimizeEnergy Numba uniform
    if minimize_uniform:
        me_uni_val, me_uni_unit = format_time(minimize_uniform["mean"])
        ratio_uniform = minimize_uniform["mean"] / mostfrequentnode["mean"]
        if ratio_uniform > 1:
            lines.append(
                f"| MinimizeEnergy | Numba | `uniform` | {me_uni_val:.2f} {me_uni_unit} | "
                f"**{ratio_uniform:.2f}x (slower)** |"
            )
        else:
            speedup_uniform = 1.0 / ratio_uniform
            lines.append(
                f"| MinimizeEnergy | Numba | `uniform` | {me_uni_val:.2f} {me_uni_unit} | "
                f"**{ratio_uniform:.4f}x ({speedup_uniform:.0f}x faster)** |"
            )

    # MinimizeEnergy Numba obs
    if minimize_obs:
        me_obs_val, me_obs_unit = format_time(minimize_obs["mean"])
        ratio_obs = minimize_obs["mean"] / mostfrequentnode["mean"]
        if ratio_obs > 1:
            lines.append(
                f"| MinimizeEnergy | Numba | `obs` | {me_obs_val:.2f} {me_obs_unit} | "
                f"**{ratio_obs:.2f}x (slower)** |"
            )
        else:
            speedup_obs = 1.0 / ratio_obs
            lines.append(
                f"| MinimizeEnergy | Numba | `obs` | {me_obs_val:.2f} {me_obs_unit} | "
                f"**{ratio_obs:.4f}x ({speedup_obs:.0f}x faster)** |"
            )

    # MinimizeEnergy Python uniform
    if minimize_uniform_python:
        me_uni_py_val, me_uni_py_unit = format_time(minimize_uniform_python["mean"])
        ratio_uniform_py = minimize_uniform_python["mean"] / mostfrequentnode["mean"]
        if ratio_uniform_py > 1:
            lines.append(
                f"| MinimizeEnergy | Python | `uniform` | {me_uni_py_val:.2f} {me_uni_py_unit} | "
                f"**{ratio_uniform_py:.2f}x (slower)** |"
            )
        else:
            speedup_uniform_py = 1.0 / ratio_uniform_py
            lines.append(
                f"| MinimizeEnergy | Python | `uniform` | {me_uni_py_val:.2f} {me_uni_py_unit} | "
                f"**{ratio_uniform_py:.4f}x ({speedup_uniform_py:.0f}x faster)** |"
            )

    # MinimizeEnergy Python obs
    if minimize_obs_python:
        me_obs_py_val, me_obs_py_unit = format_time(minimize_obs_python["mean"])
        ratio_obs_py = minimize_obs_python["mean"] / mostfrequentnode["mean"]
        if ratio_obs_py > 1:
            lines.append(
                f"| MinimizeEnergy | Python | `obs` | {me_obs_py_val:.2f} {me_obs_py_unit} | "
                f"**{ratio_obs_py:.2f}x (slower)** |"
            )
        else:
            speedup_obs_py = 1.0 / ratio_obs_py
            lines.append(
                f"| MinimizeEnergy | Python | `obs` | {me_obs_py_val:.2f} {me_obs_py_unit} | "
                f"**{ratio_obs_py:.4f}x ({speedup_obs_py:.0f}x faster)** |"
            )

    # Calculate Numba speedup if all required benchmarks are available
    speedup_text = ""
    if (
        minimize_uniform
        and minimize_uniform_python
        and minimize_obs
        and minimize_obs_python
    ):
        speedup_uniform = minimize_uniform_python["mean"] / minimize_uniform["mean"]
        speedup_obs = minimize_obs_python["mean"] / minimize_obs["mean"]
        # Sort speedups in ascending order
        speedups = sorted([speedup_uniform, speedup_obs])
        speedup_text = f"Numba acceleration provides {speedups[0]:.0f}-{speedups[1]:.0f}x speedup for the energy calculation. "

    lines.extend(
        [
            "",
            f"**Note:** `MinimizeEnergy` is significantly slower because it calls energy calculation at each `collect()` iteration (15 times for 10% robustification), while `MostFrequentNode` only performs lightweight `_closest_node()` lookups. {speedup_text}`MinimizeEnergy` may provide better clustering quality by selecting the globally optimal centers rather than the most frequently visited nodes.\n",
        ]
    )

    return "\n".join(lines)


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
            "generated_at": os.path.getmtime(Path(".benchmarks") / "Darwin-CPython-3.12-64bit" / source_file) if (Path(".benchmarks") / "Darwin-CPython-3.12-64bit" / source_file).exists() else None,
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
        nb_obs = results.get("test_benchmark_energy_numba_obs")
        
        if all([py_uniform, nb_uniform, py_obs, nb_obs]):
            structured["comparisons"]["energy_calculation"] = {
                "python_uniform_us": format_time(py_uniform["mean"])[0],
                "numba_uniform_us": format_time(nb_uniform["mean"])[0],
                "speedup_uniform": py_uniform["mean"] / nb_uniform["mean"],
                "python_obs_us": format_time(py_obs["mean"])[0],
                "numba_obs_us": format_time(nb_obs["mean"])[0],
                "speedup_obs": py_obs["mean"] / nb_obs["mean"],
            }
    
    # Add robustification comparison
    if "Robustification Strategies" in categories:
        mfn = results.get("test_benchmark_robustification_mostfrequentnode")
        me_uniform = results.get("test_benchmark_robustification_minimize_energy_uniform")
        me_obs = results.get("test_benchmark_robustification_minimize_energy_obs")
        me_uniform_py = results.get("test_benchmark_robustification_minimize_energy_uniform_python")
        me_obs_py = results.get("test_benchmark_robustification_minimize_energy_obs_python")
        
        if mfn:
            robust_comparison = {
                "baseline": {
                    "name": "MostFrequentNode",
                    "mean_us": format_time(mfn["mean"])[0],
                },
                "strategies": [],
            }
            
            if me_uniform:
                robust_comparison["strategies"].append({
                    "name": "MinimizeEnergy",
                    "implementation": "Numba",
                    "mode": "uniform",
                    "mean_us": format_time(me_uniform["mean"])[0],
                    "ratio_vs_baseline": me_uniform["mean"] / mfn["mean"],
                })
            
            if me_obs:
                robust_comparison["strategies"].append({
                    "name": "MinimizeEnergy",
                    "implementation": "Numba",
                    "mode": "obs",
                    "mean_us": format_time(me_obs["mean"])[0],
                    "ratio_vs_baseline": me_obs["mean"] / mfn["mean"],
                })
            
            if me_uniform_py:
                robust_comparison["strategies"].append({
                    "name": "MinimizeEnergy",
                    "implementation": "Python",
                    "mode": "uniform",
                    "mean_us": format_time(me_uniform_py["mean"])[0],
                    "ratio_vs_baseline": me_uniform_py["mean"] / mfn["mean"],
                })
            
            if me_obs_py:
                robust_comparison["strategies"].append({
                    "name": "MinimizeEnergy",
                    "implementation": "Python",
                    "mode": "obs",
                    "mean_us": format_time(me_obs_py["mean"])[0],
                    "ratio_vs_baseline": me_obs_py["mean"] / mfn["mean"],
                })
            
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
