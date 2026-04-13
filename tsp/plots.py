import sys
import os
import csv
import argparse
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend (works without a display)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


#  Color palette — consistent across all plots
COLORS = {
    "BruteForce":    "#e74c3c",   # red
    "HeldKarp":      "#e67e22",   # orange
    "NearestNeigh":  "#2980b9",   # blue
    "NN_MultiStart": "#27ae60",   # green
    "Christofides":  "#8e44ad",   # purple
}

MARKERS = {
    "BruteForce":    "x",
    "HeldKarp":      "s",
    "NearestNeigh":  "o",
    "NN_MultiStart": "^",
    "Christofides":  "D",
}

LINE_STYLES = {
    "BruteForce":    "--",
    "HeldKarp":      "--",
    "NearestNeigh":  "-",
    "NN_MultiStart": "-",
    "Christofides":  "-",
}

LABELS = {
    "BruteForce":    "Brute Force (Exact)",
    "HeldKarp":      "Held-Karp DP (Exact)",
    "NearestNeigh":  "Nearest Neighbor",
    "NN_MultiStart": "NN Multi-Start",
    "Christofides":  "Christofides",
}


#  Data loading
def load_csv(filepath):
    """
    Load benchmark CSV and return aggregated averages per (algorithm, n).

    Returns:
        data : dict mapping algorithm name → dict of n → {avg_time, avg_ratio, avg_cost}
    """
    raw = defaultdict(lambda: defaultdict(list))

    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alg = row["algorithm"]
            n = int(row["n"])
            try:
                time_ms = float(row["time_ms"])
                cost = float(row["cost"])
                ratio = float(row["ratio"]) if row["ratio"] != "N/A" else None
                raw[alg][n].append({"time_ms": time_ms, "cost": cost, "ratio": ratio})
            except (ValueError, KeyError):
                continue

    # Average across runs
    data = {}
    for alg, sizes in raw.items():
        data[alg] = {}
        for n, runs in sizes.items():
            times = [r["time_ms"] for r in runs]
            costs = [r["cost"] for r in runs]
            ratios = [r["ratio"] for r in runs if r["ratio"] is not None]

            data[alg][n] = {
                "avg_time":  sum(times) / len(times),
                "avg_cost":  sum(costs) / len(costs),
                "avg_ratio": sum(ratios) / len(ratios) if ratios else None,
            }

    return data


#  Plot 1: Runtime vs Input Size
def plot_runtime(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for alg, sizes in sorted(data.items()):
        ns = sorted(sizes.keys())
        times = [sizes[n]["avg_time"] for n in ns]

        ax.plot(ns, times,
                color=COLORS.get(alg, "gray"),
                marker=MARKERS.get(alg, "o"),
                linestyle=LINE_STYLES.get(alg, "-"),
                linewidth=2,
                markersize=7,
                label=LABELS.get(alg, alg))

    ax.set_yscale("log")
    ax.set_xlabel("Number of Cities (n)", fontsize=13)
    ax.set_ylabel("Average Runtime (ms, log scale)", fontsize=13)
    ax.set_title("TSP Algorithm Runtime vs Input Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3g}"))

    # Annotate complexity classes
    ax.annotate("Exponential\ngrowth zone", xy=(15, 1), fontsize=9,
                color="#e74c3c", alpha=0.7)
    ax.annotate("Polynomial\ngrowth zone", xy=(50, 0.1), fontsize=9,
                color="#2980b9", alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


#  Plot 2: Approximation Ratio vs Input Size
def plot_approx_ratio(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    approx_algs = ["NearestNeigh", "NN_MultiStart", "Christofides"]

    for alg in approx_algs:
        if alg not in data:
            continue
        sizes = data[alg]
        ns = sorted(n for n in sizes if sizes[n]["avg_ratio"] is not None)
        ratios = [sizes[n]["avg_ratio"] for n in ns]

        if not ns:
            continue

        ax.plot(ns, ratios,
                color=COLORS.get(alg, "gray"),
                marker=MARKERS.get(alg, "o"),
                linestyle=LINE_STYLES.get(alg, "-"),
                linewidth=2,
                markersize=7,
                label=LABELS.get(alg, alg))

    # Reference lines
    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1.5, alpha=0.7,
               label="Optimal (ratio = 1.0)")
    ax.axhline(y=1.5, color="#8e44ad", linestyle=":", linewidth=1.5, alpha=0.6,
               label="Christofides bound (1.5×)")

    ax.set_xlabel("Number of Cities (n)", fontsize=13)
    ax.set_ylabel("Approximation Ratio (approx / optimal)", fontsize=13)
    ax.set_title("TSP Approximation Ratio vs Input Size", fontsize=15, fontweight="bold")
    ax.set_ylim(0.9, 2.0)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


#  Plot 3: Cost Comparison Bar Chart (small n)
def plot_cost_comparison(data, output_path, max_n=15):
    """Bar chart comparing solution costs for small n (where all algorithms run)."""
    import numpy as np

    # Filter to sizes where we have data for all main algorithms
    target_algs = ["BruteForce", "HeldKarp", "NearestNeigh", "NN_MultiStart", "Christofides"]
    available_algs = [a for a in target_algs if a in data]

    # Find sizes where all selected algs have data
    all_ns = set.intersection(*(set(data[a].keys()) for a in available_algs))
    ns = sorted(n for n in all_ns if n <= max_n)

    if not ns:
        print("No overlapping sizes found for cost comparison bar chart.")
        return

    x = np.arange(len(ns))
    width = 0.15
    n_algs = len(available_algs)
    offsets = [(i - n_algs / 2 + 0.5) * width for i in range(n_algs)]

    fig, ax = plt.subplots(figsize=(12, 6))

    for i, alg in enumerate(available_algs):
        costs = [data[alg][n]["avg_cost"] for n in ns]
        bars = ax.bar(x + offsets[i], costs,
                      width=width,
                      color=COLORS.get(alg, "gray"),
                      label=LABELS.get(alg, alg),
                      alpha=0.85,
                      edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"n={n}" for n in ns], fontsize=11)
    ax.set_xlabel("Input Size (n)", fontsize=13)
    ax.set_ylabel("Average Tour Cost", fontsize=13)
    ax.set_title("TSP Solution Quality Comparison (Small Inputs)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


#  Entry point
if __name__ == "__main__":
    if not MATPLOTLIB_OK:
        print("matplotlib not found. Install with: pip install matplotlib")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate TSP benchmark plots")
    parser.add_argument("--input", default=None, help="Path to benchmark CSV")
    args = parser.parse_args()

    # Default CSV path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "..", "results", "tsp_benchmark.csv")
    csv_path = args.input or default_csv

    if not os.path.exists(csv_path):
        print(f"CSV not found at: {csv_path}")
        print("Run benchmark.py first to generate results.")
        sys.exit(1)

    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading data from: {csv_path}")
    data = load_csv(csv_path)

    plot_runtime(data,          os.path.join(results_dir, "tsp_runtime.png"))
    plot_approx_ratio(data,     os.path.join(results_dir, "tsp_approx_ratio.png"))
    plot_cost_comparison(data,  os.path.join(results_dir, "tsp_cost_comparison.png"))

    print("\nAll plots generated in:", results_dir)