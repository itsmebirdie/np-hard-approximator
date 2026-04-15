import sys
import os
import csv
import argparse
import math
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker
    import numpy as np
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


COLORS = {
    "Exact_BT": "#e74c3c",   # red
    "Greedy":   "#2980b9",   # blue
}
MARKERS = {
    "Exact_BT": "s",
    "Greedy":   "o",
}
LABELS = {
    "Exact_BT": "Exact Backtracking",
    "Greedy":   "Greedy Approximation",
}


#  Data loading
def load_csv(filepath):
    raw = defaultdict(lambda: defaultdict(list))
    with open(filepath, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            alg = row["algorithm"]
            n = int(row["n_elements"])
            try:
                time_ms = float(row["time_ms"])
                cover = int(row["cover_size"])
                ratio = float(row["ratio"]) if row["ratio"] != "N/A" else None
                raw[alg][n].append({"time_ms": time_ms, "cover": cover, "ratio": ratio})
            except (ValueError, KeyError):
                continue

    data = {}
    for alg, sizes in raw.items():
        data[alg] = {}
        for n, runs in sizes.items():
            times  = [r["time_ms"] for r in runs]
            covers = [r["cover"]   for r in runs]
            ratios = [r["ratio"]   for r in runs if r["ratio"] is not None]
            data[alg][n] = {
                "avg_time":  sum(times)  / len(times),
                "avg_cover": sum(covers) / len(covers),
                "avg_ratio": sum(ratios) / len(ratios) if ratios else None,
            }
    return data


#  Plot 1: Runtime vs Universe Size
def plot_runtime(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for alg, sizes in sorted(data.items()):
        ns    = sorted(sizes.keys())
        times = [sizes[n]["avg_time"] for n in ns]
        ax.plot(ns, times,
                color=COLORS.get(alg, "gray"),
                marker=MARKERS.get(alg, "o"),
                linewidth=2, markersize=7,
                label=LABELS.get(alg, alg))

    ax.set_yscale("log")
    ax.set_xlabel("Universe Size |U|", fontsize=13)
    ax.set_ylabel("Average Runtime (ms, log scale)", fontsize=13)
    ax.set_title("Set Cover Algorithm Runtime vs Universe Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.3g}"))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


#  Plot 2: Cover Size Comparison
def plot_cover_size(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    for alg, sizes in sorted(data.items()):
        ns     = sorted(sizes.keys())
        covers = [sizes[n]["avg_cover"] for n in ns]
        ax.plot(ns, covers,
                color=COLORS.get(alg, "gray"),
                marker=MARKERS.get(alg, "o"),
                linewidth=2, markersize=7,
                label=LABELS.get(alg, alg))

    ax.set_xlabel("Universe Size |U|", fontsize=13)
    ax.set_ylabel("Average Cover Size (# subsets used)", fontsize=13)
    ax.set_title("Set Cover Solution Quality vs Universe Size", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")

#  Plot 3: Approximation Ratio vs Universe Size
def plot_approx_ratio(data, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))

    if "Greedy" in data:
        sizes = data["Greedy"]
        ns     = sorted(n for n in sizes if sizes[n]["avg_ratio"] is not None)
        ratios = [sizes[n]["avg_ratio"] for n in ns]
        bounds = [sum(1/i for i in range(1, n+1)) for n in ns]

        ax.plot(ns, ratios,
                color="#2980b9", marker="o", linewidth=2, markersize=7,
                label="Greedy actual ratio")
        ax.plot(ns, bounds,
                color="#e74c3c", linestyle="--", linewidth=2,
                label="H(n) theoretical bound")
        ax.fill_between(ns, ratios, bounds,
                        alpha=0.12, color="#2980b9",
                        label="Gap between actual and bound")

    ax.axhline(y=1.0, color="black", linestyle="-", linewidth=1.5, alpha=0.6,
               label="Optimal (ratio = 1.0)")

    ax.set_xlabel("Universe Size |U|", fontsize=13)
    ax.set_ylabel("Approximation Ratio", fontsize=13)
    ax.set_title("Greedy Set Cover: Actual Ratio vs H(n) Bound", fontsize=15, fontweight="bold")
    ax.set_ylim(0.8, None)
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


#  Plot 4: H(n) Bound Growth (theory only)
def plot_hn_growth(output_path):
    ns  = list(range(1, 201))
    hn  = [sum(1/i for i in range(1, n+1)) for n in ns]
    lnn = [math.log(n) for n in ns]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(ns, hn,  color="#8e44ad", linewidth=2.5, label="H(n) = 1 + 1/2 + ... + 1/n")
    ax.plot(ns, lnn, color="#27ae60", linewidth=2, linestyle="--", label="ln(n)  (lower bound)")

    ax.set_xlabel("Universe Size n", fontsize=13)
    ax.set_ylabel("Approximation Ratio Bound", fontsize=13)
    ax.set_title("Theoretical Greedy Set Cover Bound: H(n) vs ln(n)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, linestyle="--", alpha=0.5)

    # Annotate a few key values
    for n_ann in [10, 50, 100, 200]:
        if n_ann <= max(ns):
            idx = n_ann - 1
            ax.annotate(f"H({n_ann})={hn[idx]:.2f}",
                        xy=(n_ann, hn[idx]),
                        xytext=(n_ann + 5, hn[idx] + 0.1),
                        fontsize=8, color="#8e44ad")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Plot saved: {output_path}")


#  Entry point
if __name__ == "__main__":
    if not MATPLOTLIB_OK:
        print("matplotlib not found. Install with: pip install matplotlib numpy")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Generate Set Cover benchmark plots")
    parser.add_argument("--input", default=None, help="Path to sc_benchmark.csv")
    args = parser.parse_args()

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    default_csv = os.path.join(script_dir, "..", "results", "sc_benchmark.csv")
    csv_path    = args.input or default_csv

    if not os.path.exists(csv_path):
        print(f"CSV not found at: {csv_path}")
        print("Run sc_benchmark.py first.")
        sys.exit(1)

    results_dir = os.path.join(script_dir, "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    print(f"Loading data from: {csv_path}")
    data = load_csv(csv_path)

    plot_runtime(    data, os.path.join(results_dir, "sc_runtime.png"))
    plot_cover_size( data, os.path.join(results_dir, "sc_cover_size.png"))
    plot_approx_ratio(data, os.path.join(results_dir, "sc_approx_ratio.png"))
    plot_hn_growth(        os.path.join(results_dir, "sc_hn_growth.png"))

    print("\nAll Set Cover plots generated in:", results_dir)