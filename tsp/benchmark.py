import sys
import os
import csv
import time
import argparse

# Add parent directory so imports work when run from /tsp/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tsp_utils import random_euclidean_instance, validate_tour
import brute_force
import dp_bitmask
import nearest_neighbor
import christofides


#  Configuration
# Sizes where brute force is feasible (≤10), DP is feasible (≤20),
# and only approximations are practical (>20)
DEFAULT_SIZES = [5, 6, 7, 8, 9, 10, 12, 15, 20, 25, 30, 50, 75, 100]

ALGORITHMS = {
    "BruteForce":   {"module": brute_force,       "max_n": 10,   "type": "exact"},
    "HeldKarp":     {"module": dp_bitmask,         "max_n": 20,   "type": "exact"},
    "NearestNeigh": {"module": nearest_neighbor,   "max_n": 9999, "type": "approx"},
    "NN_MultiStart":{"module": nearest_neighbor,   "max_n": 9999, "type": "approx", "multistart": True},
    "Christofides": {"module": christofides,       "max_n": 9999, "type": "approx"},
}


#  Core benchmark logic
def run_benchmark(sizes, seed_base=42, runs_per_size=3):
    """
    Run all algorithms on random Euclidean instances of each size.

    Args:
        sizes        : list of n values to test
        seed_base    : base random seed (each size gets a different seed)
        runs_per_size: number of different random instances per size

    Returns:
        records : list of dicts, one per (n, run, algorithm)
    """
    records = []

    for n in sizes:
        for run_idx in range(runs_per_size):
            seed = seed_base + n * 100 + run_idx
            coords, dist = random_euclidean_instance(n, seed=seed)

            # Get the exact optimal cost (only when feasible)
            optimal_cost = None
            if n <= 20:
                try:
                    opt, _, _ = dp_bitmask.run(dist)
                    optimal_cost = opt
                except Exception:
                    pass

            for alg_name, config in ALGORITHMS.items():
                if n > config["max_n"]:
                    continue

                mod = config["module"]
                is_multistart = config.get("multistart", False)

                try:
                    if is_multistart:
                        cost, tour, elapsed_ms = mod.run_multistart(dist)
                    else:
                        cost, tour, elapsed_ms = mod.run(dist)

                    valid, _ = validate_tour(tour, n)
                    ratio = (cost / optimal_cost) if optimal_cost else None

                    records.append({
                        "n":           n,
                        "run":         run_idx + 1,
                        "seed":        seed,
                        "algorithm":   alg_name,
                        "type":        config["type"],
                        "cost":        cost,
                        "optimal":     optimal_cost if optimal_cost else "N/A",
                        "ratio":       f"{ratio:.4f}" if ratio else "N/A",
                        "time_ms":     f"{elapsed_ms:.4f}",
                        "tour_valid":  valid,
                    })

                except Exception as e:
                    records.append({
                        "n":           n,
                        "run":         run_idx + 1,
                        "seed":        seed,
                        "algorithm":   alg_name,
                        "type":        config["type"],
                        "cost":        "ERROR",
                        "optimal":     optimal_cost if optimal_cost else "N/A",
                        "ratio":       "N/A",
                        "time_ms":     "N/A",
                        "tour_valid":  False,
                        "error":       str(e),
                    })

    return records


def print_table(records):
    """Print a formatted benchmark results table."""
    print(f"\n{'n':>4}  {'Algorithm':<16}  {'Cost':>8}  {'Optimal':>8}  {'Ratio':>6}  {'Time(ms)':>10}  {'Valid':>5}")
    print("  " + "-" * 68)

    last_n = None
    for r in records:
        if r["n"] != last_n:
            if last_n is not None:
                print()
            last_n = r["n"]

        ratio_str = r.get("ratio", "N/A")
        if ratio_str not in ("N/A", "ERROR") and float(ratio_str) > 1.0:
            flag = f"  ×{float(ratio_str):.3f}"
        else:
            flag = ""

        valid_sym = "✓" if r.get("tour_valid") else "✗"
        print(f"  {r['n']:>3}  {r['algorithm']:<16}  {str(r['cost']):>8}  "
              f"{str(r['optimal']):>8}  {ratio_str:>6}  {r['time_ms']:>10}  {valid_sym:>5}{flag}")


def save_csv(records, output_path):
    """Save records to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = ["n", "run", "seed", "algorithm", "type", "cost", "optimal", "ratio", "time_ms", "tour_valid"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    print(f"\nCSV saved to: {output_path}")


def save_summary(records, output_path):
    """Save a human-readable summary with averages per (algorithm, n)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Aggregate by (n, algorithm)
    from collections import defaultdict
    grouped = defaultdict(list)
    for r in records:
        if r["cost"] != "ERROR":
            grouped[(r["n"], r["algorithm"])].append(r)

    with open(output_path, "w") as f:
        f.write("TSP Benchmark Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'n':>4}  {'Algorithm':<16}  {'Avg Cost':>10}  {'Avg Ratio':>10}  {'Avg Time(ms)':>13}\n")
        f.write("-" * 60 + "\n")

        last_n = None
        for (n, alg), rs in sorted(grouped.items()):
            if n != last_n:
                if last_n is not None:
                    f.write("\n")
                last_n = n

            costs = [r["cost"] for r in rs if isinstance(r["cost"], (int, float))]
            times = [float(r["time_ms"]) for r in rs if r["time_ms"] != "N/A"]
            ratios = [float(r["ratio"]) for r in rs if r["ratio"] not in ("N/A", "ERROR")]

            avg_cost = sum(costs) / len(costs) if costs else float("nan")
            avg_time = sum(times) / len(times) if times else float("nan")
            avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")

            ratio_str = f"{avg_ratio:.4f}" if ratios else "N/A"
            f.write(f"  {n:>3}  {alg:<16}  {avg_cost:>10.1f}  {ratio_str:>10}  {avg_time:>13.4f}\n")

    print(f"Summary saved to: {output_path}")


#  Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSP Algorithm Benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES,
                        help="List of n values to benchmark")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed")
    parser.add_argument("--runs", type=int, default=3,
                        help="Number of random instances per size")
    args = parser.parse_args()

    print("=" * 70)
    print("  TSP Benchmark — NP-AA Project")
    print("=" * 70)
    print(f"\nSizes   : {args.sizes}")
    print(f"Seed    : {args.seed}")
    print(f"Runs/n  : {args.runs}")
    print(f"\nRunning benchmarks...\n")

    t0 = time.perf_counter()
    records = run_benchmark(args.sizes, seed_base=args.seed, runs_per_size=args.runs)
    total_time = time.perf_counter() - t0

    print_table(records)
    print(f"\nTotal benchmark time: {total_time:.2f}s")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_csv(records, os.path.join(results_dir, "tsp_benchmark.csv"))
    save_summary(records, os.path.join(results_dir, "tsp_summary.txt"))