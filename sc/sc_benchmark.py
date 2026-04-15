import sys
import os
import csv
import time
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sc_utils import random_instance, validate_solution
import exact_backtracking
import greedy_approx

#  Configuration
# Exact backtracking is feasible up to ~20–22 subsets.
# Beyond that, only greedy runs.
DEFAULT_ELEMENT_SIZES = [5, 8, 10, 12, 15, 18, 20, 25, 30, 50, 75, 100]
EXACT_MAX_ELEMENTS = 20     # beyond this, skip exact (too slow)
SETS_MULTIPLIER    = 3      # number of subsets = n_elements × SETS_MULTIPLIER
COVERAGE_PROB      = 0.25   # probability each element appears in a subset


ALGORITHMS = {
    "Exact_BT":     {"module": exact_backtracking, "max_n": EXACT_MAX_ELEMENTS, "type": "exact"},
    "Greedy":       {"module": greedy_approx,       "max_n": 9999,               "type": "approx"},
}


#  Core benchmark logic
def run_benchmark(element_sizes, seed_base=42, runs_per_size=3):
    records = []

    for n_el in element_sizes:
        n_sets = n_el * SETS_MULTIPLIER

        for run_idx in range(runs_per_size):
            seed = seed_base + n_el * 100 + run_idx
            universe, subsets = random_instance(n_el, n_sets,
                                                coverage=COVERAGE_PROB,
                                                seed=seed)
            actual_n_sets = len(subsets)  # may differ from n_sets due to feasibility fix

            # Get exact optimal (only when feasible)
            optimal_size = None
            if n_el <= EXACT_MAX_ELEMENTS:
                try:
                    opt_sz, _, _ = exact_backtracking.run(universe, subsets)
                    optimal_size = opt_sz
                except Exception:
                    pass

            for alg_name, config in ALGORITHMS.items():
                if n_el > config["max_n"]:
                    continue

                mod = config["module"]

                try:
                    size, chosen, elapsed_ms = mod.run(universe, subsets)

                    valid, _ = validate_solution(universe, subsets, chosen)
                    ratio = greedy_approx.compute_approximation_ratio(optimal_size, size)
                    bound = greedy_approx.theoretical_bound(n_el)
                    bound_holds = (ratio is None) or (ratio <= bound + 1e-9)

                    records.append({
                        "n_elements":     n_el,
                        "n_subsets":      actual_n_sets,
                        "run":            run_idx + 1,
                        "seed":           seed,
                        "algorithm":      alg_name,
                        "type":           config["type"],
                        "cover_size":     size,
                        "optimal":        optimal_size if optimal_size is not None else "N/A",
                        "ratio":          f"{ratio:.4f}" if ratio is not None else "N/A",
                        "H_n_bound":      f"{bound:.4f}",
                        "bound_holds":    bound_holds,
                        "time_ms":        f"{elapsed_ms:.4f}",
                        "valid":          valid,
                    })

                except Exception as e:
                    records.append({
                        "n_elements":     n_el,
                        "n_subsets":      actual_n_sets,
                        "run":            run_idx + 1,
                        "seed":           seed,
                        "algorithm":      alg_name,
                        "type":           config["type"],
                        "cover_size":     "ERROR",
                        "optimal":        optimal_size if optimal_size is not None else "N/A",
                        "ratio":          "N/A",
                        "H_n_bound":      "N/A",
                        "bound_holds":    False,
                        "time_ms":        "N/A",
                        "valid":          False,
                        "error":          str(e),
                    })

    return records


def print_table(records):
    print(f"\n  {'|U|':>5}  {'|S|':>5}  {'Algorithm':<12}  {'Cover':>6}  "
          f"{'Optimal':>8}  {'Ratio':>7}  {'H(n)':>7}  {'Time(ms)':>10}  {'OK':>3}")
    print("  " + "-" * 75)

    last_n = None
    for r in records:
        if r["n_elements"] != last_n:
            if last_n is not None:
                print()
            last_n = r["n_elements"]

        ok = "✓" if r["valid"] and r["bound_holds"] else "✗"
        print(f"  {r['n_elements']:>5}  {r['n_subsets']:>5}  {r['algorithm']:<12}  "
              f"{str(r['cover_size']):>6}  {str(r['optimal']):>8}  "
              f"{r['ratio']:>7}  {r['H_n_bound']:>7}  "
              f"{r['time_ms']:>10}  {ok:>3}")


def save_csv(records, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fieldnames = [
        "n_elements", "n_subsets", "run", "seed", "algorithm", "type",
        "cover_size", "optimal", "ratio", "H_n_bound", "bound_holds", "time_ms", "valid"
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    print(f"\nCSV saved to: {output_path}")


def save_summary(records, output_path):
    from collections import defaultdict
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    grouped = defaultdict(list)
    for r in records:
        if r["cover_size"] != "ERROR":
            grouped[(r["n_elements"], r["algorithm"])].append(r)

    with open(output_path, "w") as f:
        f.write("Set Cover Benchmark Summary\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"  {'|U|':>5}  {'Algorithm':<12}  {'Avg Cover':>10}  "
                f"{'Avg Ratio':>10}  {'H(n)':>8}  {'Avg Time(ms)':>13}\n")
        f.write("  " + "-" * 65 + "\n")

        last_n = None
        for (n_el, alg), rs in sorted(grouped.items()):
            if n_el != last_n:
                if last_n is not None:
                    f.write("\n")
                last_n = n_el

            covers = [r["cover_size"] for r in rs if isinstance(r["cover_size"], int)]
            times  = [float(r["time_ms"]) for r in rs if r["time_ms"] != "N/A"]
            ratios = [float(r["ratio"]) for r in rs if r["ratio"] not in ("N/A",)]
            hn     = greedy_approx.theoretical_bound(n_el)

            avg_cover = sum(covers) / len(covers) if covers else float("nan")
            avg_time  = sum(times)  / len(times)  if times  else float("nan")
            avg_ratio = sum(ratios) / len(ratios) if ratios else float("nan")
            ratio_str = f"{avg_ratio:.4f}" if ratios else "N/A"

            f.write(f"  {n_el:>5}  {alg:<12}  {avg_cover:>10.2f}  "
                    f"{ratio_str:>10}  {hn:>8.4f}  {avg_time:>13.4f}\n")

    print(f"Summary saved to: {output_path}")


#  Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set Cover Algorithm Benchmark")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_ELEMENT_SIZES)
    parser.add_argument("--seed",  type=int, default=42)
    parser.add_argument("--runs",  type=int, default=3)
    args = parser.parse_args()

    print("=" * 70)
    print("  Set Cover Benchmark — NP-AA Project")
    print("=" * 70)
    print(f"\nElement sizes : {args.sizes}")
    print(f"Subsets/size  : n × {SETS_MULTIPLIER}")
    print(f"Coverage prob : {COVERAGE_PROB}")
    print(f"Seed          : {args.seed}")
    print(f"Runs/size     : {args.runs}")
    print(f"\nRunning benchmarks...\n")

    t0 = time.perf_counter()
    records = run_benchmark(args.sizes, seed_base=args.seed, runs_per_size=args.runs)
    total = time.perf_counter() - t0

    print_table(records)
    print(f"\nTotal benchmark time: {total:.2f}s")

    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    save_csv(records, os.path.join(results_dir, "sc_benchmark.csv"))
    save_summary(records, os.path.join(results_dir, "sc_summary.txt"))