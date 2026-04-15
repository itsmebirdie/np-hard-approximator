import time
import math
from sc_utils import validate_solution


#  Standard Greedy Set Cover
def greedy_set_cover(universe, subsets):
    """
    Greedy approximation for minimum set cover.
    At each iteration, greedily pick the subset covering the most
    currently uncovered elements. Ties broken by smallest index.

    Args:
        universe : list or set of elements
        subsets  : list of sets

    Returns:
        (cover_size, chosen_indices)
        cover_size     : number of subsets in the solution
        chosen_indices : list of indices into `subsets`, in order chosen
    """
    uncovered = set(universe)
    chosen = []
    remaining = list(range(len(subsets)))

    while uncovered:
        # Find the subset covering the most uncovered elements
        best_idx = -1
        best_count = -1
        for i in remaining:
            count = len(subsets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1 or best_count == 0:
            # Should not happen in a feasible instance
            break

        chosen.append(best_idx)
        uncovered -= subsets[best_idx]
        remaining.remove(best_idx)

    return len(chosen), chosen


def greedy_set_cover_with_trace(universe, subsets):
    """
    Same as greedy_set_cover but also returns a trace of each step.
    Used for detailed analysis and report tables.

    Returns:
        (cover_size, chosen_indices, trace)
        trace : list of dicts, one per iteration:
                  {
                    'step'        : int,
                    'chosen_idx'  : int,
                    'newly_covered': set,
                    'covered_so_far': int,
                    'remaining_uncovered': int,
                  }
    """
    uncovered = set(universe)
    total = len(uncovered)
    chosen = []
    remaining = list(range(len(subsets)))
    trace = []
    step = 1

    while uncovered:
        best_idx = -1
        best_count = -1
        for i in remaining:
            count = len(subsets[i] & uncovered)
            if count > best_count:
                best_count = count
                best_idx = i

        if best_idx == -1 or best_count == 0:
            break

        newly = subsets[best_idx] & uncovered
        chosen.append(best_idx)
        uncovered -= newly
        remaining.remove(best_idx)

        trace.append({
            "step":                step,
            "chosen_idx":          best_idx,
            "subset_content":      sorted(subsets[best_idx]),
            "newly_covered":       sorted(newly),
            "n_newly_covered":     len(newly),
            "covered_so_far":      total - len(uncovered),
            "remaining_uncovered": len(uncovered),
        })
        step += 1

    return len(chosen), chosen, trace


#  Randomized Greedy (tie-breaking variant)
def greedy_set_cover_randomized(universe, subsets, seed=None):
    """
    Randomized greedy: when multiple subsets tie for maximum coverage,
    choose one uniformly at random among the tied options.

    Useful for averaging multiple runs to reduce variance in analysis.

    Args:
        universe : list of elements
        subsets  : list of sets
        seed     : random seed

    Returns:
        (cover_size, chosen_indices)
    """
    import random
    if seed is not None:
        random.seed(seed)

    uncovered = set(universe)
    chosen = []
    remaining = list(range(len(subsets)))

    while uncovered:
        best_count = -1
        for i in remaining:
            count = len(subsets[i] & uncovered)
            if count > best_count:
                best_count = count

        # Collect all subsets tied at best_count
        tied = [i for i in remaining if len(subsets[i] & uncovered) == best_count]
        best_idx = random.choice(tied)

        chosen.append(best_idx)
        uncovered -= subsets[best_idx]
        remaining.remove(best_idx)

    return len(chosen), chosen


#  Approximation Ratio Analysis
def harmonic_number(n):
    """
    Compute the n-th harmonic number H(n) = 1 + 1/2 + ... + 1/n.
    This is the theoretical approximation ratio bound for greedy Set Cover.
    """
    return sum(1.0 / i for i in range(1, n + 1))


def theoretical_bound(n_elements):
    """
    Return the theoretical approximation ratio upper bound H(n) for n elements.
    """
    return harmonic_number(n_elements)


def compute_approximation_ratio(optimal_size, greedy_size):
    """
    Compute the approximation ratio.

    ratio = greedy_size / optimal_size
    A ratio of 1.0 means greedy found the optimal solution.
    A ratio of 2.0 means greedy used twice as many sets as optimal.

    Returns None if optimal is unknown.
    """
    if optimal_size is None or optimal_size == 0:
        return None
    return greedy_size / optimal_size


def analyze_ratio_bound(universe, subsets, optimal_size, greedy_size):
    """
    Full ratio bound analysis report for a single instance.

    Returns:
        dict with keys:
          n_elements, n_subsets, optimal_size, greedy_size,
          actual_ratio, theoretical_bound, bound_holds, slack
    """
    n = len(universe)
    actual = compute_approximation_ratio(optimal_size, greedy_size)
    bound = theoretical_bound(n)
    bound_holds = (actual is None) or (actual <= bound + 1e-9)

    return {
        "n_elements":        n,
        "n_subsets":         len(subsets),
        "optimal_size":      optimal_size,
        "greedy_size":       greedy_size,
        "actual_ratio":      actual,
        "theoretical_bound": bound,
        "bound_holds":       bound_holds,
        "slack":             (bound - actual) if actual else None,
    }


def print_ratio_analysis(report):
    """Pretty-print the ratio analysis report."""
    print("  Approximation Ratio Analysis")
    print("  " + "-" * 40)
    print(f"    Universe size      : {report['n_elements']}")
    print(f"    Number of subsets  : {report['n_subsets']}")
    print(f"    Optimal cover size : {report['optimal_size']}")
    print(f"    Greedy cover size  : {report['greedy_size']}")
    if report["actual_ratio"] is not None:
        print(f"    Actual ratio       : {report['actual_ratio']:.4f}×")
    else:
        print(f"    Actual ratio       : N/A (optimal unknown)")
    print(f"    Theoretical bound  : H({report['n_elements']}) = {report['theoretical_bound']:.4f}×")
    holds = "YES ✓" if report["bound_holds"] else "NO  ✗"
    print(f"    Bound holds        : {holds}")
    if report["slack"] is not None:
        print(f"    Slack (bound-ratio): {report['slack']:.4f}")
    print()


#  Timed wrappers (used by benchmark.py)
def run(universe, subsets):
    """
    Run standard greedy and return (size, chosen_indices, elapsed_ms).
    """
    start = time.perf_counter()
    size, chosen = greedy_set_cover(universe, subsets)
    elapsed = (time.perf_counter() - start) * 1000
    return size, chosen, elapsed


def run_with_trace(universe, subsets):
    """
    Run greedy with trace and return (size, chosen_indices, trace, elapsed_ms).
    """
    start = time.perf_counter()
    size, chosen, trace = greedy_set_cover_with_trace(universe, subsets)
    elapsed = (time.perf_counter() - start) * 1000
    return size, chosen, trace, elapsed


#  Quick self-test
if __name__ == "__main__":
    from sc_utils import (
        small_exact_instance, random_instance,
        print_instance, print_solution
    )
    from exact_backtracking import set_cover_exact

    print("=" * 60)
    print("  Greedy Set Cover — Self-Test + Ratio Analysis")
    print("=" * 60)

    # Test 1: known instance
    universe, subsets = small_exact_instance()
    print_instance(universe, subsets, "Known Test Instance")

    # Greedy
    g_size, g_chosen, g_trace, g_ms = run_with_trace(universe, subsets)
    print_solution("Greedy Approximation", g_chosen, subsets, universe, g_ms)

    print("  Step-by-step trace:")
    for step in g_trace:
        print(f"    Step {step['step']}: chose S{step['chosen_idx']}  "
              f"newly covered={step['newly_covered']}  "
              f"remaining={step['remaining_uncovered']}")
    print()

    # Exact for comparison
    opt_size, opt_chosen, opt_ms = __import__("exact_backtracking").run(universe, subsets)
    report = analyze_ratio_bound(universe, subsets, opt_size, g_size)
    print_ratio_analysis(report)

    # Test 2: Cross-validation on random instances
    print("Cross-validating greedy vs exact on random instances:\n")
    print(f"  {'|U|':>5}  {'|S|':>5}  {'Optimal':>9}  {'Greedy':>8}  "
          f"{'Ratio':>7}  {'H(n)':>7}  {'≤H(n)?':>7}  {'Valid':>6}")
    print("  " + "-" * 70)

    all_ok = True
    for n_el in [5, 8, 10, 12, 15, 20]:
        n_sets = n_el * 3
        u, s = random_instance(n_el, n_sets, coverage=0.25, seed=n_el * 31)

        opt_sz, _, _ = __import__("exact_backtracking").run(u, s)
        g_sz, g_ch, _ = run(u, s)

        from sc_utils import validate_solution
        valid, _ = validate_solution(u, s, g_ch)
        ratio = g_sz / opt_sz if opt_sz else float("inf")
        bound = theoretical_bound(n_el)
        holds = ratio <= bound + 1e-9
        sym = "✓" if valid and holds else "✗"

        print(f"  {n_el:>5}  {len(s):>5}  {opt_sz:>9}  {g_sz:>8}  "
              f"{ratio:>7.3f}  {bound:>7.3f}  {'Yes ✓' if holds else 'No  ✗':>7}  "
              f"{'✓' if valid else '✗':>6}")

        if not valid or not holds:
            all_ok = False

    print()
    if all_ok:
        print("All instances: greedy valid and within H(n) bound.")
    else:
        print("SOME INSTANCES FAILED.")

    # Show the theoretical bound growing with n
    print("\n  Theoretical H(n) bound as n grows:")
    print(f"  {'n':>6}  {'H(n)':>8}  {'ln(n)':>8}")
    print("  " + "-" * 26)
    for n in [5, 10, 20, 50, 100, 500, 1000]:
        hn = harmonic_number(n)
        ln = math.log(n) if n > 0 else 0
        print(f"  {n:>6}  {hn:>8.4f}  {ln:>8.4f}")