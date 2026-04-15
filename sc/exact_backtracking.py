import time
from sc_utils import validate_solution


#  Core backtracking solver
def _backtrack(subsets, uncovered, current, best, start_idx, sorted_indices):
    """
    Recursive backtracking to find minimum set cover.

    Args:
        subsets        : list of all subsets (frozensets for fast ops)
        uncovered      : set of elements not yet covered
        current        : list of subset indices chosen so far
        best           : list (mutable reference) — best[0] holds current best solution
        start_idx      : index into sorted_indices to start trying from
        sorted_indices : subsets sorted by size descending (larger first)
    """
    # Base case: everything covered
    if not uncovered:
        if len(current) < len(best[0]):
            best[0] = list(current)
        return

    # Bound: can't possibly beat best known solution
    if len(current) >= len(best[0]) - 1:
        return

    # Mandatory set pruning:
    # If any element is covered by only one remaining subset, we MUST include it.
    # Build element → which subsets cover it (among remaining subsets)
    element_coverage = {}
    remaining = sorted_indices[start_idx:]
    for e in uncovered:
        covering = [i for i in remaining if e in subsets[i]]
        if not covering:
            return  # infeasible branch — some element can't be covered
        element_coverage[e] = covering

    # Force mandatory sets (elements with only one covering option)
    forced = set()
    for e, covering in element_coverage.items():
        if len(covering) == 1:
            forced.add(covering[0])

    if forced:
        # Add all forced sets at once
        new_uncovered = set(uncovered)
        for idx in forced:
            new_uncovered -= subsets[idx]
        current.extend(sorted(forced))
        _backtrack(subsets, new_uncovered, current, best, start_idx, sorted_indices)
        for _ in forced:
            current.pop()
        return

    # Lower bound estimate: at least ceil(|uncovered| / max_subset_size) more sets needed
    if remaining:
        max_coverage = max(len(subsets[i] & uncovered) for i in remaining)
        if max_coverage == 0:
            return  # no remaining set covers anything useful
        lower_bound = -(-len(uncovered) // max_coverage)  # ceiling division
        if len(current) + lower_bound >= len(best[0]):
            return

    # Branch: try each remaining subset
    for i, idx in enumerate(remaining):
        newly_covered = subsets[idx] & uncovered
        if not newly_covered:
            continue  # skip subsets that add nothing

        current.append(idx)
        _backtrack(
            subsets,
            uncovered - newly_covered,
            current,
            best,
            start_idx + i + 1,
            sorted_indices,
        )
        current.pop()


def set_cover_exact(universe, subsets):
    """
    Find the minimum set cover using backtracking with pruning.

    Args:
        universe : list of elements (or any iterable)
        subsets  : list of sets

    Returns:
        (min_size, chosen_indices)
        min_size       : number of sets in the optimal cover
        chosen_indices : sorted list of indices into `subsets`
    """
    u = set(universe)
    m = len(subsets)

    if m == 0:
        return float("inf"), []

    # Convert to frozensets for fast set operations
    frozen = [frozenset(s) for s in subsets]

    # Sort subsets by size descending — try largest first for better pruning
    sorted_indices = sorted(range(m), key=lambda i: len(frozen[i]), reverse=True)

    # Start with a greedy solution as initial upper bound
    # (avoids exploring the entire tree before finding any solution)
    greedy_sol = _greedy_upper_bound(u, frozen)
    best = [greedy_sol]

    _backtrack(frozen, u, [], best, 0, sorted_indices)

    return len(best[0]), sorted(best[0])


def _greedy_upper_bound(universe, frozen_subsets):
    """
    Quick greedy solution to use as initial upper bound in backtracking.
    At each step picks the subset that covers the most uncovered elements.
    """
    uncovered = set(universe)
    chosen = []
    remaining = list(range(len(frozen_subsets)))

    while uncovered:
        best_idx = max(remaining, key=lambda i: len(frozen_subsets[i] & uncovered))
        if not (frozen_subsets[best_idx] & uncovered):
            break
        chosen.append(best_idx)
        uncovered -= frozen_subsets[best_idx]
        remaining.remove(best_idx)

    return chosen


#  Timed wrapper (used by benchmark.py)
def run(universe, subsets):
    """
    Run exact backtracking and return (size, chosen_indices, elapsed_ms).
    """
    start = time.perf_counter()
    size, chosen = set_cover_exact(universe, subsets)
    elapsed = (time.perf_counter() - start) * 1000
    return size, chosen, elapsed


#  Quick self-test
if __name__ == "__main__":
    from sc_utils import (
        small_exact_instance, random_instance,
        print_instance, print_solution, validate_solution
    )

    print("=" * 55)
    print("  Exact Backtracking Set Cover — Self-Test")
    print("=" * 55)

    # Test 1: known instance
    universe, subsets = small_exact_instance()
    print_instance(universe, subsets, "Known Test Instance")

    size, chosen, ms = run(universe, subsets)
    print_solution("Exact Backtracking", chosen, subsets, universe, ms)
    assert size == 1, f"Expected optimal size 1, got {size}"
    print("Test 1 passed: optimal size = 1 (full-cover subset selected)\n")

    # Test 2: random instances, verify solution validity
    print("Testing on random instances...\n")
    print(f"  {'|U|':>5}  {'|S|':>5}  {'Cover Size':>12}  {'Time(ms)':>10}  {'Valid':>6}")
    print("  " + "-" * 46)

    all_valid = True
    for n_el in [5, 8, 10, 12, 15]:
        n_sets = n_el * 2
        u, s = random_instance(n_el, n_sets, coverage=0.3, seed=n_el * 17)
        sz, ch, ms = run(u, s)
        valid, _ = validate_solution(u, s, ch)
        sym = "✓" if valid else "✗"
        print(f"  {n_el:>5}  {len(s):>5}  {sz:>12}  {ms:>10.4f}  {sym:>6}")
        if not valid:
            all_valid = False

    print()
    if all_valid:
        print("All solutions are valid covers.")
    else:
        print("SOME SOLUTIONS ARE INVALID.")