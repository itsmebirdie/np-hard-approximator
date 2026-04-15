import random
import json
import os


#  Instance Generators
def random_instance(n_elements, n_sets, coverage=0.3, seed=None):
    """
    Generate a random Set Cover instance.

    Each subset is constructed by independently including each element
    with probability `coverage`. The last subset always includes all
    uncovered elements to guarantee the instance is feasible.

    Args:
        n_elements : size of the universe |U|
        n_sets     : number of available subsets |S|
        coverage   : probability each element appears in a given subset
        seed       : random seed

    Returns:
        universe : list of element indices [0, 1, ..., n_elements-1]
        subsets  : list of sets, each a subset of the universe
    """
    if seed is not None:
        random.seed(seed)

    universe = list(range(n_elements))
    subsets = []

    for _ in range(n_sets):
        subset = set(e for e in universe if random.random() < coverage)
        subsets.append(subset)

    # Ensure feasibility: add a "catch-all" set covering everything not yet covered
    covered = set().union(*subsets) if subsets else set()
    uncovered = set(universe) - covered
    if uncovered:
        subsets.append(uncovered)

    return universe, subsets


def structured_instance(n_elements, n_sets, overlap=0.2, seed=None):
    """
    Generate a structured instance where subsets have controlled overlap.

    The universe is partitioned into n_sets equal blocks. Each subset
    covers its own block completely, plus a random `overlap` fraction
    of elements from neighbouring blocks.

    Args:
        n_elements : size of universe
        n_sets     : number of subsets
        overlap    : fraction of neighbouring elements each subset also covers
        seed       : random seed

    Returns:
        universe, subsets
    """
    if seed is not None:
        random.seed(seed)

    universe = list(range(n_elements))
    block_size = max(1, n_elements // n_sets)
    subsets = []

    for i in range(n_sets):
        start = i * block_size
        end = min(start + block_size, n_elements)
        subset = set(range(start, end))

        # Add some overlap from adjacent elements
        n_extra = int(overlap * block_size)
        candidates = [e for e in universe if e not in subset]
        if candidates and n_extra > 0:
            extras = random.sample(candidates, min(n_extra, len(candidates)))
            subset.update(extras)

        subsets.append(subset)

    # Ensure feasibility
    covered = set().union(*subsets)
    uncovered = set(universe) - covered
    if uncovered:
        subsets.append(uncovered)

    return universe, subsets


def small_exact_instance(seed=None):
    """
    Return a small hand-crafted instance with a known optimal solution.
    Useful for verifying correctness.

    Universe : {0, 1, 2, 3, 4, 5}
    Subsets  :
        S0 = {0, 1, 2}
        S1 = {0, 3, 4}
        S2 = {1, 3, 5}
        S3 = {2, 4, 5}
        S4 = {0, 1, 2, 3, 4, 5}  ← trivial cover of size 1

    Optimal  : {S4} → size 1  (or {S0, S1, S3} → size 3 without S4)

    Returns:
        universe, subsets
    """
    universe = list(range(6))
    subsets = [
        {0, 1, 2},
        {0, 3, 4},
        {1, 3, 5},
        {2, 4, 5},
        {0, 1, 2, 3, 4, 5},
    ]
    return universe, subsets


#  Validators
def validate_instance(universe, subsets):
    """
    Check that the instance is well-formed and feasible.

    Returns:
        (bool, str) — (is_valid, message)
    """
    u_set = set(universe)

    for i, s in enumerate(subsets):
        if not s.issubset(u_set):
            extras = s - u_set
            return False, f"Subset {i} contains elements not in universe: {extras}"

    covered = set().union(*subsets) if subsets else set()
    uncovered = u_set - covered
    if uncovered:
        return False, f"Instance is infeasible: elements {uncovered} cannot be covered."

    return True, "Valid and feasible instance."


def validate_solution(universe, subsets, chosen_indices):
    """
    Check that a proposed solution actually covers the entire universe.

    Args:
        universe       : list of elements
        subsets        : list of all subsets
        chosen_indices : list of subset indices selected by the algorithm

    Returns:
        (bool, str) — (is_valid, message)
    """
    if not chosen_indices:
        return False, "Empty solution covers nothing."

    u_set = set(universe)
    covered = set()
    for i in chosen_indices:
        if i < 0 or i >= len(subsets):
            return False, f"Index {i} is out of range (only {len(subsets)} subsets)."
        covered |= subsets[i]

    uncovered = u_set - covered
    if uncovered:
        return False, f"Solution does not cover elements: {uncovered}"

    return True, f"Valid cover using {len(chosen_indices)} subset(s): indices {sorted(chosen_indices)}"


#  Display Helpers
def print_instance(universe, subsets, label="Set Cover Instance"):
    n = len(universe)
    m = len(subsets)
    print(f"\n{label}")
    print(f"  Universe  : {sorted(universe)}  (|U| = {n})")
    print(f"  Subsets   : {m} available")
    for i, s in enumerate(subsets):
        bar = "".join("█" if e in s else "░" for e in sorted(universe))
        print(f"    S{i:<3} [{bar}]  size={len(s):3d}  {sorted(s)}")
    print()


def print_solution(label, chosen_indices, subsets, universe, elapsed_ms):
    valid, msg = validate_solution(universe, subsets, chosen_indices)
    status = "✓" if valid else "✗"
    print(f"[{status}] {label}")
    print(f"    Cover size  : {len(chosen_indices)}")
    print(f"    Chosen sets : {sorted(chosen_indices)}")
    print(f"    Time        : {elapsed_ms:.4f} ms")
    if not valid:
        print(f"    WARNING     : {msg}")
    print()


#  Serialization (save/load instances as JSON)
def save_instance(universe, subsets, filepath):
    """Save a Set Cover instance to a JSON file."""
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    data = {
        "universe": sorted(universe),
        "subsets": [sorted(s) for s in subsets],
    }
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)


def load_instance(filepath):
    """Load a Set Cover instance from a JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)
    universe = data["universe"]
    subsets = [set(s) for s in data["subsets"]]
    return universe, subsets