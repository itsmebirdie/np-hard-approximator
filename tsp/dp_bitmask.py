import time
from .tsp_utils import tour_cost, validate_tour


def tsp_dp(dist):
    """
    Held-Karp bitmask DP for exact TSP.
    Always starts and ends at city 0.

    Args:
        dist : n x n distance matrix (list of lists)

    Returns:
        (min_cost, best_tour)
        min_cost  : int, optimal tour cost
        best_tour : list of city indices starting and ending at 0
    """
    n = len(dist)

    if n == 1:
        return 0, [0, 0]

    if n == 2:
        cost = dist[0][1] + dist[1][0]
        return cost, [0, 1, 0]

    INF = float("inf")
    SIZE = 1 << n  # 2^n

    # dp[mask][i] : min cost to reach city i having visited set `mask`
    # parent[mask][i] : the city we came from when we reached city i in state `mask`
    #                   — needed to reconstruct the actual tour path
    dp = [[INF] * n for _ in range(SIZE)]
    parent = [[-1] * n for _ in range(SIZE)]

    # Base case: start at city 0, only city 0 visited
    dp[1][0] = 0  # mask=1 means bit 0 is set (city 0 visited), cost 0

    # Fill DP table
    # Iterate over all possible subsets of cities
    for mask in range(1, SIZE):
        # Only process masks that include city 0 (bit 0 must be set)
        if not (mask & 1):
            continue

        for u in range(n):
            # u must be in the current mask
            if not (mask >> u & 1):
                continue
            if dp[mask][u] == INF:
                continue

            # Try extending the tour to every unvisited city v
            for v in range(n):
                if mask >> v & 1:
                    continue  # v already visited

                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]

                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    parent[new_mask][v] = u

    # All cities visited — find the best city to end at before returning to 0
    full_mask = SIZE - 1  # all bits set
    min_cost = INF
    last_city = -1

    for u in range(1, n):  # skip city 0 (can't end at start before returning)
        if dp[full_mask][u] == INF:
            continue
        total = dp[full_mask][u] + dist[u][0]
        if total < min_cost:
            min_cost = total
            last_city = u

    # Reconstruct the path by backtracking through `parent`
    tour = _reconstruct_path(parent, full_mask, last_city, n)

    return min_cost, tour


def _reconstruct_path(parent, full_mask, last_city, n):
    """
    Backtrack through the parent table to reconstruct the optimal tour.

    Args:
        parent    : parent[mask][city] = previous city
        full_mask : bitmask with all cities visited
        last_city : the last city before returning to 0
        n         : total number of cities

    Returns:
        tour : list of city indices [0, ..., last_city, 0]
    """
    path = []
    mask = full_mask
    current = last_city

    while current != -1:
        path.append(current)
        prev = parent[mask][current]
        mask ^= (1 << current)  # un-visit current city
        current = prev

    path.reverse()
    # path already starts with 0 (backtracking stops when parent == -1, which is city 0)
    # so we just append the return to start
    return path + [0]


# ─────────────────────────────────────────────
#  Timed wrapper (used by benchmark.py)
# ─────────────────────────────────────────────

def run(dist):
    """
    Run Held-Karp DP TSP and return (cost, tour, elapsed_ms).
    """
    start = time.perf_counter()
    cost, tour = tsp_dp(dist)
    elapsed = (time.perf_counter() - start) * 1000
    return cost, tour, elapsed


# ─────────────────────────────────────────────
#  Quick self-test
# ─────────────────────────────────────────────

if __name__ == "__main__":
    from tsp_utils import print_matrix, print_result, random_symmetric_matrix
    from brute_force import tsp_brute_force

    print("=" * 55)
    print("  Held-Karp DP Self-Test")
    print("=" * 55)

    # Test 1: known small example
    dist4 = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]
    print_matrix(dist4, "4-city Test Instance")
    cost_dp, tour_dp, ms_dp = run(dist4)
    print_result("Held-Karp DP", cost_dp, tour_dp, ms_dp)

    # Test 2: cross-validate against brute force on random instances
    print("Cross-validating DP vs Brute Force on random instances...\n")
    all_passed = True
    for n in range(3, 11):
        dist = random_symmetric_matrix(n, seed=n * 42)
        bf_cost, _, _ = __import__("brute_force").run(dist)
        dp_cost, dp_tour, _ = run(dist)

        valid, msg = validate_tour(dp_tour, n)
        match = bf_cost == dp_cost

        status = "PASS" if (match and valid) else "FAIL"
        print(f"  n={n:2d} | BF={bf_cost:5d}  DP={dp_cost:5d}  Match={str(match):<5}  Tour valid={str(valid):<5} [{status}]")
        if not match or not valid:
            all_passed = False

    print()
    if all_passed:
        print("All tests passed. Held-Karp DP matches brute force on n=3..10.")
    else:
        print("SOME TESTS FAILED. Check implementation.")