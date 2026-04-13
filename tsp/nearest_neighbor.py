import time
from tsp_utils import tour_cost, validate_tour


def nearest_neighbor(dist, start=0):
    """
    Run the Nearest Neighbor heuristic from a given start city.

    Args:
        dist  : n x n distance matrix (list of lists)
        start : starting city index (default 0)

    Returns:
        (cost, tour)
        cost : total tour distance
        tour : list of city indices starting and ending at `start`
               (note: returned as 0-indexed, starting city is `start`)
    """
    n = len(dist)
    visited = [False] * n
    tour = [start]
    visited[start] = True

    current = start
    for _ in range(n - 1):
        nearest = -1
        nearest_dist = float("inf")

        for city in range(n):
            if not visited[city] and dist[current][city] < nearest_dist:
                nearest_dist = dist[current][city]
                nearest = city

        tour.append(nearest)
        visited[nearest] = True
        current = nearest

    tour.append(start)  # return to starting city
    cost = tour_cost(tour, dist)
    return cost, tour


def nearest_neighbor_multistart(dist):
    """
    Run Nearest Neighbor from every city as the start point.
    Returns the best tour found across all n starting cities.

    Time complexity: O(n³) — n starts × O(n²) each

    Args:
        dist : n x n distance matrix

    Returns:
        (best_cost, best_tour, best_start)
    """
    n = len(dist)
    best_cost = float("inf")
    best_tour = None
    best_start = 0

    for start in range(n):
        cost, tour = nearest_neighbor(dist, start=start)
        if cost < best_cost:
            best_cost = cost
            best_tour = tour
            best_start = start

    # normalize tour to start at city 0 for consistency with other solvers
    best_tour = _normalize_tour(best_tour)
    return best_cost, best_tour, best_start


def _normalize_tour(tour):
    """
    Rotate a tour so that it starts (and ends) at city 0.
    e.g. [2, 0, 1, 3, 2] → [0, 1, 3, 2, 0]
    """
    if tour[0] == 0:
        return tour

    # find position of city 0 in the tour (excluding the last element which repeats start)
    inner = tour[:-1]  # remove the repeated start at the end
    idx = inner.index(0)
    rotated = inner[idx:] + inner[:idx]
    return rotated + [0]


#  Timed wrappers (used by benchmark.py)
def run(dist):
    """
    Run single-start Nearest Neighbor (from city 0) and return (cost, tour, elapsed_ms).
    """
    start = time.perf_counter()
    cost, tour = nearest_neighbor(dist, start=0)
    elapsed = (time.perf_counter() - start) * 1000
    return cost, tour, elapsed


def run_multistart(dist):
    """
    Run multi-start Nearest Neighbor and return (cost, tour, elapsed_ms).
    """
    start = time.perf_counter()
    cost, tour, _ = nearest_neighbor_multistart(dist)
    elapsed = (time.perf_counter() - start) * 1000
    return cost, tour, elapsed


#  Quick self-test
if __name__ == "__main__":
    from tsp_utils import print_matrix, print_result, random_euclidean_instance
    from dp_bitmask import tsp_dp

    print("=" * 55)
    print("  Nearest Neighbor Self-Test")
    print("=" * 55)

    # Test on a few random euclidean instances, compare with exact DP
    print("\nComparing Nearest Neighbor vs Held-Karp DP:\n")
    print(f"  {'n':>3}  {'Optimal':>9}  {'NN Single':>10}  {'Ratio':>6}  {'NN Multi':>9}  {'Ratio':>6}")
    print("  " + "-" * 55)

    for n in [5, 6, 7, 8, 9, 10, 12, 15]:
        coords, dist = random_euclidean_instance(n, seed=n * 7)

        # Exact (only for small n)
        if n <= 15:
            opt_cost, _ = tsp_dp(dist)
        else:
            opt_cost = None

        nn_cost, nn_tour, _ = run(dist)
        ms_cost, ms_tour, _ = run_multistart(dist)

        valid_nn, _ = validate_tour(nn_tour, n)
        valid_ms, _ = validate_tour(ms_tour, n)

        if opt_cost:
            ratio_nn = nn_cost / opt_cost
            ratio_ms = ms_cost / opt_cost
            print(f"  {n:>3}  {opt_cost:>9}  {nn_cost:>10}  {ratio_nn:>6.3f}  {ms_cost:>9}  {ratio_ms:>6.3f}"
                  f"  {'✓' if valid_nn else '✗'}{'✓' if valid_ms else '✗'}")
        else:
            print(f"  {n:>3}  {'N/A':>9}  {nn_cost:>10}  {'N/A':>6}  {ms_cost:>9}  {'N/A':>6}")

    print("\nNote: ratio = approximate_cost / optimal_cost. Lower is better. 1.0 = optimal.")