import itertools
import time
from tsp_utils import tour_cost, validate_tour


def tsp_brute_force(dist):
    """
    Find the optimal TSP tour by checking all (n-1)! permutations.
    Always starts and ends at city 0.

    Args:
        dist : n x n distance matrix (list of lists)

    Returns:
        (min_cost, best_tour)
        min_cost  : integer, total distance of optimal tour
        best_tour : list of city indices starting and ending at 0
    """
    n = len(dist)

    if n == 1:
        return 0, [0, 0]

    if n == 2:
        cost = dist[0][1] + dist[1][0]
        return cost, [0, 1, 0]

    cities = list(range(1, n))  # fix city 0 as start, permute the rest
    min_cost = float("inf")
    best_tour = None

    for perm in itertools.permutations(cities):
        tour = [0] + list(perm) + [0]
        cost = tour_cost(tour, dist)
        if cost < min_cost:
            min_cost = cost
            best_tour = tour

    return min_cost, best_tour


#  Timed wrapper (used by benchmark.py)
def run(dist):
    """
    Run brute force TSP and return (cost, tour, elapsed_ms).
    """
    start = time.perf_counter()
    cost, tour = tsp_brute_force(dist)
    elapsed = (time.perf_counter() - start) * 1000  # convert to ms
    return cost, tour, elapsed


#  Quick self-test
if __name__ == "__main__":
    from tsp_utils import print_matrix, print_result

    # 4-city example — optimal answer is known: 0→1→3→2→0 = 10+25+30+15 = 80
    dist = [
        [0, 10, 15, 20],
        [10, 0, 35, 25],
        [15, 35, 0, 30],
        [20, 25, 30, 0],
    ]

    print_matrix(dist, "4-city Test Instance")
    cost, tour, ms = run(dist)
    print_result("Brute Force", cost, tour, ms)

    valid, msg = validate_tour(tour, len(dist))
    assert valid, f"Tour validation failed: {msg}"
    print(f"Self-test passed. Optimal cost = {cost}")