import random
import math
import os

#  Distance Matrix Generators

def random_symmetric_matrix(n, low=1, high=100, seed=None):
    """
    Generate a random symmetric distance matrix of size n x n.
    dist[i][i] = 0, dist[i][j] = dist[j][i].

    Args:
        n    : number of cities
        low  : minimum edge weight (inclusive)
        high : maximum edge weight (inclusive)
        seed : random seed for reproducibility

    Returns:
        dist : list of lists (n x n)
    """
    if seed is not None:
        random.seed(seed)

    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            w = random.randint(low, high)
            dist[i][j] = w
            dist[j][i] = w
    return dist


def euclidean_matrix(coords):
    """
    Build a distance matrix from a list of (x, y) coordinates.
    Uses rounded Euclidean distance (standard for TSPLIB EUC_2D).

    Args:
        coords : list of (x, y) tuples

    Returns:
        dist : list of lists (n x n)
    """
    n = len(coords)
    dist = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = coords[i][0] - coords[j][0]
                dy = coords[i][1] - coords[j][1]
                dist[i][j] = int(round(math.sqrt(dx * dx + dy * dy)))
    return dist


def random_euclidean_instance(n, width=1000, height=1000, seed=None):
    """
    Generate a random set of n cities as (x, y) coordinates
    and return both the coordinates and the resulting distance matrix.

    Args:
        n      : number of cities
        width  : x-axis range [0, width]
        height : y-axis range [0, height]
        seed   : random seed

    Returns:
        coords : list of (x, y) tuples
        dist   : list of lists (n x n)
    """
    if seed is not None:
        random.seed(seed)

    coords = [(random.uniform(0, width), random.uniform(0, height)) for _ in range(n)]
    dist = euclidean_matrix(coords)
    return coords, dist


#  TSPLIB Parser (EUC_2D format only)

def parse_tsplib(filepath):
    """
    Parse a TSPLIB .tsp file in EUC_2D format.
    Only handles NODE_COORD_SECTION with EDGE_WEIGHT_TYPE: EUC_2D.

    Args:
        filepath : path to .tsp file

    Returns:
        name   : instance name (string)
        coords : list of (x, y) tuples (0-indexed)
        dist   : distance matrix (list of lists)

    Raises:
        ValueError if the file is not EUC_2D format.
    """
    name = "unknown"
    coords = []
    in_coord_section = False
    edge_weight_type = None

    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line == "EOF":
                break

            if line.startswith("NAME"):
                name = line.split(":")[1].strip() if ":" in line else line.split()[1]

            elif line.startswith("EDGE_WEIGHT_TYPE"):
                edge_weight_type = line.split(":")[1].strip() if ":" in line else line.split()[1]

            elif line == "NODE_COORD_SECTION":
                in_coord_section = True

            elif in_coord_section:
                parts = line.split()
                if len(parts) >= 3:
                    # parts[0] is node index (1-based), parts[1] and parts[2] are x, y
                    x, y = float(parts[1]), float(parts[2])
                    coords.append((x, y))

    if edge_weight_type not in ("EUC_2D", None):
        raise ValueError(f"Unsupported EDGE_WEIGHT_TYPE: {edge_weight_type}. Only EUC_2D is supported.")

    dist = euclidean_matrix(coords)
    return name, coords, dist


#  Tour Validator

def validate_tour(tour, n):
    """
    Check that a tour is a valid Hamiltonian cycle.
    A valid tour visits all n cities exactly once and returns to start.

    Args:
        tour : list of city indices (should start and end at 0)
        n    : number of cities

    Returns:
        (bool, str) — (is_valid, message)
    """
    if len(tour) != n + 1:
        return False, f"Tour length {len(tour)} is wrong. Expected {n + 1} (n cities + return to start)."
    if tour[0] != tour[-1]:
        return False, f"Tour does not form a cycle: starts at {tour[0]}, ends at {tour[-1]}."
    visited = set(tour[:-1])
    if len(visited) != n:
        return False, f"Tour visits {len(visited)} unique cities, expected {n}."
    if visited != set(range(n)):
        missing = set(range(n)) - visited
        return False, f"Tour is missing cities: {missing}."
    return True, "Valid tour."


def tour_cost(tour, dist):
    """
    Calculate the total cost of a given tour.

    Args:
        tour : list of city indices (starts and ends at same city)
        dist : distance matrix

    Returns:
        total cost (int or float)
    """
    return sum(dist[tour[i]][tour[i + 1]] for i in range(len(tour) - 1))


#  Display Helpers

def print_matrix(dist, label="Distance Matrix"):
    n = len(dist)
    print(f"\n{label} ({n}x{n}):")
    header = "     " + "  ".join(f"{j:4d}" for j in range(n))
    print(header)
    print("     " + "-" * (6 * n))
    for i in range(n):
        row = "  ".join(f"{dist[i][j]:4d}" for j in range(n))
        print(f"  {i:2d} | {row}")
    print()


def print_result(algorithm_name, cost, tour, elapsed_ms):
    valid, msg = validate_tour(tour, len(tour) - 1)
    status = "✓" if valid else "✗"
    print(f"[{status}] {algorithm_name}")
    print(f"    Cost    : {cost}")
    print(f"    Tour    : {' → '.join(map(str, tour))}")
    print(f"    Time    : {elapsed_ms:.4f} ms")
    if not valid:
        print(f"    WARNING : {msg}")
    print()