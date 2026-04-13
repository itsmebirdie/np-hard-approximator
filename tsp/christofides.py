import time
from tsp_utils import tour_cost, validate_tour

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False


#  Step 1: Minimum Spanning Tree (Prim's)
def prim_mst(dist):
    """
    Build a Minimum Spanning Tree using Prim's algorithm.

    Args:
        dist : n x n distance matrix

    Returns:
        mst_edges : list of (u, v, weight) tuples
        adj       : adjacency list — adj[u] = list of v connected to u in MST
    """
    n = len(dist)
    in_mst = [False] * n
    min_edge = [float("inf")] * n
    parent = [-1] * n

    min_edge[0] = 0
    mst_edges = []
    adj = [[] for _ in range(n)]

    for _ in range(n):
        # Pick the vertex not in MST with the smallest edge weight
        u = min((v for v in range(n) if not in_mst[v]), key=lambda v: min_edge[v])
        in_mst[u] = True

        if parent[u] != -1:
            w = dist[parent[u]][u]
            mst_edges.append((parent[u], u, w))
            adj[parent[u]].append(u)
            adj[u].append(parent[u])

        # Update edge weights for neighbors of u
        for v in range(n):
            if not in_mst[v] and dist[u][v] < min_edge[v]:
                min_edge[v] = dist[u][v]
                parent[v] = u

    return mst_edges, adj


#  Step 2: Find odd-degree vertices in MST
def find_odd_degree_vertices(adj, n):
    """
    Return list of vertices with odd degree in the MST adjacency list.
    """
    return [v for v in range(n) if len(adj[v]) % 2 == 1]

#  Step 3a: Greedy Minimum Weight Matching
#           (fallback when networkx not available)
def greedy_matching(odd_vertices, dist):
    """
    Greedy approximation for minimum weight perfect matching.
    Sort all pairs of odd-degree vertices by distance, greedily pick
    the shortest unmatched pair.

    Not optimal, but O(k² log k) where k = |odd_vertices|.
    In practice gives results close to optimal matching.

    Args:
        odd_vertices : list of vertex indices with odd degree
        dist         : full distance matrix

    Returns:
        matching : list of (u, v) pairs
    """
    remaining = set(odd_vertices)
    matching = []

    # Build all pairs sorted by distance
    pairs = []
    vlist = list(odd_vertices)
    for i in range(len(vlist)):
        for j in range(i + 1, len(vlist)):
            pairs.append((dist[vlist[i]][vlist[j]], vlist[i], vlist[j]))
    pairs.sort()

    for _, u, v in pairs:
        if u in remaining and v in remaining:
            matching.append((u, v))
            remaining.discard(u)
            remaining.discard(v)
            if not remaining:
                break

    return matching


#  Step 3b: True MWPM via NetworkX (if available)
def networkx_matching(odd_vertices, dist):
    """
    Minimum weight perfect matching using NetworkX's implementation.
    Requires networkx >= 2.6.

    Args:
        odd_vertices : list of vertex indices with odd degree
        dist         : full distance matrix

    Returns:
        matching : list of (u, v) pairs
    """
    G = nx.Graph()
    for i in range(len(odd_vertices)):
        for j in range(i + 1, len(odd_vertices)):
            u, v = odd_vertices[i], odd_vertices[j]
            G.add_edge(u, v, weight=dist[u][v])

    matched = nx.min_weight_matching(G)
    return list(matched)


#  Step 4: Build multigraph (MST + matching)
def build_multigraph(adj, matching, n):
    """
    Combine MST adjacency list with matching edges to form a multigraph.
    Every vertex will now have even degree (Eulerian condition).

    Returns:
        multi_adj : adjacency list allowing repeated edges (list of lists)
    """
    multi_adj = [list(neighbors) for neighbors in adj]
    for u, v in matching:
        multi_adj[u].append(v)
        multi_adj[v].append(u)
    return multi_adj


#  Step 5: Eulerian circuit (Hierholzer's algorithm)
def eulerian_circuit(multi_adj, n, start=0):
    """
    Find an Eulerian circuit in a multigraph using Hierholzer's algorithm.
    Assumes all vertices have even degree.

    Args:
        multi_adj : adjacency list (will be modified — pass a copy)
        n         : number of vertices
        start     : starting vertex

    Returns:
        circuit : list of vertex indices forming an Eulerian circuit
    """
    # Work on a mutable copy
    adj_copy = [list(neighbors) for neighbors in multi_adj]

    stack = [start]
    circuit = []

    while stack:
        v = stack[-1]
        if adj_copy[v]:
            u = adj_copy[v].pop()
            adj_copy[u].remove(v)  # remove the corresponding edge
            stack.append(u)
        else:
            circuit.append(stack.pop())

    circuit.reverse()
    return circuit


#  Step 6: Shortcut Eulerian circuit → Hamiltonian tour
def shortcut_to_hamiltonian(circuit):
    """
    Convert an Eulerian circuit to a Hamiltonian tour by skipping
    already-visited cities (shortcutting).

    By the triangle inequality, shortcuts never increase tour cost.

    Args:
        circuit : list of vertex indices (Eulerian circuit)

    Returns:
        tour : Hamiltonian tour (each city visited exactly once, returns to start)
    """
    visited = set()
    tour = []
    for city in circuit:
        if city not in visited:
            tour.append(city)
            visited.add(city)
    tour.append(tour[0])  # close the tour
    return tour


#  Main Christofides function
def christofides(dist):
    """
    Christofides-style approximation for metric TSP.
    Guaranteed 1.5× optimal for triangle-inequality instances
    (uses true MWPM if networkx is available, greedy matching otherwise).

    Args:
        dist : n x n distance matrix satisfying triangle inequality

    Returns:
        (cost, tour)
    """
    n = len(dist)

    if n == 1:
        return 0, [0, 0]
    if n == 2:
        return dist[0][1] + dist[1][0], [0, 1, 0]

    # Step 1: MST
    _, mst_adj = prim_mst(dist)

    # Step 2: Odd degree vertices
    odd_verts = find_odd_degree_vertices(mst_adj, n)

    # Step 3: Minimum weight perfect matching on odd-degree vertices
    if NETWORKX_AVAILABLE:
        matching = networkx_matching(odd_verts, dist)
    else:
        matching = greedy_matching(odd_verts, dist)

    # Step 4: Build multigraph
    multi_adj = build_multigraph(mst_adj, matching, n)

    # Step 5: Eulerian circuit
    circuit = eulerian_circuit(multi_adj, n, start=0)

    # Step 6: Shortcut to Hamiltonian tour
    tour = shortcut_to_hamiltonian(circuit)
    cost = tour_cost(tour, dist)

    return cost, tour


#  Timed wrapper (used by benchmark.py)
def run(dist):
    """
    Run Christofides and return (cost, tour, elapsed_ms).
    """
    start = time.perf_counter()
    cost, tour = christofides(dist)
    elapsed = (time.perf_counter() - start) * 1000
    return cost, tour, elapsed


#  Quick self-test
if __name__ == "__main__":
    from tsp_utils import print_matrix, print_result, random_euclidean_instance
    from dp_bitmask import tsp_dp

    matching_method = "NetworkX MWPM" if NETWORKX_AVAILABLE else "Greedy Matching (fallback)"
    print("=" * 55)
    print(f"  Christofides Self-Test  [{matching_method}]")
    print("=" * 55)

    print(f"\n  {'n':>3}  {'Optimal':>9}  {'Christofides':>13}  {'Ratio':>6}  {'≤1.5?':>6}")
    print("  " + "-" * 50)

    all_within_bound = True
    for n in [5, 6, 7, 8, 9, 10, 12, 15]:
        coords, dist = random_euclidean_instance(n, seed=n * 13)

        opt_cost, _ = tsp_dp(dist)
        ch_cost, ch_tour, _ = run(dist)

        valid, _ = validate_tour(ch_tour, n)
        ratio = ch_cost / opt_cost if opt_cost else float("inf")
        within = ratio <= 1.501  # small tolerance for rounding

        print(f"  {n:>3}  {opt_cost:>9}  {ch_cost:>13}  {ratio:>6.3f}  {'Yes ✓' if within else 'No  ✗'}")
        if not within:
            all_within_bound = False

    print()
    if all_within_bound:
        print("All tested instances within 1.5× bound.")
    else:
        print("Some instances exceeded 1.5× (expected if using greedy matching fallback).")

    print(f"\nnetworkx available: {NETWORKX_AVAILABLE}")
    if not NETWORKX_AVAILABLE:
        print("Install networkx for true MWPM: pip install networkx")