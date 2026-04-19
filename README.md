# NP-AA – NP-Hard Problem Solver & Approximation Analyzer

> A DAA Course Project

Maanas Nair \
Liza Borah \
Nisha Ahlawat

⚠️ THESE BENCHMARKS WERE TESTED ON A SPECIFIC HARDWARE DESCRIBED BELOW \
Intel i7-14700HX \
RTX5060 8GB \
32GB DDR5 RAM


## What This Project Is About

We're building a system that tackles two classic NP-hard problems — **Travelling Salesman Problem (TSP)** and **Set Cover** and analyzes how exact algorithms compare against approximation algorithms. The core idea is simple: exact solutions work fine for small inputs, but become computationally infeasible as input size grows. Approximation algorithms trade a bit of optimality for massive gains in speed. We implement both, measure both, and rigorously compare them.


## Project Structure

```
NP-AA/
├── README.md
├── requirements.txt
│
├── tsp/
│   ├── brute_force.py          # Exact TSP via permutations (tiny inputs, ≤10 nodes)
│   ├── dp_bitmask.py           # Exact TSP via DP + bitmask (small inputs, ≤20 nodes)
│   ├── nearest_neighbor.py     # Approximation: Nearest Neighbor heuristic
│   ├── christofides.py         # Advanced: Christofides algorithm (bonus)
│   └── tsp_utils.py            # Distance matrix generators, TSPLIB parser
│
├── sc/
│   ├── exact_backtracking.py   # Exact Set Cover via backtracking
│   ├── greedy_approx.py        # Greedy approximation (ln(n) bound)
│   └── sc_utils.py             # Random instance generators
│
└── results/
    └── (auto-generated CSVs and plots go here)
```


## Algorithms We're Implementing

### TSP (Travelling Salesman Problem)

**Problem:** Given `n` cities and distances between each pair, find the shortest tour that visits every city exactly once and returns to the start.

**Why it's NP-hard:** No known polynomial-time exact algorithm exists. Brute force checks all `(n-1)!` permutations.

| Algorithm | Type | Time Complexity | Feasible Up To |
|---|---|---|---|
| TBA | TBA | TBA | TBA |


### Set Cover

**Problem:** Given a universe `U` of `n` elements and a collection `S` of subsets, find the minimum number of subsets whose union equals `U`.

**Why it's NP-hard:** Deciding if a cover of size `k` exists is NP-complete.

| Algorithm | Type | Time Complexity | Approximation Ratio |
|---|---|---|---|
| TBA | TBA | TBA | TBA |



## How to Compute Approximation Ratio

This is the key analytical component of the project.

```
Approximation Ratio = (Cost of Approximate Solution) / (Cost of Optimal Solution)
```

- A ratio of **1.0** means the approximation found the optimal solution.  
- A ratio of **1.5** means the approximate solution is 50% worse than optimal.  
- For TSP Nearest Neighbor, you'll typically see ratios between 1.1 and 1.4 on random instances.  
- For Greedy Set Cover, the theoretical upper bound is `ln(n) + 1`, but in practice it's usually much better.


## Tech Stack

TBA

## Starter Code Reference

The project kit provides a brute-force TSP skeleton. Here's what a clean implementation structure looks like:

```python

def tsp_dp(dist):
    """
    Held-Karp algorithm for exact TSP.
    dist: n x n distance matrix (list of lists)
    Returns: (min_cost, optimal_path)
    """
    n = len(dist)
    INF = float('inf')
    
    # dp[mask][i] = min cost to reach city i, having visited cities in mask
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]
    dp[1][0] = 0  # start at city 0
    
    for mask in range(1, 1 << n):
        for u in range(n):
            if not (mask >> u & 1):
                continue
            if dp[mask][u] == INF:
                continue
            for v in range(n):
                if mask >> v & 1:
                    continue
                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + dist[u][v]
                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    parent[new_mask][v] = u
    
    full_mask = (1 << n) - 1
    min_cost = INF
    last = -1
    for u in range(1, n):
        cost = dp[full_mask][u] + dist[u][0]
        if cost < min_cost:
            min_cost = cost
            last = u
    
    # Reconstruct path
    path = []
    mask = full_mask
    while last != -1:
        path.append(last)
        prev = parent[mask][last]
        mask ^= (1 << last)
        last = prev
    path.reverse()
    
    return min_cost, [0] + path + [0]
```

## Quick Reference: Approximation Ratios

| Algorithm | Problem | Theoretical Bound | Typical Observed |
|---|---|---|---|
| TBA | TBA | TBA | TBA |
