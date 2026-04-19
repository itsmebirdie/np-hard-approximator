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
├── app.py                      # Streamlit app for web-based results
├── .gitignore
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
├── plots/
│   ├── sc.py                   # csv to plots for Set Cover
│   ├── tsp.py                  # csv to plots for TSP
│   ├── sc/
│   │   └── (auto-generated plots go here)
│   └── tsp/
│       └── (auto-generated plots go here)
│
└── results/
    └── (auto-generated CSVs go here)
```


## Algorithms We're Implementing

### TSP (Travelling Salesman Problem)

**Problem:** Given `n` cities and distances between each pair, find the shortest tour that visits every city exactly once and returns to the start.

**Why it's NP-hard:** No known polynomial-time exact algorithm exists. Brute force checks all `(n-1)!` permutations.

| Algorithm | Type | Time Complexity | Feasible Up To |
|---|---|---|---|
| Brute Force | Exact | O(n!) | ~10 nodes |
| DP + Bitmask (Held-Karp) | Exact | O(n² × 2ⁿ) | ~20 nodes |
| Nearest Neighbor | Approximation | O(n²) | 1000+ nodes |
| Christofides | Approximation | O(n³) | 500+ nodes (bonus) |

**Held-Karp DP (the one to actually implement well):**  
- State: `dp[mask][i]` = minimum cost to reach city `i` having visited the set of cities encoded in `mask`  
- Transition: `dp[mask][i] = min over all j in mask: dp[mask ^ (1<<i)][j] + dist[j][i]`  
- Base case: `dp[1][0] = 0` (start at city 0)  
- Answer: `min over i: dp[(1<<n)-1][i] + dist[i][0]`

**Nearest Neighbor Heuristic:**  
- Start at city 0. At each step, go to the nearest unvisited city. Repeat until all visited. Return to start.  
- Approximation ratio: No guaranteed bound in general, but in practice gives ~20–25% above optimal.


### Set Cover

**Problem:** Given a universe `U` of `n` elements and a collection `S` of subsets, find the minimum number of subsets whose union equals `U`.

**Why it's NP-hard:** Deciding if a cover of size `k` exists is NP-complete.

| Algorithm | Type | Time Complexity | Approximation Ratio |
|---|---|---|---|
| Backtracking (exact) | Exact | Exponential | Optimal (1.0) |
| Greedy | Approximation | O(n × \|S\|) | H(n) ≈ ln(n) + 1 |

**Greedy Set Cover:**  
- Repeatedly pick the subset that covers the most currently uncovered elements.  
- The approximation ratio is `H(n)` where `H(n)` is the n-th harmonic number (~ln(n)).  
- This is actually the best possible polynomial approximation unless P = NP.


## How to Compute Approximation Ratio

This is the key analytical component of the project.

```
Approximation Ratio = (Cost of Approximate Solution) / (Cost of Optimal Solution)
```

- A ratio of **1.0** means the approximation found the optimal solution.  
- A ratio of **1.5** means the approximate solution is 50% worse than optimal.  
- For TSP Nearest Neighbor, we'll typically see ratios between 1.1 and 1.4 on random instances.  
- For Greedy Set Cover, the theoretical upper bound is `ln(n) + 1`, but in practice it's usually much better.


## Tech Stack

- **Language:** Python 3.10+
- **Libraries:** `itertools`, `numpy`, `matplotlib`, `time`, `csv`, `streamlit`, `plotly`, `pandas`, `networkx`
- **Optional:** `networkx` for graph visualization, `scipy` for MST in Christofides
- **Version Control:** Git + GitHub (public repo)
- **Dataset Source:** TSPLIB — http://comopt.ifi.uni-heidelberg.de/software/TSPLIB95/

```bash
pip install -r requirements.txt
```

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
| Nearest Neighbor | TSP | No constant bound | 1.1 – 1.4× |
| Christofides | Metric TSP | 1.5× | 1.1 – 1.3× |
| Greedy | Set Cover | H(n) ≈ ln(n) + 1 | Often 1.0 – 2.0× |
