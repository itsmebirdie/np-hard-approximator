import sys
import os
import time
import math

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ── Algorithm imports ──────────────────────────────────────────────────────────
from tsp.tsp_utils import random_euclidean_instance, tour_cost, validate_tour
from tsp.brute_force import tsp_brute_force
from tsp.dp_bitmask import tsp_dp
from tsp.nearest_neighbor import nearest_neighbor, nearest_neighbor_multistart
from tsp.christofides import christofides
from sc.sc_utils import random_instance, validate_solution
from sc.exact_backtracking import set_cover_exact
from sc.greedy_approx import (
    greedy_set_cover, greedy_set_cover_with_trace,
    theoretical_bound, compute_approximation_ratio
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NP-AA Visualizer",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main font & background */
    html, body, [class*="css"] { font-family: 'JetBrains Mono', 'Fira Code', monospace; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background: #0f1117; border-right: 1px solid #1e2130; }
    section[data-testid="stSidebar"] * { color: #e0e0e0 !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #1a1d2e;
        border: 1px solid #2a2d3e;
        border-radius: 8px;
        padding: 12px 16px;
    }
    div[data-testid="metric-container"] label { color: #8892b0 !important; font-size: 0.75rem; }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ccd6f6 !important; font-size: 1.4rem; font-weight: 700;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricDelta"] { font-size: 0.8rem; }

    /* Section headers */
    .section-header {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.15em;
        color: #64ffda; text-transform: uppercase;
        border-bottom: 1px solid #1e2130;
        padding-bottom: 6px; margin-bottom: 12px; margin-top: 4px;
    }

    /* Algorithm tags */
    .alg-tag {
        display: inline-block; padding: 2px 10px;
        border-radius: 12px; font-size: 0.72rem; font-weight: 600;
        margin: 2px;
    }
    .exact  { background: #1a2a1a; color: #50fa7b; border: 1px solid #50fa7b44; }
    .approx { background: #1a1a2a; color: #8be9fd; border: 1px solid #8be9fd44; }

    /* Result boxes */
    .result-box {
        background: #13141f;
        border: 1px solid #2a2d3e;
        border-left: 3px solid #64ffda;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
    }
    .result-box .alg-name { color: #ccd6f6; font-weight: 700; font-size: 0.9rem; }
    .result-box .cost     { color: #64ffda; font-size: 1.1rem; font-weight: 700; }
    .result-box .meta     { color: #8892b0; font-size: 0.75rem; margin-top: 4px; }

    /* Warning / info */
    .warn { color: #ffb86c; font-size: 0.8rem; }
    .info { color: #8be9fd; font-size: 0.8rem; }

    /* Tabs */
    button[data-baseweb="tab"] { font-size: 0.82rem; }

    /* Divider */
    hr { border-color: #1e2130 !important; }

    /* Dataframe */
    .stDataFrame { font-size: 0.78rem; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette for algorithms ──────────────────────────────────────────────
ALG_COLORS = {
    "Brute Force":       "#ff5555",
    "Held-Karp DP":      "#ffb86c",
    "Nearest Neighbor":  "#8be9fd",
    "NN Multi-Start":    "#50fa7b",
    "Christofides":      "#bd93f9",
    "Exact Backtrack":   "#ff5555",
    "Greedy":            "#8be9fd",
}

# ══════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def run_timed(fn, *args, **kwargs):
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    ms = (time.perf_counter() - t0) * 1000
    return result, ms


def tsp_fig_tour(coords, tour, title, color="#64ffda", show_order=True):
    """Build a Plotly figure for a TSP tour on 2D coordinates."""
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]

    fig = go.Figure()

    # Tour edges
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        fig.add_trace(go.Scatter(
            x=[coords[a][0], coords[b][0]],
            y=[coords[a][1], coords[b][1]],
            mode="lines",
            line=dict(color=color, width=2),
            showlegend=False,
            hoverinfo="skip",
        ))

    # Edge direction arrows (midpoints)
    for i in range(len(tour) - 1):
        a, b = tour[i], tour[i + 1]
        mx = (coords[a][0] + coords[b][0]) / 2
        my = (coords[a][1] + coords[b][1]) / 2
        fig.add_annotation(
            x=mx, y=my,
            ax=coords[a][0], ay=coords[a][1],
            axref="x", ayref="y",
            arrowhead=2, arrowsize=1, arrowwidth=1.5,
            arrowcolor=color, showarrow=True,
        )

    # City nodes
    labels = [f"City {i}<br>({coords[i][0]:.0f}, {coords[i][1]:.0f})" for i in range(len(coords))]
    node_colors = ["#ff5555" if i == 0 else "#ccd6f6" for i in range(len(coords))]
    node_sizes  = [14 if i == 0 else 10 for i in range(len(coords))]

    fig.add_trace(go.Scatter(
        x=xs, y=ys,
        mode="markers+text",
        marker=dict(color=node_colors, size=node_sizes,
                    line=dict(color="#0f1117", width=2)),
        text=[str(i) for i in range(len(coords))],
        textposition="top center",
        textfont=dict(color="#e0e0e0", size=11),
        hovertext=labels,
        hoverinfo="text",
        showlegend=False,
    ))

    if show_order:
        # Show visit order along the tour
        for step, city in enumerate(tour[:-1]):
            fig.add_annotation(
                x=coords[city][0], y=coords[city][1],
                text=str(step + 1),
                font=dict(size=8, color="#ffb86c"),
                showarrow=False,
                xshift=14, yshift=-10,
            )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccd6f6", size=14)),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#13141f",
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   color="#8892b0"),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False,
                   color="#8892b0", scaleanchor="x"),
        height=420,
    )
    return fig


def sc_fig_matrix(universe, subsets, chosen_indices=None, title="Set Cover Matrix"):
    """
    Heatmap showing which elements each subset covers.
    Chosen subsets are highlighted.
    """
    n_el  = len(universe)
    n_sub = len(subsets)
    chosen_set = set(chosen_indices or [])

    z    = []
    yticks = []
    for i in range(n_sub):
        row = [1 if e in subsets[i] else 0 for e in range(n_el)]
        z.append(row)
        marker = " ✓" if i in chosen_set else ""
        yticks.append(f"S{i}{marker}")

    # Color scale: grey for uncovered, green for covered by chosen, teal for other
    cell_colors = []
    for i in range(n_sub):
        row_colors = []
        for e in range(n_el):
            if subsets[i] and e in subsets[i]:
                if i in chosen_set:
                    row_colors.append("#50fa7b")
                else:
                    row_colors.append("#2a4a3a")
            else:
                row_colors.append("#1a1d2e")
        cell_colors.append(row_colors)

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=[f"e{e}" for e in range(n_el)],
        y=yticks,
        colorscale=[[0, "#1a1d2e"], [1, "#2a4a3a"]],
        showscale=False,
        hovertemplate="Subset %{y}<br>Element %{x}<br>Covered: %{z}<extra></extra>",
    ))

    # Overlay colored rectangles for chosen subsets
    for i in chosen_set:
        if i < n_sub:
            for e in range(n_el):
                if e in subsets[i]:
                    fig.add_shape(
                        type="rect",
                        x0=e - 0.5, x1=e + 0.5,
                        y0=i - 0.5, y1=i + 0.5,
                        fillcolor="#50fa7b33",
                        line=dict(color="#50fa7b", width=1),
                    )

    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccd6f6", size=14)),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#13141f",
        margin=dict(l=60, r=20, t=40, b=40),
        xaxis=dict(color="#8892b0", tickfont=dict(size=9)),
        yaxis=dict(color="#8892b0", tickfont=dict(size=9), autorange="reversed"),
        height=max(280, 28 * n_sub + 60),
    )
    return fig


def sc_fig_coverage_progress(trace, n_elements, title="Greedy Coverage Progress"):
    """Bar chart showing how many elements each greedy step covers."""
    if not trace:
        return go.Figure()

    steps    = [f"Step {t['step']}: S{t['chosen_idx']}" for t in trace]
    new_cov  = [t['n_newly_covered'] for t in trace]
    cum_cov  = [t['covered_so_far'] for t in trace]

    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(go.Bar(
        x=steps, y=new_cov,
        name="Newly covered",
        marker_color="#8be9fd",
        opacity=0.85,
    ), secondary_y=False)

    fig.add_trace(go.Scatter(
        x=steps, y=cum_cov,
        name="Cumulative covered",
        line=dict(color="#50fa7b", width=2.5),
        mode="lines+markers",
        marker=dict(size=7),
    ), secondary_y=True)

    fig.add_hline(y=n_elements, line_dash="dot",
                  line_color="#ff5555", annotation_text="Full coverage",
                  secondary_y=True)

    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccd6f6", size=14)),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#13141f",
        margin=dict(l=20, r=20, t=40, b=60),
        legend=dict(bgcolor="#13141f", font=dict(color="#e0e0e0", size=11)),
        xaxis=dict(color="#8892b0", tickangle=-25, tickfont=dict(size=9)),
        yaxis=dict(color="#8892b0", title="Newly covered"),
        yaxis2=dict(color="#50fa7b", title="Cumulative"),
        height=340,
    )
    return fig


def comparison_bar_chart(results, metric_key, metric_label, title):
    """Horizontal bar chart comparing a metric across algorithms."""
    algs   = [r["name"] for r in results]
    values = [r[metric_key] for r in results]
    colors = [ALG_COLORS.get(a, "#ccd6f6") for a in algs]

    fig = go.Figure(go.Bar(
        x=values, y=algs,
        orientation="h",
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color="#0f1117", width=1)),
        text=[f"{v:.3f}" if isinstance(v, float) else str(v) for v in values],
        textposition="outside",
        textfont=dict(color="#ccd6f6", size=11),
    ))
    fig.update_layout(
        title=dict(text=title, font=dict(color="#ccd6f6", size=14)),
        paper_bgcolor="#0f1117",
        plot_bgcolor="#13141f",
        margin=dict(l=20, r=60, t=40, b=20),
        xaxis=dict(color="#8892b0", title=metric_label,
                   gridcolor="#1e2130"),
        yaxis=dict(color="#8892b0"),
        height=60 + 50 * len(results),
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 🔬 NP-AA Visualizer")
    st.markdown("<div class='section-header'>Problem</div>", unsafe_allow_html=True)
    problem = st.radio(
        "Select problem",
        ["TSP — Travelling Salesman", "Set Cover"],
        label_visibility="collapsed",
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>Instance Parameters</div>", unsafe_allow_html=True)

    if problem.startswith("TSP"):
        n_cities = st.slider("Number of cities", min_value=3, max_value=20, value=8, step=1)
        seed     = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

        st.markdown("<div class='section-header'>Algorithms</div>", unsafe_allow_html=True)

        can_exact = n_cities <= 10
        can_dp    = n_cities <= 20

        if can_exact:
            use_bf = st.checkbox("Brute Force (Exact)", value=(n_cities <= 8))
        else:
            st.markdown("<span class='warn'>⚠ Brute Force: n>10, disabled</span>",
                        unsafe_allow_html=True)
            use_bf = False

        if can_dp:
            use_dp = st.checkbox("Held-Karp DP (Exact)", value=True)
        else:
            st.markdown("<span class='warn'>⚠ Held-Karp DP: n>20, disabled</span>",
                        unsafe_allow_html=True)
            use_dp = False

        use_nn  = st.checkbox("Nearest Neighbor", value=True)
        use_nms = st.checkbox("NN Multi-Start", value=True)
        use_ch  = st.checkbox("Christofides", value=True)

        selected_algs = {
            "Brute Force":      use_bf,
            "Held-Karp DP":     use_dp,
            "Nearest Neighbor": use_nn,
            "NN Multi-Start":   use_nms,
            "Christofides":     use_ch,
        }

    else:  # Set Cover
        n_elements = st.slider("Universe size |U|", min_value=4, max_value=30, value=12, step=1)
        n_sets     = st.slider("Number of subsets", min_value=n_elements,
                               max_value=n_elements * 5, value=n_elements * 3, step=1)
        coverage   = st.slider("Coverage density", min_value=0.10, max_value=0.70,
                               value=0.25, step=0.05,
                               help="Probability each element appears in a given subset")
        seed       = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

        st.markdown("<div class='section-header'>Algorithms</div>", unsafe_allow_html=True)
        can_exact  = n_elements <= 20
        if can_exact:
            use_exact = st.checkbox("Exact Backtracking", value=True)
        else:
            st.markdown("<span class='warn'>⚠ Exact BT: |U|>20, too slow</span>",
                        unsafe_allow_html=True)
            use_exact = False

        use_greedy = st.checkbox("Greedy Approximation", value=True)

        selected_algs = {
            "Exact Backtrack": use_exact,
            "Greedy":          use_greedy,
        }

    st.markdown("<hr>", unsafe_allow_html=True)
    run_btn = st.button("▶  Run", use_container_width=True, type="primary")

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>About</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='font-size:0.72rem; color:#8892b0; line-height:1.6'>
DAA Course Project — NP-AA<br>
NP-Hard Problem Solver &<br>Approximation Analyzer
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "results"      not in st.session_state: st.session_state.results      = None
if "problem_type" not in st.session_state: st.session_state.problem_type = None
if "instance"     not in st.session_state: st.session_state.instance     = None


# ══════════════════════════════════════════════════════════════════════════════
#  RUN ALGORITHMS
# ══════════════════════════════════════════════════════════════════════════════

if run_btn:
    with st.spinner("Solving..."):
        if problem.startswith("TSP"):
            coords, dist = random_euclidean_instance(n_cities, seed=int(seed))
            st.session_state.instance = {"coords": coords, "dist": dist, "n": n_cities}
            st.session_state.problem_type = "tsp"

            results = []

            if selected_algs.get("Brute Force"):
                (cost, tour), ms = run_timed(tsp_brute_force, dist)
                results.append({
                    "name": "Brute Force", "type": "exact",
                    "cost": cost, "tour": tour, "time_ms": ms,
                })

            if selected_algs.get("Held-Karp DP"):
                (cost, tour), ms = run_timed(tsp_dp, dist)
                results.append({
                    "name": "Held-Karp DP", "type": "exact",
                    "cost": cost, "tour": tour, "time_ms": ms,
                })

            if selected_algs.get("Nearest Neighbor"):
                (cost, tour), ms = run_timed(nearest_neighbor, dist, 0)
                results.append({
                    "name": "Nearest Neighbor", "type": "approx",
                    "cost": cost, "tour": tour, "time_ms": ms,
                })

            if selected_algs.get("NN Multi-Start"):
                (cost, tour, _), ms = run_timed(nearest_neighbor_multistart, dist)
                results.append({
                    "name": "NN Multi-Start", "type": "approx",
                    "cost": cost, "tour": tour, "time_ms": ms,
                })

            if selected_algs.get("Christofides"):
                (cost, tour), ms = run_timed(christofides, dist)
                results.append({
                    "name": "Christofides", "type": "approx",
                    "cost": cost, "tour": tour, "time_ms": ms,
                })

            # Compute approximation ratios using best exact as baseline
            exact_costs = [r["cost"] for r in results if r["type"] == "exact"]
            optimal = min(exact_costs) if exact_costs else None
            for r in results:
                r["ratio"] = (r["cost"] / optimal) if optimal else None

            st.session_state.results = results

        else:  # Set Cover
            universe, subsets = random_instance(n_elements, n_sets,
                                                coverage=coverage, seed=int(seed))
            st.session_state.instance = {
                "universe": universe, "subsets": subsets,
                "n": n_elements,
            }
            st.session_state.problem_type = "sc"

            results = []

            if selected_algs.get("Exact Backtrack"):
                (size, chosen), ms = run_timed(set_cover_exact, universe, subsets)
                results.append({
                    "name": "Exact Backtrack", "type": "exact",
                    "cover_size": size, "chosen": chosen, "time_ms": ms,
                })

            if selected_algs.get("Greedy"):
                (size, chosen, trace), ms = run_timed(
                    greedy_set_cover_with_trace, universe, subsets)
                results.append({
                    "name": "Greedy", "type": "approx",
                    "cover_size": size, "chosen": chosen,
                    "trace": trace, "time_ms": ms,
                })

            # Approximation ratios
            exact_sizes = [r["cover_size"] for r in results if r["type"] == "exact"]
            optimal = min(exact_sizes) if exact_sizes else None
            hn_bound = theoretical_bound(n_elements)
            for r in results:
                r["ratio"]    = compute_approximation_ratio(optimal, r["cover_size"])
                r["hn_bound"] = hn_bound

            st.session_state.results = results


# ══════════════════════════════════════════════════════════════════════════════
#  DISPLAY
# ══════════════════════════════════════════════════════════════════════════════

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='padding: 0.5rem 0 1rem 0'>
    <span style='font-size:1.6rem; font-weight:800; color:#ccd6f6'>NP-AA</span>
    <span style='font-size:1.1rem; color:#8892b0; margin-left:10px'>
        NP-Hard Problem Solver & Approximation Analyzer
    </span>
</div>
""", unsafe_allow_html=True)

if st.session_state.results is None:
    # Welcome screen
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
<div style='background:#13141f; border:1px solid #2a2d3e; border-radius:10px; padding:24px'>
<div style='color:#64ffda; font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px'>TSP — Travelling Salesman</div>
<div style='color:#ccd6f6; font-size:0.9rem; line-height:1.7'>
Given <b>n cities</b> with pairwise distances, find the shortest tour visiting every city exactly once and returning to the start.<br><br>
<span style='color:#8892b0'>Exact algorithms:</span><br>
&nbsp;• Brute Force — O(n!)<br>
&nbsp;• Held-Karp DP — O(n²·2ⁿ)<br><br>
<span style='color:#8892b0'>Approximation:</span><br>
&nbsp;• Nearest Neighbor — O(n²)<br>
&nbsp;• NN Multi-Start — O(n³)<br>
&nbsp;• Christofides — O(n³), 1.5× bound
</div>
</div>
""", unsafe_allow_html=True)

    with c2:
        st.markdown("""
<div style='background:#13141f; border:1px solid #2a2d3e; border-radius:10px; padding:24px'>
<div style='color:#64ffda; font-size:0.7rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; margin-bottom:10px'>Set Cover</div>
<div style='color:#ccd6f6; font-size:0.9rem; line-height:1.7'>
Given a universe <b>U</b> and subsets <b>S₁…Sₘ</b>, find the minimum collection of subsets whose union equals U.<br><br>
<span style='color:#8892b0'>Exact algorithms:</span><br>
&nbsp;• Backtracking with pruning — O(2ᵐ)<br><br>
<span style='color:#8892b0'>Approximation:</span><br>
&nbsp;• Greedy — O(n·m)<br>
&nbsp;• Ratio: H(n) = 1 + ½ + … + 1/n ≈ ln(n)<br>
&nbsp;• Optimal ratio (unless P=NP)
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("""
<div style='text-align:center; color:#8892b0; margin-top:30px; font-size:0.85rem'>
    ← Configure parameters in the sidebar and click <b style='color:#64ffda'>▶ Run</b>
</div>
""", unsafe_allow_html=True)
    st.stop()


# ── Results are available ──────────────────────────────────────────────────────
results      = st.session_state.results
instance     = st.session_state.instance
problem_type = st.session_state.problem_type

if not results:
    st.warning("No algorithms were selected. Enable at least one in the sidebar.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
#  TSP RESULTS
# ══════════════════════════════════════════════════════════════════════════════
if problem_type == "tsp":
    coords = instance["coords"]
    dist   = instance["dist"]
    n      = instance["n"]

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Summary</div>", unsafe_allow_html=True)

    metric_cols = st.columns(len(results))
    for col, r in zip(metric_cols, results):
        tag  = "exact" if r["type"] == "exact" else "approx"
        with col:
            delta_str = None
            if r["ratio"] is not None and r["ratio"] > 1.0:
                delta_str = f"+{(r['ratio'] - 1) * 100:.1f}% vs optimal"
            st.metric(
                label=r["name"],
                value=f"{r['cost']:,}",
                delta=delta_str,
                delta_color="inverse",
            )

    # ── Algorithm tags ─────────────────────────────────────────────────────────
    tags_html = ""
    for r in results:
        cls  = "exact" if r["type"] == "exact" else "approx"
        tags_html += f"<span class='alg-tag {cls}'>{r['name']}</span>"
    st.markdown(tags_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🗺 Tour Maps", "📊 Comparison", "🔢 Distance Matrix", "📋 Details"])

    # ── TAB 1: Tour Maps ───────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("<div class='section-header'>Tour Visualizations — Same Instance, All Algorithms</div>",
                    unsafe_allow_html=True)

        # Show all tours in a responsive grid (2 per row)
        pairs = [results[i:i+2] for i in range(0, len(results), 2)]
        for pair in pairs:
            cols = st.columns(len(pair))
            for col, r in zip(cols, pair):
                with col:
                    ratio_str = f"  |  ratio {r['ratio']:.3f}×" if r["ratio"] else ""
                    fig = tsp_fig_tour(
                        coords, r["tour"],
                        title=f"{r['name']} — cost {r['cost']:,}{ratio_str}",
                        color=ALG_COLORS.get(r["name"], "#64ffda"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

        # If more than one algorithm, show best tour highlighted
        if len(results) > 1:
            best = min(results, key=lambda x: x["cost"])
            st.markdown(f"<div class='info'>↑ Best tour found: <b>{best['name']}</b> with cost <b>{best['cost']:,}</b></div>",
                        unsafe_allow_html=True)

    # ── TAB 2: Comparison ─────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("<div class='section-header'>Algorithm Comparison</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)

        with c1:
            fig_cost = comparison_bar_chart(
                results, "cost", "Tour Cost",
                "Tour Cost by Algorithm (lower is better)",
            )
            st.plotly_chart(fig_cost, use_container_width=True)

        with c2:
            fig_time = comparison_bar_chart(
                [{**r, "time_ms": round(r["time_ms"], 4)} for r in results],
                "time_ms", "Runtime (ms)",
                "Runtime by Algorithm (ms)",
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # Approximation ratio chart (only when we have exact results)
        exact_results = [r for r in results if r["type"] == "exact"]
        if exact_results and any(r["ratio"] is not None for r in results):
            approx_results = [r for r in results if r["ratio"] is not None]
            fig_ratio = comparison_bar_chart(
                [{**r, "ratio_val": r["ratio"]} for r in approx_results],
                "ratio_val", "Ratio (cost / optimal)",
                "Approximation Ratio (1.0 = optimal)",
            )
            # Add reference line at 1.0
            fig_ratio.add_vline(x=1.0, line_dash="dot",
                                line_color="#64ffda",
                                annotation_text="Optimal", annotation_position="top")
            fig_ratio.add_vline(x=1.5, line_dash="dot",
                                line_color="#ffb86c",
                                annotation_text="1.5× (Christofides bound)",
                                annotation_position="top")
            st.plotly_chart(fig_ratio, use_container_width=True)

    # ── TAB 3: Distance Matrix ─────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("<div class='section-header'>Distance Matrix</div>",
                    unsafe_allow_html=True)

        import pandas as pd
        labels = [f"C{i}" for i in range(n)]
        df = pd.DataFrame(dist, index=labels, columns=labels)
        st.dataframe(df.style.background_gradient(cmap="Blues", axis=None)
                     .format("{:,.0f}"), use_container_width=True)

        st.markdown("<div class='section-header'>City Coordinates</div>",
                    unsafe_allow_html=True)
        coord_df = pd.DataFrame(
            [(i, f"{c[0]:.1f}", f"{c[1]:.1f}") for i, c in enumerate(coords)],
            columns=["City", "X", "Y"]
        )
        st.dataframe(coord_df, use_container_width=True, hide_index=True)

    # ── TAB 4: Details ─────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("<div class='section-header'>Per-Algorithm Details</div>",
                    unsafe_allow_html=True)

        for r in results:
            cls = "exact" if r["type"] == "exact" else "approx"
            ratio_str = f"ratio: {r['ratio']:.4f}×" if r["ratio"] else "ratio: N/A"
            tour_str = " → ".join(map(str, r["tour"]))
            st.markdown(f"""
<div class='result-box'>
  <span class='alg-name'>{r['name']}</span>
  <span class='alg-tag {cls}' style='margin-left:8px'>{r['type']}</span><br>
  <span class='cost'>{r['cost']:,}</span>
  <span class='meta'>tour cost</span><br>
  <div class='meta' style='margin-top:6px'>
    ⏱ {r['time_ms']:.4f} ms &nbsp;|&nbsp; {ratio_str}<br>
    🗺 {tour_str}
  </div>
</div>
""", unsafe_allow_html=True)

        # Summary table
        st.markdown("<div class='section-header'>Summary Table</div>",
                    unsafe_allow_html=True)
        import pandas as pd
        table_data = []
        for r in results:
            table_data.append({
                "Algorithm":       r["name"],
                "Type":            r["type"],
                "Tour Cost":       r["cost"],
                "Runtime (ms)":    round(r["time_ms"], 4),
                "Approx Ratio":    f"{r['ratio']:.4f}" if r["ratio"] else "N/A",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SET COVER RESULTS
# ══════════════════════════════════════════════════════════════════════════════
elif problem_type == "sc":
    universe = instance["universe"]
    subsets  = instance["subsets"]
    n        = instance["n"]

    # ── Summary metrics ────────────────────────────────────────────────────────
    st.markdown("<div class='section-header'>Summary</div>", unsafe_allow_html=True)

    metric_cols = st.columns(len(results) + 1)
    with metric_cols[0]:
        st.metric("Universe |U|", n)

    for col, r in zip(metric_cols[1:], results):
        with col:
            delta_str = None
            if r["ratio"] is not None and r["ratio"] > 1.0:
                delta_str = f"+{(r['ratio']-1)*100:.1f}% vs optimal"
            st.metric(
                label=r["name"],
                value=f"{r['cover_size']} sets",
                delta=delta_str,
                delta_color="inverse",
            )

    # H(n) bound info
    hn = theoretical_bound(n)
    st.markdown(
        f"<div class='info'>H({n}) theoretical bound = <b>{hn:.4f}×</b> &nbsp;|&nbsp; "
        f"Total subsets available: <b>{len(subsets)}</b></div>",
        unsafe_allow_html=True
    )

    tags_html = ""
    for r in results:
        cls = "exact" if r["type"] == "exact" else "approx"
        tags_html += f"<span class='alg-tag {cls}'>{r['name']}</span>"
    st.markdown(tags_html + "<br>", unsafe_allow_html=True)

    # ── Tabs ───────────────────────────────────────────────────────────────────
    tabs = st.tabs(["🗂 Coverage Matrix", "📊 Comparison", "📈 Greedy Trace", "📋 Details"])

    # ── TAB 1: Coverage Matrix ─────────────────────────────────────────────────
    with tabs[0]:
        st.markdown("<div class='section-header'>Subset Coverage Matrix — Same Instance, All Algorithms</div>",
                    unsafe_allow_html=True)

        if len(subsets) > 40:
            st.markdown("<div class='warn'>Large instance: showing first 40 subsets</div>",
                        unsafe_allow_html=True)
            disp_subsets = subsets[:40]
        else:
            disp_subsets = subsets

        cols = st.columns(len(results)) if len(results) > 1 else [st.container()]
        for col, r in zip(cols, results):
            with col:
                fig = sc_fig_matrix(
                    universe, disp_subsets,
                    chosen_indices=r["chosen"],
                    title=f"{r['name']} — {r['cover_size']} subsets chosen",
                )
                st.plotly_chart(fig, use_container_width=True)

        # Venn-style coverage summary
        st.markdown("<div class='section-header'>Element Coverage Summary</div>",
                    unsafe_allow_html=True)
        import pandas as pd
        cov_rows = []
        for r in results:
            covered = set()
            for idx in r["chosen"]:
                covered |= subsets[idx]
            cov_rows.append({
                "Algorithm":    r["name"],
                "Sets chosen":  r["cover_size"],
                "Elements covered": len(covered),
                "Coverage %":   f"{100*len(covered)/n:.1f}%",
                "Chosen indices": str(sorted(r["chosen"])),
            })
        st.dataframe(pd.DataFrame(cov_rows), use_container_width=True, hide_index=True)

    # ── TAB 2: Comparison ─────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown("<div class='section-header'>Algorithm Comparison</div>",
                    unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            fig_cov = comparison_bar_chart(
                results, "cover_size", "Cover Size (# subsets)",
                "Cover Size by Algorithm (lower is better)",
            )
            st.plotly_chart(fig_cov, use_container_width=True)

        with c2:
            fig_time = comparison_bar_chart(
                [{**r, "time_ms": round(r["time_ms"], 4)} for r in results],
                "time_ms", "Runtime (ms)",
                "Runtime by Algorithm (ms)",
            )
            st.plotly_chart(fig_time, use_container_width=True)

        # Ratio chart with H(n) bound
        if any(r["ratio"] is not None for r in results):
            ratio_results = [r for r in results if r["ratio"] is not None]
            fig_ratio = comparison_bar_chart(
                [{**r, "ratio_val": r["ratio"]} for r in ratio_results],
                "ratio_val", "Ratio (cover / optimal)",
                "Approximation Ratio (1.0 = optimal)",
            )
            fig_ratio.add_vline(x=1.0, line_dash="dot", line_color="#50fa7b",
                                annotation_text="Optimal")
            fig_ratio.add_vline(x=hn, line_dash="dot", line_color="#ffb86c",
                                annotation_text=f"H({n}) = {hn:.2f}")
            st.plotly_chart(fig_ratio, use_container_width=True)

    # ── TAB 3: Greedy Trace ────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown("<div class='section-header'>Greedy Algorithm Step-by-Step Trace</div>",
                    unsafe_allow_html=True)

        greedy_r = next((r for r in results if r["name"] == "Greedy"), None)
        if greedy_r and "trace" in greedy_r:
            trace = greedy_r["trace"]

            fig_prog = sc_fig_coverage_progress(trace, n,
                "Greedy Coverage Progress — Elements Covered Per Step")
            st.plotly_chart(fig_prog, use_container_width=True)

            st.markdown("<div class='section-header'>Step Log</div>", unsafe_allow_html=True)
            import pandas as pd
            trace_df = pd.DataFrame([{
                "Step":             t["step"],
                "Chosen subset":    f"S{t['chosen_idx']}",
                "Subset content":   str(t["subset_content"]),
                "Newly covered":    t["n_newly_covered"],
                "Covered so far":   t["covered_so_far"],
                "Remaining":        t["remaining_uncovered"],
                "Progress":         f"{100*t['covered_so_far']/n:.1f}%",
            } for t in trace])
            st.dataframe(trace_df, use_container_width=True, hide_index=True)
        else:
            st.info("Enable the Greedy algorithm to see the step-by-step trace.")

    # ── TAB 4: Details ─────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown("<div class='section-header'>Per-Algorithm Details</div>",
                    unsafe_allow_html=True)

        for r in results:
            cls = "exact" if r["type"] == "exact" else "approx"
            ratio_str  = f"{r['ratio']:.4f}×"  if r["ratio"]    else "N/A"
            bound_str  = f"{r['hn_bound']:.4f}" if r["hn_bound"] else "N/A"
            holds      = (r["ratio"] is not None and r["ratio"] <= r["hn_bound"] + 1e-9)
            holds_str  = "✓ Yes" if holds else "—"
            chosen_str = str(sorted(r["chosen"]))

            st.markdown(f"""
<div class='result-box'>
  <span class='alg-name'>{r['name']}</span>
  <span class='alg-tag {cls}' style='margin-left:8px'>{r['type']}</span><br>
  <span class='cost'>{r['cover_size']} subsets</span>
  <span class='meta'>in the cover</span><br>
  <div class='meta' style='margin-top:6px'>
    ⏱ {r['time_ms']:.4f} ms &nbsp;|&nbsp;
    ratio: {ratio_str} &nbsp;|&nbsp;
    H({n}) bound: {bound_str} &nbsp;|&nbsp;
    within bound: {holds_str}<br>
    📋 Chosen indices: {chosen_str}
  </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("<div class='section-header'>Summary Table</div>",
                    unsafe_allow_html=True)
        import pandas as pd
        table_data = []
        for r in results:
            holds = r["ratio"] is not None and r["ratio"] <= r["hn_bound"] + 1e-9
            table_data.append({
                "Algorithm":        r["name"],
                "Type":             r["type"],
                "Cover Size":       r["cover_size"],
                "Runtime (ms)":     round(r["time_ms"], 4),
                "Approx Ratio":     f"{r['ratio']:.4f}" if r["ratio"] else "N/A",
                f"H({n}) Bound":    f"{r['hn_bound']:.4f}",
                "Within Bound":     "✓" if holds else "—",
            })
        st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)