import random
import re
import math
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from matplotlib.patches import FancyArrowPatch, Circle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict



def build_networkx_dag(parsed):
    G = nx.DiGraph()

    for concept in parsed["concepts"]:
        c_name = concept["name"]
        G.add_node(c_name)
        for prereq in concept["prerequisites"]:
            G.add_node(prereq)
            G.add_edge(prereq, c_name)

    # ── Cycle detection ───────────────────────────────────────────────────────
    cycles = _detect_cycles_dfs(G)

    if cycles:
        print("\n  ❌ CIRCULAR DEPENDENCIES DETECTED")
        print("  ══════════════════════════════════════════════════════")
        for i, cycle in enumerate(cycles):
            print(f"  Cycle {i+1}: {' → '.join(cycle + [cycle[0]])}")
        print(f"\n  Total cycles found: {len(cycles)}")
        print("  ══════════════════════════════════════════════════════")

        # ── Ask user ──────────────────────────────────────────────────────────
        print("\n  Options:")
        print("  [1] Auto-fix — remove the most conflicted edge in each cycle")
        print("  [2] Abort   — raise an error and stop")
        choice = input("\n  Enter choice (1 or 2): ").strip()

        if choice == "1":
            G = _remove_cycles(G)
            print("\n  ✅ All cycles resolved. Continuing with fixed DAG.")
        else:
            cycle_report = "\n".join(
                f"  Cycle {i+1}: {' → '.join(cycle + [cycle[0]])}"
                for i, cycle in enumerate(cycles)
            )
            raise ValueError(
                f"\n\n  ❌ DAG construction aborted — {len(cycles)} cycle(s) detected:\n"
                f"{cycle_report}\n\n"
                f"  Fix the prerequisite relationships above and re-run."
            )

    return G


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMAL TOPOLOGICAL SORT  (greedy + random restarts)
# ═══════════════════════════════════════════════════════════════════════════════

def _greedy_ordering(G, objective="S1", seed=None):
    """
    Kahn's-style greedy: at each step, from all currently available nodes
    (in-degree 0 in the remaining graph), pick the one that minimises the
    chosen objective contribution.

    Tie-breaking is randomised so multiple calls can produce different orderings.
    """
    rng        = random.Random(seed)
    in_degree  = {n: G.in_degree(n) for n in G.nodes()}
    available  = [n for n, d in in_degree.items() if d == 0]
    ordering   = []
    pos        = {}           # position assigned so far

    while available:
        rng.shuffle(available)   # randomise ties

        best_node  = None
        best_cost  = float("inf")

        for candidate in available:
            # Tentatively place candidate at next position
            trial_pos = len(ordering) + 1

            # Cost = sum of distances from candidate to its already-placed
            # predecessors (those already in ordering)
            placed_preds = [u for u in G.predecessors(candidate) if u in pos]
            cost = sum(trial_pos - pos[u] for u in placed_preds)

            if cost < best_cost:
                best_cost = cost
                best_node = candidate

        ordering.append(best_node)
        pos[best_node] = len(ordering)

        # Update in-degrees
        for successor in G.successors(best_node):
            in_degree[successor] -= 1
            if in_degree[successor] == 0:
                available.append(successor)

        available.remove(best_node)

    return ordering


# ═══════════════════════════════════════════════════════════════════════════════
# MEASURE COMPUTATION  (Antunović & Vukičević, 2021)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ordering_measures(G, ordering):
    """
    Given a DAG G and a topological ordering (list of node names),
    computes the three per-vertex measures and their graph-level aggregates.

    Per vertex v:
        s_p(v) = sum   of distances from each incoming neighbour u to v
        a_p(v) = mean  of distances from each incoming neighbour u to v
        m_p(v) = max   of distances from each incoming neighbour u to v

    Graph-level (mean across vertices, i.e. α=1 / superscript 1):
        S¹  = mean of s_p(v)   — average total prerequisite span per concept
        A¹  = mean of a_p(v)   — average mean prerequisite distance
        M¹  = mean of m_p(v)   — average maximum prerequisite distance

    Graph-level (max across vertices, i.e. superscript ∞):
        S∞  = max  of s_p(v)
        A∞  = max  of a_p(v)
        M∞  = max  of m_p(v)
    """
    pos = {node: i + 1 for i, node in enumerate(ordering)}  # 1-indexed position

    s_v, a_v, m_v = {}, {}, {}

    for v in G.nodes():
        incoming = list(G.predecessors(v))
        if not incoming:
            s_v[v] = 0.0
            a_v[v] = 0.0
            m_v[v] = 0.0
            continue

        distances = [pos[v] - pos[u] for u in incoming]
        s_v[v] = sum(distances)
        a_v[v] = sum(distances) / len(distances)
        m_v[v] = max(distances)

    n = G.number_of_nodes()

    if n == 0:
        return {
            "s_per_vertex": {},
            "a_per_vertex": {},
            "m_per_vertex": {},
            "S1": 0.0, "A1": 0.0, "M1": 0.0,
            "S_inf": 0.0, "A_inf": 0.0, "M_inf": 0.0,
        }

    return {
        # Per-vertex dicts
        "s_per_vertex": s_v,
        "a_per_vertex": a_v,
        "m_per_vertex": m_v,

        # Mean across all vertices (superscript 1)
        "S1": sum(s_v.values()) / n,
        "A1": sum(a_v.values()) / n,
        "M1": sum(m_v.values()) / n,

        # Max across all vertices (superscript ∞)
        "S_inf": max(s_v.values()),
        "A_inf": max(a_v.values()),
        "M_inf": max(m_v.values()),
    }




def optimal_topological_sort(G, restarts=200, objective="S1", verbose=True):
    """
    Finds the topological ordering that minimises the chosen measure
    using repeated greedy search with random tie-breaking.

    Parameters
    ----------
    G         : nx.DiGraph  — must be a DAG
    restarts  : int         — number of random restarts (more = better, slower)
    objective : str         — measure to minimise: "S1", "A1", "M1",
                              "S_inf", "A_inf", "M_inf"

    Returns
    -------
    best_ordering : list of node names
    best_measures : dict with all six measures for that ordering
    history       : list of objective values across restarts (for analysis)
    """
    if not nx.is_directed_acyclic_graph(G):
        raise ValueError("Graph must be a DAG before optimising ordering.")

    best_ordering = None
    best_measures = None
    best_score    = float("inf")
    history       = []

    for i in range(restarts):
        ordering = _greedy_ordering(G, objective=objective, seed=i)
        measures = compute_ordering_measures(G, ordering)
        score    = measures[objective]
        history.append(score)

        if score < best_score:
            best_score    = score
            best_ordering = ordering
            best_measures = measures

    if verbose:
        print("\n══════════════════════════════════════════════════════")
        print(f"  OPTIMAL TOPOLOGICAL ORDERING  (objective: {objective})")
        print("══════════════════════════════════════════════════════")
        print(f"  Restarts:          {restarts}")
        print(f"  Best {objective:<6}:        {best_score:.4f}")
        print(f"  Worst seen:        {max(history):.4f}")
        print(f"  Mean across runs:  {sum(history)/len(history):.4f}")
        print()
        print("  Graph-level measures for best ordering:")
        print(f"    S¹  (mean total prereq span)   = {best_measures['S1']:.4f}")
        print(f"    A¹  (mean avg prereq distance) = {best_measures['A1']:.4f}")
        print(f"    M¹  (mean max prereq distance) = {best_measures['M1']:.4f}")
        print(f"    S∞  (max total prereq span)    = {best_measures['S_inf']:.4f}")
        print(f"    A∞  (max avg prereq distance)  = {best_measures['A_inf']:.4f}")
        print(f"    M∞  (max max prereq distance)  = {best_measures['M_inf']:.4f}")
        print()
        print("  Optimal learning sequence:")
        for i, concept in enumerate(best_ordering, 1):
            s = best_measures['s_per_vertex'][concept]
            a = best_measures['a_per_vertex'][concept]
            m = best_measures['m_per_vertex'][concept]
            n_prereqs = G.in_degree(concept)
            detail = f"s={s:.0f}, a={a:.2f}, m={m:.0f}" if n_prereqs else "no prerequisites"
            print(f"    {i:>2}. {concept:<40} [{detail}]")

    return best_ordering, best_measures, history

# ═══════════════════════════════════════════════════════════════════════════════


def _detect_cycles_dfs(G):
 
    WHITE, GRAY, BLACK = 0, 1, 2
    color  = {n: WHITE for n in G.nodes()}
    cycles = []

    def dfs(u, path):
        color[u] = GRAY
        path.append(u)
        for v in G.successors(u):
            if color[v] == GRAY:
                # Back edge found — extract the cycle
                cycle_start = path.index(v)
                cycles.append(path[cycle_start:])
            elif color[v] == WHITE:
                dfs(v, path)
        path.pop()
        color[u] = BLACK

    for node in G.nodes():
        if color[node] == WHITE:
            dfs(node, [])

    return cycles


def get_topological_order(G, optimal=True, objective="S1", restarts=200):
    """
    Returns topological order — optimal (minimised) if optimal=True,
    otherwise NetworkX default DFS order.
    """
    if optimal:
        ordering, _, _ = optimal_topological_sort(
            G, restarts=restarts, objective=objective, verbose=False
        )
        return ordering
    return list(nx.topological_sort(G))




def _remove_cycles(G):
    """
    Iteratively removes the edge involved in the most cycles
    until the graph is a valid DAG.
    """
    cycles = list(nx.simple_cycles(G))

    while cycles:
        edge_cycle_count = {}
        for cycle in cycles:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                if G.has_edge(u, v):
                    edge_cycle_count[(u, v)] = edge_cycle_count.get((u, v), 0) + 1

        if not edge_cycle_count:
            break

        most_conflicted = max(edge_cycle_count, key=edge_cycle_count.get)
        print(f"  ⚠  Removing edge: '{most_conflicted[0]}' → '{most_conflicted[1]}'"
              f"  (appeared in {edge_cycle_count[most_conflicted]} cycle(s))")
        G.remove_edge(*most_conflicted)

        cycles = list(nx.simple_cycles(G))

    return G


def print_dag_summary(G, optimal=True, objective="S1"):
    ordering = get_topological_order(G, optimal=optimal, objective=objective)
    measures = compute_ordering_measures(G, ordering)

    print("\n══════════════════════════════════════════════════════")
    print("   DAG SUMMARY")
    print("══════════════════════════════════════════════════════")
    print(f"  Nodes (concepts):    {G.number_of_nodes()}")
    print(f"  Edges (prerequisites): {G.number_of_edges()}")
    print(f"  Is DAG:              {nx.is_directed_acyclic_graph(G)}")
    print(f"  Ordering strategy:   {'Optimised (' + objective + ')' if optimal else 'Default DFS'}")

    print(f"\n  S¹ = {measures['S1']:.4f}  |  A¹ = {measures['A1']:.4f}  |  M¹ = {measures['M1']:.4f}")
    print(f"  S∞ = {measures['S_inf']:.4f}  |  A∞ = {measures['A_inf']:.4f}  |  M∞ = {measures['M_inf']:.4f}")

    print("\n  Learning Sequence:")
    for i, concept in enumerate(ordering, 1):
        print(f"    {i:>2}. {concept}")

    print("\n  Prerequisite Map:")
    for node in G.nodes():
        prereqs = list(G.predecessors(node))
        prereq_str = ", ".join(prereqs) if prereqs else "None"
        print(f"    {node} ← {prereq_str}")



def plot_dag(G, file_name="dag.png", title="Curriculum Prerequisite DAG",
             optimal=True, objective="S1", restarts=200):

    ordering, measures, _ = optimal_topological_sort(
        G, restarts=restarts, objective=objective, verbose=optimal
    )

    # ── 1. Assign hierarchical levels ────────────────────────────────────────
    def get_levels(ordering, G):
        levels = {}
        for node in ordering:
            preds = list(G.predecessors(node))
            levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
        return levels

    levels   = get_levels(ordering, G)
    by_level = {}
    for node, lvl in levels.items():
        by_level.setdefault(lvl, []).append(node)

    n_levels = max(levels.values()) + 1

    # ── 2. Compute node radius from longest label at each level ───────────────
    BASE_FONT   = 7.5          # font size in points
    CHAR_W      = 0.045        # approx width per character in data units
    PADDING     = 0.18         # extra padding around text
    H_GAP       = 0.55         # min horizontal gap between node edges
    V_GAP       = 1.15         # vertical distance between level centres

    def node_radius(label):
        # Wrap long labels
        words = label.split()
        lines, line = [], []
        for w in words:
            line.append(w)
            if len(" ".join(line)) > 18:
                lines.append(" ".join(line[:-1]))
                line = [w]
        lines.append(" ".join(line))
        max_chars = max(len(l) for l in lines)
        r_w = max_chars * CHAR_W / 2 + PADDING
        r_h = (len(lines) * BASE_FONT * 0.016) + PADDING
        return max(r_w, r_h, 0.22), lines   # (radius, wrapped lines)

    node_meta = {n: node_radius(n) for n in G.nodes()}  # {node: (r, lines)}

    # ── 3. Position nodes with proper spacing ─────────────────────────────────
    pos = {}
    for lvl, nodes in by_level.items():
        radii = [node_meta[n][0] for n in nodes]
        # Total width needed
        total_w = sum(2 * r for r in radii) + H_GAP * (len(nodes) - 1)
        x = -total_w / 2
        for node, r in zip(nodes, radii):
            pos[node] = (x + r, -lvl * V_GAP)
            x += 2 * r + H_GAP

    # ── 4. Figure setup ───────────────────────────────────────────────────────
    all_x = [p[0] for p in pos.values()]
    all_y = [p[1] for p in pos.values()]
    max_r = max(m[0] for m in node_meta.values())

    fig_w = max(14, (max(all_x) - min(all_x) + 2) * 1.5)
    fig_h = max(7,  (max(all_y) - min(all_y) + 2) * 1.4)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor("#FAFAF8")
    ax.set_facecolor("#FAFAF8")

    # ── 5. Color palette per level ────────────────────────────────────────────
    LEVEL_PALETTE = [
        ("#EAF3FB", "#185FA5", "#0C447C"),   # blue
        ("#E4F4EE", "#0F6E56", "#085041"),   # teal
        ("#EEEDFE", "#534AB7", "#3C3489"),   # purple
        ("#FEF3E2", "#854F0B", "#633806"),   # amber
        ("#FAE9E9", "#A32D2D", "#791F1F"),   # red
    ]
    def level_colors(lvl):
        return LEVEL_PALETTE[lvl % len(LEVEL_PALETTE)]

    # ── 6. Draw edges FIRST (behind nodes) ───────────────────────────────────
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        r0 = node_meta[u][0]
        r1 = node_meta[v][0]
           # Arrow colour matches SOURCE node's stroke colour
        _, source_stroke, _ = level_colors(levels[u])

        # Direction vector
        dx, dy  = x1 - x0, y1 - y0
        dist    = math.hypot(dx, dy)
        if dist == 0:
            continue
        ux, uy  = dx / dist, dy / dist

        # Start/end points on circle edges (not centres)
        sx, sy  = x0 + ux * r0, y0 + uy * r0
        ex, ey  = x1 - ux * r1, y1 - uy * r1

        # Slight curve to separate parallel edges
        ax.annotate(
            "",
            xy        = (ex, ey),
            xytext    = (sx, sy),
            arrowprops=dict(
                arrowstyle      = "-|>",
                color           = source_stroke, 
                lw              = 1.1,
                mutation_scale  = 14,
                connectionstyle = "arc3,rad=0.12",
                shrinkA         = 0,
                shrinkB         = 0,
            ),
            zorder=1,
        )

    # ── 7. Draw nodes ─────────────────────────────────────────────────────────
    for node in G.nodes():
        x, y         = pos[node]
        r, lines     = node_meta[node]
        lvl          = levels[node]
        fill, stroke, text_col = level_colors(lvl)

        # Shadow
        shadow = Circle((x + 0.018, y - 0.018), r,
                         facecolor="#CCCAC4", edgecolor="none",
                         zorder=2, alpha=0.35)
        ax.add_patch(shadow)

        # Node circle
        circle = Circle((x, y), r,
                         facecolor=fill,
                         edgecolor=stroke,
                         linewidth=1.2,
                         zorder=3)
        ax.add_patch(circle)

        # Text — vertically centred, multi-line
        line_h   = BASE_FONT * 0.018
        total_th = line_h * (len(lines) - 1)
        for i, line in enumerate(lines):
            ty = y + total_th / 2 - i * line_h
            ax.text(
                x, ty, line,
                ha="center", va="center",
                fontsize=BASE_FONT,
                fontweight="600",
                color=text_col,
                zorder=4,
                fontfamily="DejaVu Sans",
            )

    # ── 8. Level labels on left margin ───────────────────────────────────────
    for lvl, nodes in by_level.items():
        ys   = [pos[n][1] for n in nodes]
        y_c  = sum(ys) / len(ys)
        x_l  = min(all_x) - max_r - 0.55
        _, stroke, _ = level_colors(lvl)
        ax.text(
            x_l, y_c,
            f"Level {lvl}",
            ha="right", va="center",
            fontsize=7, color=stroke,
            fontweight="600",
            fontstyle="italic",
            fontfamily="DejaVu Sans",
        )
        # Dashed level separator line
        if lvl > 0:
            y_sep = -lvl * V_GAP + V_GAP / 2
            ax.axhline(
                y_sep,
                color="#DDDBD4", linewidth=0.6,
                linestyle="--", zorder=0, alpha=0.7,
            )

    # ── 9. Titles & measure subtitle ─────────────────────────────────────────
    fig.text(
        0.5, 0.97, title,
        ha="center", va="top",
        fontsize=14, fontweight="700",
        color="#2C2C2A",
        fontfamily="DejaVu Sans",
    )
    subtitle = (
        f"S¹ = {measures['S1']:.2f}   "
        f"A¹ = {measures['A1']:.2f}   "
        f"M¹ = {measures['M1']:.2f}   │   "
        f"S∞ = {measures['S_inf']:.0f}   "
        f"A∞ = {measures['A_inf']:.2f}   "
        f"M∞ = {measures['M_inf']:.0f}"
    )
    fig.text(
        0.5, 0.93, subtitle,
        ha="center", va="top",
        fontsize=8, color="#5F5E5A",
        fontfamily="DejaVu Sans",
    )

    # ── 10. Legend ────────────────────────────────────────────────────────────
    legend_handles = []
    seen_lvls = sorted(by_level.keys())
    for lvl in seen_lvls:
        fill, stroke, _ = level_colors(lvl)
        patch = mpatches.Patch(
            facecolor=fill, edgecolor=stroke,
            linewidth=1.2, label=f"Level {lvl}"
        )
        legend_handles.append(patch)

    ax.legend(
        handles     = legend_handles,
        loc         = "lower right",
        fontsize    = 7,
        framealpha  = 0.85,
        edgecolor   = "#DDDBD4",
        facecolor   = "#FAFAF8",
        title       = "Curriculum Level",
        title_fontsize = 7.5,
    )

    # ── 11. Axis limits with padding ──────────────────────────────────────────
    pad = max_r + 0.4
    ax.set_xlim(min(all_x) - max_r - 0.8, max(all_x) + max_r + 0.3)
    ax.set_ylim(min(all_y) - max_r - 0.4, max(all_y) + max_r + 0.5)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(file_name, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"✅ Saved to {file_name}")