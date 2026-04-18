import re
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import numpy as np


def get_topological_order(G):
    return list(nx.topological_sort(G))


def build_networkx_dag(parsed):
    G = nx.DiGraph()

    # ── Add all concept nodes and edges ──────────────────────────────────────
    for concept in parsed["concepts"]:
        c_name = concept["name"]
        G.add_node(c_name)

        for prereq in concept["prerequisites"]:
            G.add_node(prereq)
            # Edge direction: prereq → concept (must learn prereq first)
            G.add_edge(prereq, c_name)

    # ── Remove cycles to guarantee a valid DAG ────────────────────────────────
    # Strategy: find cycles, remove the edge that appears most frequently
    # across all cycles (most problematic edge), repeat until clean.
    try:
        cycles = list(nx.simple_cycles(G))

        while cycles:
            # Count how often each edge appears across all cycles
            edge_cycle_count = {}
            for cycle in cycles:
                for i in range(len(cycle)):
                    u = cycle[i]
                    v = cycle[(i + 1) % len(cycle)]
                    if G.has_edge(u, v):
                        edge_cycle_count[(u, v)] = edge_cycle_count.get((u, v), 0) + 1

            # Remove the edge involved in the most cycles
            if edge_cycle_count:
                most_conflicted = max(edge_cycle_count, key=edge_cycle_count.get)
                print(f"  ⚠ Removing cycle edge: {most_conflicted[0]} → {most_conflicted[1]}")
                G.remove_edge(*most_conflicted)

            cycles = list(nx.simple_cycles(G))

    except Exception as e:
        print(f"  ⚠ Cycle removal error: {e}")

    return G


def print_dag_summary(G):
    print("\n══════════════════════════════════════════════════════")
    print("   DAG SUMMARY")
    print("══════════════════════════════════════════════════════")
    print(f"  Nodes (concepts): {G.number_of_nodes()}")
    print(f"  Edges (prerequisites): {G.number_of_edges()}")
    print(f"  Is DAG: {nx.is_directed_acyclic_graph(G)}")

    print("\n  Topological Order (learning sequence):")
    for i, concept in enumerate(get_topological_order(G), 1):
        print(f"    {i}. {concept}")

    print("\n  Prerequisite Map:")
    for node in G.nodes():
        prereqs = list(G.predecessors(node))
        prereq_str = ", ".join(prereqs) if prereqs else "None"
        print(f"    {node} ← {prereq_str}")




def plot_dag(G):
    # ── Manual hierarchical layout (no graphviz needed) ───────────────────────
    # Assign levels by longest path from a root
    def get_levels(G):
        levels = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
        return levels

    levels     = get_levels(G)
    max_level  = max(levels.values())
    by_level   = {}
    for node, lvl in levels.items():
        by_level.setdefault(lvl, []).append(node)

    # Assign x/y positions
    pos = {}
    for lvl, nodes in by_level.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i - (n - 1) / 2), -lvl)   # centre each level

    # ── Draw ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axis_off()

    COLORS = ["#B5D4F4", "#9FE1CB", "#CECBF6", "#FAC775"]

    node_colors = [COLORS[levels[n] % len(COLORS)] for n in G.nodes()]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#888780",
        arrows=True,
        arrowstyle="-|>",
        arrowsize=18,
        width=1.2,
        connectionstyle="arc3,rad=0.05",   # slight curve to avoid overlap
        node_size=3500,                     # so arrows stop at node edge
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors,
        node_size=3500,
        linewidths=0.8,
        edgecolors="#444441",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8,
        font_weight="500",
        font_color="#2C2C2A",
    )

    plt.title("Curriculum Prerequisite DAG", fontsize=13, fontweight="500", pad=16)
    plt.tight_layout()
    plt.savefig("dag.png", dpi=180, bbox_inches="tight")
    plt.show()
    print("✅ Saved to dag.png")