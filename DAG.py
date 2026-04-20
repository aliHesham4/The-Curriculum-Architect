import re
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np



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


def get_topological_order(G):
    return list(nx.topological_sort(G))


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


def plot_dag(G,file_name="dag.png",title="Curriculum Prerequisite DAG"):
    def get_levels(G):
        levels = {}
        for node in nx.topological_sort(G):
            preds = list(G.predecessors(node))
            levels[node] = 0 if not preds else max(levels[p] for p in preds) + 1
        return levels

    levels   = get_levels(G)
    by_level = {}
    for node, lvl in levels.items():
        by_level.setdefault(lvl, []).append(node)

    pos = {}
    for lvl, nodes in by_level.items():
        n = len(nodes)
        for i, node in enumerate(nodes):
            pos[node] = ((i - (n - 1) / 2), -lvl)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_axis_off()

    COLORS = ["#B5D4F4", "#9FE1CB", "#CECBF6", "#FAC775"]
    node_colors = [COLORS[levels[n] % len(COLORS)] for n in G.nodes()]

    nx.draw_networkx_edges(
        G, pos, ax=ax,
        edge_color="#888780", arrows=True,
        arrowstyle="-|>", arrowsize=18,
        width=1.2, connectionstyle="arc3,rad=0.05",
        node_size=3500,
    )
    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_color=node_colors, node_size=3500,
        linewidths=0.8, edgecolors="#444441",
    )
    nx.draw_networkx_labels(
        G, pos, ax=ax,
        font_size=8, font_weight="500", font_color="#2C2C2A",
    )

    plt.title(title, fontsize=13, fontweight="500", pad=16)
    plt.tight_layout()
    plt.savefig(file_name, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"✅ Saved to {file_name}")