import json
import matplotlib.pyplot as plt
from dataclasses import dataclass
from PO_RRT_Star_EXACT import spectral_cluster_paths
from visualization import plot_paths_summary, init_progress_plot_3d, interactive_spectral_cluster_plot, plot_paths_metrics

@dataclass
class RNode:
    x: float; y: float; cost: float; p_fail: float

class RPath:
    def __init__(self, nodes): self.nodes = nodes
    @property
    def cost(self): return self.nodes[-1].cost if self.nodes else 0.0
    @property
    def p_fail(self): return self.nodes[-1].p_fail if self.nodes else 0.0

def load_paths_entries(json_paths):
    entries = []
    for e in json_paths:
        nodes = [RNode(**n) for n in e["nodes"]]
        entries.append({"path": RPath(nodes), "cost": e["cost"], "p_fail": e["p_fail"]})
    return entries

def show_paths(json_file, which="filtered"):
    data = json.load(open(json_file))
    entries = load_paths_entries(data["paths"][which])

    # First: show summary (Pareto front left, Spatial paths right)
    plot_paths_summary(entries, obstacles=data.get("obstacles"))

    # Optional: interactive clustering prompt
    # do_spec = input("Open spectral clustering viewer? (y/N): ").strip().lower() == 'y'
    # if do_spec:
    #     interactive_spectral_cluster_plot(
    #         entries,
    #         spectral_cluster_paths,
    #         obstacles=data.get("obstacles")
    #     )

def show_pareto(json_file, which="filtered"):
    """
    Shows only the Pareto front (Cost vs Failure Probability scatter plot).
    """
    data = json.load(open(json_file))
    entries = load_paths_entries(data["paths"][which])
    
    # Uses the existing plot_paths_metrics function from visualization.py
    plot_paths_metrics(entries)

def show_tree(json_file, which=None):
    """
    Show the whole saved tree, and optionally highlight paths in green.

    :param json_file: exported porrt_export_*.json
    :param which: "filtered" or "multiple" or None
    """
    data = json.load(open(json_file))
    start = tuple(data["meta"]["start"])
    goal  = tuple(data["meta"]["goal"])
    w, h  = data["meta"]["grid_size"]
    obs   = data.get("obstacles")
    edges = data["tree"]["edges"]  # always [[x,y,z], [x2,y2,z2]]

    # Rebuild any paths we want to highlight
    highlight_entries = []
    if which is not None:
        json_paths = data["paths"].get(which, [])
        highlight_entries = load_paths_entries(json_paths)

    # Initialize 3D plot
    fig, ax, lc, edge_segments = init_progress_plot_3d(
        start,
        goal,
        x_lim=(0, w),
        y_lim=(0, h),
        obstacles=obs,
        z_lim=(0.0, 1.0),
    )

    # 1) Tree segments (whole tree) – gray
    tree_segments = [
        ((a[0], a[1], a[2]), (b[0], b[1], b[2]))
        for (a, b) in edges
    ]

    # 2) Highlighted path segments – green
    highlight_segments = []
    for entry in highlight_entries:
        path = entry["path"]
        nodes = path.nodes
        for i in range(1, len(nodes)):
            n1, n2 = nodes[i - 1], nodes[i]
            seg = (
                (n1.x, n1.y, n1.p_fail),
                (n2.x, n2.y, n2.p_fail),
            )
            highlight_segments.append(seg)

    # 3) Combine and color
    all_segments = tree_segments + highlight_segments
    lc.set_segments(all_segments)

    if highlight_segments:
        colors = ['gray'] * len(tree_segments) + ['green'] * len(highlight_segments)
        lc.set_color(colors)

    plt.show(block=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="porrt_export_*.json")
    ap.add_argument("--tree", action="store_true", help="show whole tree")
    ap.add_argument("--pareto", action="store_true", help="show only the Pareto front scatter plot")
    ap.add_argument("--which", choices=["filtered", "multiple"], default="filtered")
    args = ap.parse_args()

    if args.pareto:
        show_pareto(args.file, which=args.which)
    elif args.tree:
        show_tree(args.file, which=args.which)
    else:
        show_paths(args.file, which=args.which)