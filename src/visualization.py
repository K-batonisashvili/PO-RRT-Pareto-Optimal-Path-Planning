import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.collections import LineCollection 
import matplotlib.colors as mcolors


def init_progress_plot_2d(start, goal, x_lim, y_lim, obstacles):
    """
    Initialize a 2D progress plot for the RRT* tree.

    Returns:
    - fig, ax: Matplotlib Figure and 2D Axes
    - lc: LineCollection for tree edges
    - edge_segments: mutable list of (segment, color) tuples
    """
    fig, ax = plt.subplots()

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect('equal', adjustable='box')

    # Plot start and goal
    ax.scatter(start[0], start[1], c='red', s=60, label='Start')
    ax.scatter(goal[0], goal[1], c='red', s=80, marker='*', label='Goal')

    # Plot obstacles (2D)
    if obstacles is not None:
        for obs in obstacles:
            if obs.get("type") == "circular":
                cx, cy = obs["center"]
                radius = obs["radius"]
                theta = np.linspace(0, 2 * np.pi, 100)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                ax.plot(x, y, linestyle='--', alpha=0.5, color='red')
            elif obs.get("type") == "rectangular":
                # x_range / y_range style (matches other parts of the code)
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                xs = [x0, x1, x1, x0, x0]
                ys = [y0, y0, y1, y1, y0]
                ax.plot(xs, ys, linestyle='--', alpha=0.5, color='orange')

    # Empty line collection for tree edges
    lc = LineCollection([], linewidths=1.0)
    ax.add_collection(lc)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("PORRT* (2D)")
    ax.grid(True)
    ax.legend(loc="best")

    # We will keep this as a list of (segment, color) tuples
    edge_segments = []
    return fig, ax, lc, edge_segments



def init_progress_plot_3d(start, goal, x_lim, y_lim, obstacles, z_lim=(0.0, 1.0)):
    """
    Initialize a 3D progress plot for RRT*.

    Returns:
    - fig, ax: Matplotlib Figure and 3D Axes
    - lc: Line3DCollection for tree edges
    - edge_segments: mutable list of (segment, color) tuples
    """
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    # Axis settings
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability of Failure')
    ax.set_title("PORRT* (3D)")

    # Plot start and goal
    ax.scatter(start[0], start[1], 0, c='red', s=60, label='Start')
    ax.scatter(goal[0], goal[1], 0, c='red', s=80, label='Goal')

    # Plot obstacles
    if obstacles is not None:
        for obs in obstacles:
            if obs.get("type") == "circular":
                cx, cy = obs["center"]
                radius = obs["radius"]
                theta = np.linspace(0, 2 * np.pi, 100)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                z = np.zeros_like(x)
                ax.plot(x, y, z, color='red', alpha=0.7)
            elif obs.get("type") == "rectangular":
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                x_bounds = [x0, x1, x1, x0, x0]
                y_bounds = [y0, y0, y1, y1, y0]
                z = np.zeros_like(x_bounds)
                ax.plot_trisurf(x_bounds, y_bounds, z, color='orange', alpha=0.3)

    ax.legend()

    # Tree edge visualization
    edge_segments = []  # list of (segment, color) tuples
    lc = Line3DCollection([], linewidths=1.5, alpha=0.7)
    ax.add_collection(lc)

    plt.ion()
    plt.show()

    return fig, ax, lc, edge_segments

def update_progress_plot_3d(lc, edge_segments, parent_node, child_node, remove=False, pause_time=0.001):
    """
    Update the 3D progress plot with new edges or remove old edges.
    """
    edge = [(parent_node.x, parent_node.y, parent_node.p_fail),
            (child_node.x, child_node.y, child_node.p_fail)]
    
    if remove:
        # Check if the edge exists before removing
        if edge in edge_segments:
            edge_segments.remove(edge)
        # else:
        #     print(f"No edge between {parent_node} and {child_node} to remove.")
    else:
        # Add the edge between parent_node and child_node
        edge_segments.append(edge)

    # Update the line collection
    lc.set_segments(edge_segments)
    plt.pause(pause_time)

def plot_paths_metrics(paths):
    """
    Scatter-plot cost vs. failure probability for each complete path.

    Parameters:
    - paths: list of path objects with `cost` and `p_fail` attributes.
    """
    # Extract metrics
    all_path = [entry ["path"] for entry in paths]
    costs  = [entry["cost"] for entry in paths]
    pfails = [entry["p_fail"] for entry in paths]   
    
    # Create a figure and axis for the plot
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.clear()
    ax.scatter(costs, pfails, marker='o', color='blue', label='Paths')
    ax.set_xlabel('Total Cost')
    ax.set_ylabel('Failure Probability')
    ax.set_title('Cost vs. Failure Probability for Extracted Paths')
    ax.grid(True)
    ax.legend()
    plt.show(block=True)
    print("Paths metrics:")

def plot_full_paths(paths):
    """
    Plot the full path(s) from start to goal in 2D.
    Each path is plotted as a green line.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, entry in enumerate(paths):
        path = entry["path"]
        nodes = path.nodes if hasattr(path, "nodes") else path

        xs = [node.x for node in nodes]
        ys = [node.y for node in nodes]

        # Plot the path in green
        ax.plot(xs, ys, color='green', linewidth=2, marker='o', label=f'Path {i+1}' if i == 0 else None)

        # Start and goal in red
        ax.scatter(xs[0], ys[0], c='red', s=50, marker='o', label='Start' if i == 0 else "")
        ax.scatter(xs[-1], ys[-1], c='red', s=80, marker='*', label='Goal' if i == 0 else "")

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Full Completed Paths (Green), Start/Goal (Red)')
    ax.grid(True)
    ax.legend()
    plt.show(block=True)


def plot_paths_summary(paths, obstacles=None):
    """
    Show cost vs. failure probability and full path(s) from start to goal side by side.
    Only the top 10 paths with the least cost are shown.
    Each path is given a unique color in both plots.
    Optionally plots obstacles on the right plot if obstacles is provided.
    """
    # Sort by p_fail (increasing)
    if len(paths) <= 15:
        top_paths = sorted(paths, key=lambda entry: entry["p_fail"])[:10]
        show_legend = True
    else:
        top_paths = sorted(paths, key=lambda entry: entry["p_fail"])
        show_legend = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Assign a unique color to each path ---
    colors = cm.get_cmap('tab10', len(top_paths))

    # --- Left: Cost vs. Failure Probability ---
    for idx, entry in enumerate(top_paths):
        cost = entry["cost"]
        pfail = entry["p_fail"]
        ax1.scatter(cost, pfail, color=colors(idx), marker='o', s=100,
                    label=f'Path {idx+1}')
    ax1.set_xlabel('Total Cost')
    ax1.set_ylabel('Failure Probability')
    ax1.set_title('Cost vs. Failure Probability (Top 10)')
    ax1.grid(True)
    ncol = min(2, len(top_paths))
    if show_legend:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=ncol)

    # --- Right: Plot Obstacles if provided ---
    if obstacles is not None:
        for obs in obstacles:
            if obs["type"] == "circular":
                cx, cy = obs["center"]
                radius = obs["radius"]
                theta = np.linspace(0, 2 * np.pi, 100)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                ax2.plot(x, y, color='red', alpha=0.7)
            elif obs["type"] == "rectangular":
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                x_bounds = [x0, x1, x1, x0, x0]
                y_bounds = [y0, y0, y1, y1, y0]
                ax2.plot(x_bounds, y_bounds, color='orange', alpha=0.7)

    # --- Right: Full Paths ---
    for idx, entry in enumerate(top_paths):
        path = entry["path"]
        nodes = path.nodes if hasattr(path, "nodes") else path
        xs = [node.x for node in nodes]
        ys = [node.y for node in nodes]
        ax2.plot(xs, ys, marker='o', color=colors(idx),
                 label=f'Path {idx+1}\n(cost={entry["cost"]:.6f}, p_fail={entry["p_fail"]:.6f})')
        ax2.scatter(xs[0], ys[0], c='green', s=50, label='Start' if idx == 0 else "")
        ax2.scatter(xs[-1], ys[-1], c='blue', s=50, label='Goal' if idx == 0 else "")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Full Paths from Start to Goal (Top 10)')
    ax2.grid(True)
    if show_legend:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=ncol)

    plt.tight_layout()
    plt.show(block=True)

def redraw_tree_2d(tree, lc, edge_segments, highlighted_paths=None):
    """
    Redraw the current tree in 2D.
      - Gray for all tree edges (exploration) from parent→child
      - Green for completed / highlighted paths
    """
    # --- 1) Build gray segments from the true tree structure ---
    gray_segments = []

    for child in getattr(tree, "node_list", []):
        parent = getattr(child, "parent", None)
        if parent is None:
            continue
        seg = [(parent.x, parent.y), (child.x, child.y)]
        gray_segments.append(seg)

    # --- 2) Decide which paths to highlight in green ---
    if highlighted_paths is None:
        completed_paths = [
            p for p in getattr(tree, "paths", [])
            if hasattr(p, "is_complete") and p.is_complete
        ]
    else:
        completed_paths = []
        for p in highlighted_paths:
            if isinstance(p, dict):
                p = p.get("path", p)
            completed_paths.append(p)

    # --- 3) Build green segments from highlighted paths ---
    edge_segments.clear()  # reuse this list as "green segments only"
    for path in completed_paths:
        nodes = path.nodes if hasattr(path, "nodes") else path
        for i in range(1, len(nodes)):
            n1, n2 = nodes[i - 1], nodes[i]
            seg = [(n1.x, n1.y), (n2.x, n2.y)]
            if seg not in edge_segments:
                edge_segments.append(seg)

    # --- 4) Combine & draw ---
    all_segments = gray_segments + edge_segments
    colors = ['gray'] * len(gray_segments) + ['green'] * len(edge_segments)

    linewidths = [0.8] * len(gray_segments) + [2.5] * len(edge_segments)

    lc.set_segments(all_segments)
    lc.set_color(colors)
    lc.set_linewidths(linewidths)
    plt.pause(0.001)




def redraw_tree(tree, lc, edge_segments, highlighted_paths=None):
    """
    Redraw the current tree in 3D.
      - Gray for all tree edges (exploration) from parent→child
      - Green for selected complete paths (by default: all complete paths)
    """
    # --- 1) Build gray segments from the true tree structure ---
    gray_segments = []

    for child in getattr(tree, "node_list", []):
        parent = getattr(child, "parent", None)
        if parent is None:
            continue
        seg = [
            (parent.x, parent.y, parent.p_fail),
            (child.x,  child.y,  child.p_fail),
        ]
        gray_segments.append(seg)

    # --- 2) Decide which paths to highlight in green ---
    if highlighted_paths is None:
        completed_paths = [
            p for p in getattr(tree, "paths", [])
            if hasattr(p, "is_complete") and p.is_complete
        ]
    else:
        # highlighted_paths may be Path objects or dicts {"path": Path, ...}
        completed_paths = []
        for p in highlighted_paths:
            if isinstance(p, dict):
                p = p.get("path", p)
            completed_paths.append(p)

    # --- 3) Build green segments from highlighted paths ---
    edge_segments.clear()  # reuse this list as "green segments only"
    for path in completed_paths:
        nodes = path.nodes if hasattr(path, "nodes") else path
        for i in range(1, len(nodes)):
            n1, n2 = nodes[i - 1], nodes[i]
            seg = [
                (n1.x, n1.y, n1.p_fail),
                (n2.x, n2.y, n2.p_fail),
            ]
            if seg not in edge_segments:
                edge_segments.append(seg)

    # --- 4) Combine & draw ---
    all_segments = gray_segments + edge_segments
    colors = ['gray'] * len(gray_segments) + ['green'] * len(edge_segments)

    linewidths = [0.8] * len(gray_segments) + [2.5] * len(edge_segments)

    lc.set_segments(all_segments)
    lc.set_color(colors)
    lc.set_linewidths(linewidths)
    plt.pause(0.001)





def interactive_cluster_plot(paths, cluster_func, obstacles=None):
    """
    Interactive cluster visualizer.

    - paths: list of entries {"path": Path, "cost": float, "p_fail": float}
    - cluster_func: function(paths, spatial_tol, cost_tol, p_fail_tol, criteria) -> clusters
    - obstacles: optional obstacles list to draw on map
    """
    if not paths:
        print("No paths to visualize.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.25)

    # slider axes (only cost and p_fail)
    axcolor = 'lightgoldenrodyellow'
    ax_cost = plt.axes([0.08, 0.10, 0.4, 0.03], facecolor=axcolor)
    ax_pfail = plt.axes([0.08, 0.05, 0.4, 0.03], facecolor=axcolor)

    # initial values
    cost0 = 1.0
    pfail0 = 0.05

    s_cost = Slider(ax_cost, 'cost_tol', 0.0, 200.0, valinit=cost0)
    s_pfail = Slider(ax_pfail, 'p_fail_tol', 0.0, 1.0, valinit=pfail0)

    def compute_and_draw(_=None):
        clusters = cluster_func(paths, cost_tol=s_cost.val, p_fail_tol=s_pfail.val)

        ax1.clear(); ax2.clear()

        # Left: cost vs p_fail, plot representative and annotate count
        if clusters:
            cmap = cm.get_cmap('tab10', max(1, len(clusters)))
            for idx, cl in enumerate(clusters):
                rep = cl['representative']
                ax1.scatter(rep['cost'], rep['p_fail'], color=cmap(idx), s=120)
                ax1.text(rep['cost'], rep['p_fail'], f"  C{idx+1} ({len(cl['members'])})", fontsize=9)
        else:
            # no clusters -> scatter all
            costs = [e['cost'] for e in paths]
            pfails = [e['p_fail'] for e in paths]
            ax1.scatter(costs, pfails, color='gray')

        ax1.set_xlabel('Total Cost')
        ax1.set_ylabel('Failure Probability')
        ax1.set_title('Cost vs Failure Probability (cluster reps)')
        ax1.grid(True)

        # Right: spatial plot of paths colored by cluster
        if clusters:
            cmap = cm.get_cmap('tab10', max(1, len(clusters)))
            for idx, cl in enumerate(clusters):
                color = cmap(idx)
                for entry in cl['members']:
                    nodes = entry['path'].nodes
                    xs = [n.x for n in nodes]
                    ys = [n.y for n in nodes]
                    ax2.plot(xs, ys, color=color, alpha=0.6)
                # centroid marker
                rep = cl['representative']
                nodes = rep['path'].nodes
                cx = sum(n.x for n in nodes) / len(nodes)
                cy = sum(n.y for n in nodes) / len(nodes)
                ax2.scatter(cx, cy, color=color, marker='x', s=80)
        else:
            for entry in paths:
                nodes = entry['path'].nodes
                xs = [n.x for n in nodes]
                ys = [n.y for n in nodes]
                ax2.plot(xs, ys, color='gray', alpha=0.6)

        # Draw obstacles if provided
        if obstacles is not None:
            for obs in obstacles:
                if obs['type'] == 'circular':
                    cx, cy = obs['center']
                    radius = obs['radius']
                    theta = np.linspace(0, 2*np.pi, 100)
                    x = cx + radius * np.cos(theta)
                    y = cy + radius * np.sin(theta)
                    ax2.plot(x, y, color='red', alpha=0.6)
                elif obs['type'] == 'rectangular':
                    x0, x1 = obs['x_range']
                    y0, y1 = obs['y_range']
                    x_bounds = [x0, x1, x1, x0, x0]
                    y_bounds = [y0, y0, y1, y1, y0]
                    ax2.plot(x_bounds, y_bounds, color='orange', alpha=0.6)

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Spatial Paths (colored by cluster)')
        ax2.grid(True)

        fig.canvas.draw_idle()

    # hook sliders
    s_cost.on_changed(compute_and_draw)
    s_pfail.on_changed(compute_and_draw)

    # initial draw
    compute_and_draw()
    plt.show(block=True)


def interactive_spectral_cluster_plot(paths, spectral_cluster_func, obstacles=None):
    """
    Interactive spectral-clustering visualizer.

    - paths: list of {"path": Path, "cost": float, "p_fail": float}
    - spectral_cluster_func: callable(paths, n_clusters=..., m_points=..., w_xy=..., w_cost=..., w_pfail=..., neighbor_k=...) -> (clusters, labels, debug)
    - obstacles: optional
    """
    if not paths:
        print("No paths to visualize.")
        return

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider
    import matplotlib.cm as cm

    N = len(paths)
    kmax = min(10, N)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    plt.subplots_adjust(left=0.08, right=0.98, top=0.95, bottom=0.25)

    axcolor = 'lightgoldenrodyellow'
    ax_k      = plt.axes([0.08, 0.10, 0.40, 0.03], facecolor=axcolor)
    ax_wxy    = plt.axes([0.08, 0.06, 0.40, 0.03], facecolor=axcolor)
    ax_wcost  = plt.axes([0.08, 0.02, 0.40, 0.03], facecolor=axcolor)
    ax_wpfail = plt.axes([0.58, 0.06, 0.34, 0.03], facecolor=axcolor)
    ax_m      = plt.axes([0.58, 0.10, 0.34, 0.03], facecolor=axcolor)

    s_k      = Slider(ax_k,     'k (0=auto)', 0, kmax, valinit=0, valstep=1)
    s_wxy    = Slider(ax_wxy,   'w_xy',       0.0, 3.0, valinit=1.0)
    s_wcost  = Slider(ax_wcost, 'w_cost',     0.0, 3.0, valinit=0.25)
    s_wpfail = Slider(ax_wpfail,'w_p_fail',   0.0, 3.0, valinit=0.75)
    s_m      = Slider(ax_m,     'samples m',  16, 128, valinit=64, valstep=1)

    def compute_and_draw(_=None):
        ax1.clear(); ax2.clear()

        k = int(s_k.val)
        k_arg = "auto" if k == 0 else k
        clusters, labels, debug = spectral_cluster_func(
            paths,
            n_clusters=k_arg,
            m_points=int(s_m.val),
            w_xy=s_wxy.val,
            w_cost=s_wcost.val,
            w_pfail=s_wpfail.val,
        )

        # Left: scatter representatives (cost vs p_fail), annotate with (size)
        if clusters:
            cmap = cm.get_cmap('tab10', max(1, len(clusters)))
            for idx, cl in enumerate(clusters):
                rep = cl['representative']
                ax1.scatter(rep['cost'], rep['p_fail'], color=cmap(idx), s=140)
                ax1.text(rep['cost'], rep['p_fail'], f"  C{idx+1} ({len(cl['members'])})", fontsize=9)
        else:
            costs = [e['cost'] for e in paths]
            pfails = [e['p_fail'] for e in paths]
            ax1.scatter(costs, pfails, color='gray')

        ax1.set_xlabel('Total Cost'); ax1.set_ylabel('Failure Probability')
        ax1.set_title('Spectral clusters (representatives)')
        ax1.grid(True)

        # Right: spatial plot colored by cluster
        if clusters:
            cmap = cm.get_cmap('tab10', max(1, len(clusters)))
            for idx, cl in enumerate(clusters):
                color = cmap(idx)
                for entry in cl['members']:
                    nodes = entry['path'].nodes
                    xs = [n.x for n in nodes]; ys = [n.y for n in nodes]
                    ax2.plot(xs, ys, color=color, alpha=0.7)
                # centroid mark for the representative
                rep_nodes = cl['representative']['path'].nodes
                cx = sum(n.x for n in rep_nodes) / len(rep_nodes)
                cy = sum(n.y for n in rep_nodes) / len(rep_nodes)
                ax2.scatter(cx, cy, color=color, marker='x', s=80)
        else:
            for entry in paths:
                nodes = entry['path'].nodes
                xs = [n.x for n in nodes]; ys = [n.y for n in nodes]
                ax2.plot(xs, ys, color='gray', alpha=0.6)

        # obstacles (optional)
        if obstacles is not None:
            for obs in obstacles:
                if obs.get('type') == 'circular':
                    cx, cy = obs['center']; r = obs['radius']
                    th = np.linspace(0, 2*np.pi, 100)
                    x = cx + r*np.cos(th); y = cy + r*np.sin(th)
                    ax2.plot(x, y, color='red', alpha=0.6)
                elif obs.get('type') == 'rectangular':
                    x0, x1 = obs['x_range']; y0, y1 = obs['y_range']
                    xb = [x0, x1, x1, x0, x0]; yb = [y0, y0, y1, y1, y0]
                    ax2.plot(xb, yb, color='orange', alpha=0.6)

        ax2.set_xlabel('X'); ax2.set_ylabel('Y')
        ax2.set_title('Paths colored by spectral cluster')
        ax2.grid(True)

        fig.canvas.draw_idle()

    # hook sliders
    for s in (s_k, s_wxy, s_wcost, s_wpfail, s_m):
        s.on_changed(compute_and_draw)

    compute_and_draw()
    plt.show(block=True)

def plot_final_tree_2d(tree,
                       filtered_paths=None,
                       grid=None,
                       obstacles=None,
                       max_highlight_paths=10,
                       title="PORRT* Final Tree (2D)"):
    """
    Static 2D visualization of the final RRT* tree.

    - Draws ALL tree edges in a colormap based on p_fail.
    - Overlays filtered (Pareto) paths in thick, bright lines.
    - Optionally plots obstacles and start/goal.

    Parameters
    ----------
    tree : Tree
        Your RRT* Tree instance (with tree.paths and Node.x/y/p_fail).
    filtered_paths : list[dict] | None
        Entries like {"path": Path, "cost": float, "p_fail": float}.
    grid : Grid | None
        Used for x/y limits and obstacles. If None, tries tree.grid.
    obstacles : list | None
        Optional explicit obstacles list (overrides grid.obstacles).
    max_highlight_paths : int
        Max number of filtered paths to overlay.
    title : str
        Plot title.
    """
    # Try to infer grid & obstacles from the tree if not provided
    if grid is None:
        grid = getattr(tree, "grid", None)

    if obstacles is None and grid is not None:
        obstacles = getattr(grid, "obstacles", None)

    # --- Collect all unique edges from the tree ---

    segments = []   # list of [(x1,y1), (x2,y2)]
    pfails   = []   # one scalar per edge (we'll use child.p_fail)
    seen     = set()

    segments = []   # list of [(x1,y1), (x2,y2)]
    pfails   = []   # one scalar per edge (we'll use child.p_fail)
    seen     = set()

    for child in tree.node_list:
        parent = getattr(child, "parent", None)
        if parent is None:
            continue

        key = (round(parent.x, 4), round(parent.y, 4),
               round(child.x, 4),  round(child.y, 4))
        if key in seen:
            continue
        seen.add(key)

        segments.append([(parent.x, parent.y), (child.x, child.y)])
        pfails.append(float(getattr(child, "p_fail", 0.0)))

    # --- Set up figure & axes ---

    fig, ax = plt.subplots(figsize=(7, 7))

    if grid is not None:
        ax.set_xlim(0, grid.width)
        ax.set_ylim(0, grid.height)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    # --- Obstacles in 2D (if any) ---

    if obstacles is not None:
        for obs in obstacles:
            t = obs.get("type")
            if t == "circular":
                cx, cy = obs["center"]
                radius = obs["radius"]
                theta = np.linspace(0, 2 * np.pi, 100)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                ax.plot(x, y, linestyle="--", alpha=0.6, color="red")
            elif t == "rectangular":
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                xs = [x0, x1, x1, x0, x0]
                ys = [y0, y0, y1, y1, y0]
                ax.plot(xs, ys, linestyle="--", alpha=0.6, color="orange")
            elif t == "rect":
                # in case you ever use bottom_left / top_right format
                x1, y1 = obs["bottom_left"]
                x2, y2 = obs["top_right"]
                xs = [x1, x2, x2, x1, x1]
                ys = [y1, y1, y2, y2, y1]
                ax.plot(xs, ys, linestyle="--", alpha=0.6, color="orange")

    # --- Draw tree edges colored by p_fail ---

    if segments:
        if any(np.isfinite(p) for p in pfails):
            vmin = min(pfails)
            vmax = max(pfails)
            # avoid vmin == vmax which breaks Normalize
            if abs(vmax - vmin) < 1e-9:
                vmin, vmax = 0.0, max(1e-6, vmax)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            lc = LineCollection(segments, cmap=cm.viridis, norm=norm, linewidths=0.8, alpha=0.7)
            lc.set_array(np.array(pfails))
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax)
            cbar.set_label("p_fail")
        else:
            # fallback: plain gray tree
            lc = LineCollection(segments, colors="gray", linewidths=0.8, alpha=0.7)
            ax.add_collection(lc)

    # --- Overlay filtered Pareto-optimal paths ---

    if filtered_paths:
        # Sort by cost (or any metric) and take a few
        sorted_paths = sorted(filtered_paths, key=lambda e: e["cost"])
        for idx, entry in enumerate(sorted_paths[:max_highlight_paths]):
            path = entry["path"]
            nodes = path.nodes if hasattr(path, "nodes") else path
            xs = [n.x for n in nodes]
            ys = [n.y for n in nodes]

            # Use a colormap but thicker lines
            color = cm.tab10(idx % 10)
            ax.plot(xs, ys,
                    color=color,
                    linewidth=2.5,
                    alpha=0.95,
                    label=f"Pareto {idx+1}: cost={entry['cost']:.2f}, p_fail={entry['p_fail']:.3f}")

    # --- Mark start & goal if we can infer them ---

    start_node = None
    goal_node  = None
    # Try to find marked start/goal in the tree nodes
    for n in getattr(tree, "node_list", []):
        if getattr(n, "is_start", False):
            start_node = n
        if getattr(n, "is_goal", False):
            goal_node = n

    if start_node is not None:
        ax.scatter(start_node.x, start_node.y, c="green", s=80, marker="o", label="Start")

    if goal_node is not None:
        ax.scatter(goal_node.x, goal_node.y, c="red", s=80, marker="*", label="Goal")

    # If we added any labels for filtered paths or start/goal, show legend
    if (filtered_paths and len(filtered_paths) > 0) or start_node or goal_node:
        ax.legend(loc="best")

    plt.tight_layout()
    plt.show(block=True)

