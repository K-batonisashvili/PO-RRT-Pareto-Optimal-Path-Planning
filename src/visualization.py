import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm
from matplotlib.widgets import Slider, RadioButtons
from matplotlib.collections import LineCollection 
import matplotlib.colors as mcolors

# Utility: derive a consistent color list for a sequence of paths
def _get_path_colors(paths):
    """Return a list of RGBA colors for the provided paths using tab10.

    The ordering matches the input `paths` order, which allows consistent
    coloring across different plots (e.g., Pareto scatter and spatial paths).
    """
    n = len(paths) if paths is not None else 0
    cmap = cm.get_cmap('tab10', max(1, n))
    return [cmap(i) for i in range(n)]

# Map rectangular obstacle "probability" in [0.01, 1.0] to a yellow→red color.
# Values near 0.01 map to yellow (darkened for visibility), values near 1.0 map to red.
def _occupancy_color(prob):
    """Return an RGBA color for occupancy probability `prob`.

    To improve visibility, the low end of the colormap (near 0.01) is shifted
    away from very pale yellow toward a darker yellow by applying a minimum
    `t` offset before sampling the 'YlOrRd' colormap.
    """
    cmap = cm.get_cmap('YlOrRd')
    MIN_T = 0.40  # lower bound for colormap sampling to avoid very pale yellows
    try:
        p = float(prob)
    except Exception:
        # Fallback to a visible dark-yellow
        return cmap(MIN_T)
    # clip to supported range
    p_clipped = min(max(p, 0.01), 1.0)
    t = (p_clipped - 0.01) / (1.0 - 0.01)
    # Shift sampling range upward so the low end is darker/more visible
    t = MIN_T + (1.0 - MIN_T) * t
    return cmap(t)

# Helper to reorder legend
def _reorder_legend(ax):
    handles, labels = ax.get_legend_handles_labels()
    # Separate Start/Goal from others
    sg_pairs = []
    other_pairs = []
    
    for h, l in zip(handles, labels):
        if l.startswith("Start") or l.startswith("Goal"):
            sg_pairs.append((h, l))
        else:
            other_pairs.append((h, l))
            
    # Sort sg_pairs so Start is usually before Goal if desired, or just keep order found
    # Let's ensure Start comes before Goal if both exist
    sg_pairs.sort(key=lambda x: x[1], reverse=True) # Start... vs Goal... -> Start comes last alphabetically? No. S > G.
    # Actually, let's just prioritize "Start" then "Goal"
    start_pair = [p for p in sg_pairs if p[1].startswith("Start")]
    goal_pair = [p for p in sg_pairs if p[1].startswith("Goal")]
    
    # Combine: Start, Goal, Others
    final_handles = []
    final_labels = []
    
    for h, l in start_pair + goal_pair + other_pairs:
        final_handles.append(h)
        final_labels.append(l)
        
    if final_handles:
        ax.legend(handles=final_handles, labels=final_labels, bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)

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

    # Plot start and goal (include coordinates in labels)
    start_coord = (round(start[0], 2), round(start[1], 2))
    goal_coord = (round(goal[0], 2), round(goal[1], 2))
    ax.scatter(start[0], start[1], c='red', s=60, label=f'Start {start_coord}')
    ax.scatter(goal[0], goal[1], c='red', s=80, marker='*', label=f'Goal {goal_coord}')

    # Ensure legend keeps Start/Goal at the top
    _reorder_legend(ax)

    # Plot obstacles (2D)
    if obstacles is not None:
        for obs in obstacles:
            if obs.get("type") == "circular":
                cx, cy = obs["center"]
                radius = obs["radius"]
                theta = np.linspace(0, 2 * np.pi, 100)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                prob = obs.get("probability", 1.0)
                color = _occupancy_color(prob)
                ax.plot(x, y, linestyle='--', alpha=0.5, color=color)
            elif obs.get("type") == "rectangular":
                # x_range / y_range style (matches other parts of the code)
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                xs = [x0, x1, x1, x0, x0]
                ys = [y0, y0, y1, y1, y0]
                prob = obs.get("probability", None)
                color = _occupancy_color(prob) if prob is not None else 'orange'
                ax.plot(xs, ys, linestyle='--', alpha=0.5, color=color)

    # Empty line collection for tree edges
    lc = LineCollection([], linewidths=1.0)
    ax.add_collection(lc)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("PORRT*")
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
    ax.set_title("PORRT* with P_fail Dimension")

    # Plot start and goal (include coordinates in labels)
    start_coord = (round(start[0], 2), round(start[1], 2))
    goal_coord = (round(goal[0], 2), round(goal[1], 2))
    ax.scatter(start[0], start[1], 0, c='red', s=60, label=f'Start {start_coord}')
    ax.scatter(goal[0], goal[1], 0, c='red', s=80, label=f'Goal {goal_coord}')

    # Ensure legend keeps Start/Goal at the top
    _reorder_legend(ax)

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
                prob = obs.get("probability", 1.0)
                color = _occupancy_color(prob)
                ax.plot(x, y, z, color=color, alpha=0.7)
            elif obs.get("type") == "rectangular":
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                x_bounds = [x0, x1, x1, x0, x0]
                y_bounds = [y0, y0, y1, y1, y0]
                z = np.zeros_like(x_bounds)
                prob = obs.get("probability", None)
                color = _occupancy_color(prob) if prob is not None else 'orange'
                ax.plot_trisurf(x_bounds, y_bounds, z, color=color, alpha=0.3)

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
    Ensures colors are CONSISTENT with the original 'paths' list order.
    """
    # 1. Generate consistent colors based on the ORIGINAL list order
    all_colors = _get_path_colors(paths)
    
    # Map specific path objects to their assigned color using object ID
    # This ensures that even after sorting, we use the original color.
    path_to_color = {id(entry["path"]): col for entry, col in zip(paths, all_colors)}
    
    # 2. Sort by p_fail (increasing) for the summary logic
    if len(paths) <= 15:
        top_paths = sorted(paths, key=lambda entry: entry["p_fail"])[:10]
        show_legend = True
    else:
        top_paths = sorted(paths, key=lambda entry: entry["p_fail"])
        show_legend = False

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Left: Cost vs. Failure Probability ---
    for idx, entry in enumerate(top_paths):
        cost = entry["cost"]
        pfail = entry["p_fail"]
        
        # Retrieve the consistent color
        color = path_to_color[id(entry["path"])]
        
        # We label it based on its original index if possible, or just "Path idx" from the sorted list
        # If you want the label to match the original index, you'd need to find it.
        # For now, we label based on the sorted rank (1st safest, 2nd safest...) 
        # but keep the color consistent with other plots.
        ax1.scatter(cost, pfail, color=color, marker='o', s=100, label=f'Path {idx+1}')
        
    ax1.set_xlabel('Euclidean Distance')
    ax1.set_ylabel('Failure Probability')
    ax1.set_title('Euclidean Distance vs. Failure Probability')
    ax1.grid(True)
    
    if show_legend:
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., ncol=1)

    # --- Right: Plot Obstacles if provided ---
    if obstacles is not None:
        for obs in obstacles:
            if obs["type"] == "circular":
                cx, cy = obs["center"]
                radius = obs["radius"]
                theta = np.linspace(0, 2 * np.pi, 100)
                x = cx + radius * np.cos(theta)
                y = cy + radius * np.sin(theta)
                prob = obs.get("probability", 1.0)
                color = _occupancy_color(prob)
                ax2.plot(x, y, color=color, alpha=0.7)
            elif obs["type"] == "rectangular":
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                x_bounds = [x0, x1, x1, x0, x0]
                y_bounds = [y0, y0, y1, y1, y0]
                prob = obs.get("probability", None)
                color = _occupancy_color(prob) if prob is not None else 'orange'
                ax2.plot(x_bounds, y_bounds, color=color, alpha=0.7)

    # --- Right: Full Paths ---
    for idx, entry in enumerate(top_paths):
        path = entry["path"]
        nodes = path.nodes if hasattr(path, "nodes") else path
        xs = [node.x for node in nodes]
        ys = [node.y for node in nodes]
        
        # Retrieve the consistent color
        color = path_to_color[id(entry["path"])]
        
        ax2.plot(xs, ys, marker='o', color=color,
                 label=f'Path {idx+1}\n(Dist.={entry["cost"]:.6f}, P_fail={entry["p_fail"]:.6f})')
        
        if idx == 0:
            start_coord = (round(xs[0], 2), round(ys[0], 2))
            goal_coord = (round(xs[-1], 2), round(ys[-1], 2))
            ax2.scatter(xs[0], ys[0], c='green', s=50, label=f'Start {start_coord}')
            ax2.scatter(xs[-1], ys[-1], c='blue', s=50, label=f'Goal {goal_coord}')

    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Full Paths from Start to Goal')
    ax2.grid(True)
    
    if show_legend:
        _reorder_legend(ax2)

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

    # --- 3) Build highlighted segments from highlighted paths ---
    edge_segments.clear()  # reuse this list as "highlighted segments only"
    highlight_colors = []

    # construct a color palette matching the highlighted_paths order
    path_colors = _get_path_colors(completed_paths) if completed_paths else []

    for p_idx, path in enumerate(completed_paths):
        nodes = path.nodes if hasattr(path, "nodes") else path
        col = path_colors[p_idx] if p_idx < len(path_colors) else 'green'
        for i in range(1, len(nodes)):
            n1, n2 = nodes[i - 1], nodes[i]
            seg = [(n1.x, n1.y), (n2.x, n2.y)]
            if seg not in edge_segments:
                edge_segments.append(seg)
                highlight_colors.append(col)

    # --- 4) Combine & draw ---
    all_segments = gray_segments + edge_segments
    colors = ['gray'] * len(gray_segments) + highlight_colors

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

    def _chain_from_node_via_parents(node):
        """Return [start ... node] by following .parent pointers."""
        chain = []
        cur = node
        while cur is not None:
            chain.append(cur)
            cur = getattr(cur, "parent", None)
        chain.reverse()
        return chain



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

        # --- 2) Decide which goal nodes to highlight in green ---
    if highlighted_paths is None:
        goal_nodes = [n for n in getattr(tree, "node_list", [])
                      if getattr(n, "is_goal", False)]
    else:
        goal_nodes = []
        for p in highlighted_paths:
            if isinstance(p, dict):
                p = p.get("path", p)
            nodes = p.nodes if hasattr(p, "nodes") else p
            if nodes:
                goal_nodes.append(nodes[-1])  # last node = goal

    # --- 3) Build green segments from parent pointers (rewire-safe) ---
    edge_segments.clear()  # reuse this list as "highlighted segments only"

    # color palette for goal/highlighted paths
    path_colors = _get_path_colors(goal_nodes) if goal_nodes else []
    highlight_colors = []

    for g_idx, g in enumerate(goal_nodes):
        chain = _chain_from_node_via_parents(g)
        col = path_colors[g_idx] if g_idx < len(path_colors) else 'green'
        for i in range(1, len(chain)):
            n1, n2 = chain[i - 1], chain[i]
            seg = [
                (n1.x, n1.y, getattr(n1, "p_fail", 0.0)),
                (n2.x, n2.y, getattr(n2, "p_fail", 0.0)),
            ]
            if seg not in edge_segments:
                edge_segments.append(seg)
                highlight_colors.append(col)

    # --- 4) Combine & draw ---
    all_segments = gray_segments + edge_segments
    colors = ['gray'] * len(gray_segments) + highlight_colors

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

        ax1.set_xlabel('Euclidean Distance')
        ax1.set_ylabel('Failure Probability')
        ax1.set_title('Euclidean Distance vs Failure Probability')
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
                    prob = obs.get('probability', 1.0)
                    color = _occupancy_color(prob)
                    ax2.plot(x, y, color=color, alpha=0.6)
                elif obs['type'] == 'rectangular':
                    x0, x1 = obs['x_range']
                    y0, y1 = obs['y_range']
                    x_bounds = [x0, x1, x1, x0, x0]
                    y_bounds = [y0, y0, y1, y1, y0]
                    prob = obs.get('probability', None)
                    color = _occupancy_color(prob) if prob is not None else 'orange'
                    ax2.plot(x_bounds, y_bounds, color=color, alpha=0.6)

        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        ax2.set_title('Spatial Paths')
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

        ax1.set_xlabel('Euclidean Distance'); ax1.set_ylabel('Failure Probability')
        ax1.set_title('Spectral clusters')
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
                    prob = obs.get('probability', 1.0)
                    color = _occupancy_color(prob)
                    ax2.plot(x, y, color=color, alpha=0.6)
                elif obs.get('type') == 'rectangular':
                    x0, x1 = obs['x_range']; y0, y1 = obs['y_range']
                    xb = [x0, x1, x1, x0, x0]; yb = [y0, y0, y1, y1, y0]
                    prob = obs.get('probability', None)
                    color = _occupancy_color(prob) if prob is not None else 'orange'
                    ax2.plot(xb, yb, color=color, alpha=0.6)

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
                       title="PORRT* Tree"):
    """
    Static 2D visualization of the final RRT* tree.
    """
    # ... [Keep tree/grid inference and edge collection logic unchanged] ...
    # Try to infer grid & obstacles from the tree if not provided
    if grid is None:
        grid = getattr(tree, "grid", None)

    if obstacles is None and grid is not None:
        obstacles = getattr(grid, "obstacles", None)

    # --- Collect all unique edges from the tree ---
    segments = []
    pfails   = []
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
                prob = obs.get("probability", 1.0)
                color = _occupancy_color(prob)
                ax.plot(x, y, linestyle="--", alpha=0.6, color=color)
            elif t == "rectangular":
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                xs = [x0, x1, x1, x0, x0]
                ys = [y0, y0, y1, y1, y0]
                prob = obs.get("probability", None)
                color = _occupancy_color(prob) if prob is not None else 'orange'
                ax.plot(xs, ys, linestyle="--", alpha=0.6, color=color)
            elif t == "rect":
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
            if abs(vmax - vmin) < 1e-9:
                vmin, vmax = 0.0, max(1e-6, vmax)
            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            lc = LineCollection(segments, cmap=cm.viridis, norm=norm, linewidths=0.8, alpha=0.7)
            lc.set_array(np.array(pfails))
            ax.add_collection(lc)
            cbar = fig.colorbar(lc, ax=ax)
            cbar.set_label("p_fail")
        else:
            lc = LineCollection(segments, colors="gray", linewidths=0.8, alpha=0.7)
            ax.add_collection(lc)

    # --- Overlay filtered Pareto-optimal paths ---
    if filtered_paths:
        sorted_paths = sorted(filtered_paths, key=lambda e: e["cost"])
        for idx, entry in enumerate(sorted_paths[:max_highlight_paths]):
            path = entry["path"]
            nodes = path.nodes if hasattr(path, "nodes") else path
            xs = [n.x for n in nodes]
            ys = [n.y for n in nodes]
            color = cm.tab10(idx % 10)
            ax.plot(xs, ys,
                    color=color,
                    linewidth=2.5,
                    alpha=0.95,
                    label=f"Pareto {idx+1}: cost={entry['cost']:.2f}, p_fail={entry['p_fail']:.3f}")

    # --- Mark start & goal if we can infer them ---
    start_node = None
    goal_node  = None
    for n in getattr(tree, "node_list", []):
        if getattr(n, "is_start", False):
            start_node = n
        if getattr(n, "is_goal", False):
            goal_node = n

    if start_node is not None:
        start_coord = (round(start_node.x, 2), round(start_node.y, 2))
        ax.scatter(start_node.x, start_node.y, c="green", s=80, marker="o", label=f"Start {start_coord}")

    if goal_node is not None:
        goal_coord = (round(goal_node.x, 2), round(goal_node.y, 2))
        ax.scatter(goal_node.x, goal_node.y, c="red", s=80, marker="*", label=f"Goal {goal_coord}")

    if (filtered_paths and len(filtered_paths) > 0) or start_node or goal_node:
        _reorder_legend(ax)

    plt.tight_layout()
    plt.show(block=True)

def plot_tree_2d_basic(
    tree=None,
    nodes=None,
    grid=None,
    obstacles=None,
    title="RRT* Tree",
    edge_color="0.6",
    edge_lw=0.6,
    edge_alpha=0.7,
):
    """
    Basic static 2D visualization of an RRT* tree:
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    # ... [Keep node/grid/segment resolution unchanged] ...
    if nodes is None and tree is not None:
        nodes = getattr(tree, "node_list", None)
    if nodes is None:
        raise ValueError("plot_tree_2d_basic: provide either tree=... or nodes=[...]")

    if grid is None and tree is not None:
        grid = getattr(tree, "grid", None)
    if obstacles is None and grid is not None:
        obstacles = getattr(grid, "obstacles", None)

    segments = []
    seen = set()
    start_node = None
    goal_node = None

    for child in nodes:
        if getattr(child, "is_start", False):
            start_node = child
        if getattr(child, "is_goal", False):
            goal_node = child
        parent = getattr(child, "parent", None)
        if parent is None:
            continue
        key = (round(parent.x, 4), round(parent.y, 4),
               round(child.x, 4),  round(child.y, 4))
        if key in seen:
            continue
        seen.add(key)
        segments.append([(parent.x, parent.y), (child.x, child.y)])

    fig, ax = plt.subplots(figsize=(7, 7))

    if grid is not None:
        ax.set_xlim(0, grid.width)
        ax.set_ylim(0, grid.height)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True)

    # Obstacles
    if obstacles is not None:
        for obs in obstacles:
            if obs.get("type") == "circular":
                cx, cy = obs["center"]
                r = obs["radius"]
                theta = np.linspace(0, 2*np.pi, 200)
                prob = obs.get("probability", 1.0)
                color = _occupancy_color(prob)
                ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta),
                        linestyle="--", alpha=0.6, color=color)
            elif obs.get("type") == "rectangular":
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                xs = [x0, x1, x1, x0, x0]
                ys = [y0, y0, y1, y1, y0]
                prob = obs.get("probability", None)
                color = _occupancy_color(prob) if prob is not None else 'orange'
                ax.plot(xs, ys, linestyle="--", alpha=0.6, color=color)

    # Tree edges
    if segments:
        lc = LineCollection(segments, colors=edge_color, linewidths=edge_lw, alpha=edge_alpha)
        ax.add_collection(lc)

    # Start/goal markers
    if start_node is not None:
        start_coord = (round(start_node.x, 2), round(start_node.y, 2))
        ax.scatter(start_node.x, start_node.y, c="green", s=80, marker="o", label=f"Start {start_coord}")
    if goal_node is not None:
        goal_coord = (round(goal_node.x, 2), round(goal_node.y, 2))
        ax.scatter(goal_node.x, goal_node.y, c="red", s=80, marker="*", label=f"Goal {goal_coord}")

    if start_node is not None or goal_node is not None:
        _reorder_legend(ax)

    plt.tight_layout()
    plt.show(block=True)