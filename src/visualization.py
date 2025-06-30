import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.cm as cm

def init_progress_plot_3d(start, goal, x_lim, y_lim, obstacles, z_lim=(0.0, 1.0)):
    """
    Initialize a 3D progress plot for RRT*.

    Parameters:
    - start, goal: (x, y, θ) tuples
    - x_lim, y_lim, z_lim: plot bounds
    - obstacles: list of obstacle dictionaries

    Returns:
    - fig, ax: Matplotlib Figure and 3D Axes
    - lc: Line3DCollection for tree edges
    - edge_segments: mutable list for edge segments
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

    # Plot start and goal
    ax.scatter(start[0], start[1], 0.0, c='red', s=60, label='Start')
    ax.scatter(goal[0], goal[1], 0.0, c='red', s=80, label='Goal')

    # Plot obstacles
    for obs in obstacles:
        if obs["type"] == "circular":
            cx, cy = obs["center"]
            radius = obs["radius"]
            theta = np.linspace(0, 2 * np.pi, 100)
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            z = np.zeros_like(x)
            ax.plot(x, y, z, color='red', alpha=0.7)
        elif obs["type"] == "rectangular":
            x0, x1 = obs["x_range"]
            y0, y1 = obs["y_range"]
            x_bounds = [x0, x1, x1, x0, x0]
            y_bounds = [y0, y0, y1, y1, y0]
            z = np.zeros_like(x_bounds)
            ax.plot_trisurf(x_bounds, y_bounds, z, color='orange', alpha=0.3)

    ax.legend()

    # Tree edge visualization
    edge_segments = []
    lc = Line3DCollection(edge_segments, linewidths=1.5, alpha=0.7, color='C0')
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
    # Sort by cost and select top 10
    top_paths = sorted(paths, key=lambda entry: entry["cost"])[:10] if len(paths) > 10 else paths

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
    ax1.legend()

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
                 label=f'Path {idx+1}\n(cost={entry["cost"]:.2f}, p_fail={entry["p_fail"]:.3f})')
        ax2.scatter(xs[0], ys[0], c='green', s=50, label='Start' if idx == 0 else "")
        ax2.scatter(xs[-1], ys[-1], c='blue', s=50, label='Goal' if idx == 0 else "")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Full Paths from Start to Goal (Top 10)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()
    plt.show(block=True)

def redraw_tree(tree, lc, edge_segments):
    """
    Redraw the current tree. Black for exploration paths, green for completed paths.
    This version uses path-based tree structure.
    """
    new_segments = []

    # Draw all tree edges in black (parent → child) from all paths
    for path in tree.paths:
        nodes = path.nodes
        for i in range(1, len(nodes)):
            parent = nodes[i - 1]
            child = nodes[i]
            segment = [(parent.x, parent.y, parent.p_fail), (child.x, child.y, child.p_fail)]
            new_segments.append(segment)

    # Draw completed (start → goal) paths in green
    for path in tree.paths:
        if hasattr(path, "is_complete") and path.is_complete:
            for i in range(1, len(path.nodes)):
                n1 = path.nodes[i - 1]
                n2 = path.nodes[i]
                segment = [(n1.x, n1.y, n1.p_fail), (n2.x, n2.y, n2.p_fail)]

                # Prevent duplicate green edges
                if (segment, 'green') not in edge_segments:
                    edge_segments.append((segment, 'green'))

    # Combine all segments for drawing
    all_segments = new_segments + [seg for seg, color in edge_segments]
    colors = ['black'] * len(new_segments) + [color for seg, color in edge_segments]

    lc.set_segments(all_segments)
    lc.set_color(colors)
    plt.pause(0.001)

