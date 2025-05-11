import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection

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
    ax.scatter(start[0], start[1], 0.0, c='green', s=50, label='Start')
    ax.scatter(goal[0], goal[1], 0.0, c='blue', s=50, label='Goal')

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
        else:
            print(f"No edge between {parent_node} and {child_node} to remove.")
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
    costs  = [entry[1] for entry in paths]
    pfails = [entry[2] for entry in paths]
    
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

def redraw_tree(tree, lc, edge_segments):
    """
    Redraw the entire tree to ensure the plot is consistent with the tree structure.
    """
    edge_segments.clear()
    for path in tree.paths:
        for node in path.nodes:
            if node.parent:
                edge_segments.append([(node.parent.x, node.parent.y, node.parent.p_fail),
                                      (node.x, node.y, node.p_fail)])
    lc.set_segments(edge_segments)
    plt.pause(0.001)