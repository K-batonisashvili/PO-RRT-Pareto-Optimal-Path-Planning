import numpy as np
import pytest
from PO_RRT_Star import Node, Tree, Grid, GRID_RESOLUTION
from visualization import init_progress_plot_3d, update_progress_plot_3d

def test_rewire_prefers_lower_risk_path_with_plot():
    # Create a 10x10 grid with an unknown area from (3,3) to (7,7) with high probability
    obstacles = [
        {"type": "rectangular", "x_range": (3, 7), "y_range": (3, 7), "probability": 0.05}
    ]
    grid = Grid(100, 100, obstacles)

    # Create start and goal nodes
    start = Node(2, 2, 0)
    goal = Node(8, 2, 0)

    # Create a tree and add the start node
    tree = Tree(grid)
    tree.add_node(start)

    # Set up plotting
    fig, ax, lc, edge_segments = init_progress_plot_3d(
        (start.x, start.y, 0), (goal.x, goal.y, 0),
        x_lim=(0, 10), y_lim=(0, 10), obstacles=obstacles
    )

    # Manually create optimal path nodes outside the unknown area
    n1 = Node(4, 2, 0)
    n2 = Node(6, 2, 0)
    for n, parent in zip([n1, n2, goal], [start, n1, n2]):
        n.parent = parent
        parent.children.append(n)
        n.cost = parent.cost + np.linalg.norm([n.x - parent.x, n.y - parent.y])
        n.log_survival = parent.log_survival  # No risk outside unknown area
        n.p_fail = 1 - np.exp(n.log_survival)
        tree.add_node(n)
        update_progress_plot_3d(lc, edge_segments, parent, n, pause_time=0.2)

    # Create a manual risky path above, inside the unknown area
    n1_risk = Node(4, 4, 0)
    n2_risk = Node(6, 4, 0)
    goal_risk = Node(8, 4, 0)
    for n, parent in zip([n1_risk, n2_risk, goal_risk, goal], [start, n1_risk, n2_risk, goal_risk]):
        n.parent = parent
        parent.children.append(n)
        n.cost = parent.cost + np.linalg.norm([n.x - parent.x, n.y - parent.y])
        # Add risk from grid
        xi = int((n.x - grid.x_min) / grid.resolution)
        yi = int((n.y - grid.y_min) / grid.resolution)
        raw_p = grid.grid[xi, yi]
        log_s_step = np.log(1 - raw_p) if raw_p > 0 else 0.0
        n.log_survival = parent.log_survival + log_s_step
        n.p_fail = 1 - np.exp(n.log_survival)
        tree.add_node(n)
        update_progress_plot_3d(lc, edge_segments, parent, n, pause_time=0.2)

    # Show the tree before rewiring
    import matplotlib.pyplot as plt
    plt.pause(1.0)

    # test rewire: try to rewire goal_risk to n2 (the optimal path)
    znear = [goal_risk]
    tree.rewire(znear, n2, grid.grid, lc, edge_segments)

    # plot the tree after rewiring
    plt.pause(1.0)

    # After rewiring, goal_risk's parent should be n2, and its cost/log_survival should match the optimal path
    assert goal_risk.parent == n2
    assert np.isclose(goal_risk.cost, n2.cost + np.linalg.norm([goal_risk.x - n2.x, goal_risk.y - n2.y]))
    assert np.isclose(goal_risk.log_survival, n2.log_survival)

    # The risky path should now be "rewired" to the optimal path
    assert n2 in tree.paths[0].nodes or n2 in tree.paths[1].nodes
    assert goal_risk in n2.children

    plt.ioff()
    plt.show()