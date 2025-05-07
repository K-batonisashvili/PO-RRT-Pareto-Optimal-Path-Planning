import numpy as np
import matplotlib.pyplot as plt

# ----------------------- #
#     Helper Functions    #
# ----------------------- #
def distance_to(a, b):
    # pull out (x,y) whether it's a Node or a bare tuple/list
    x1, y1 = (a.x, a.y) if hasattr(a, 'x') else (a[0], a[1])
    x2, y2 = (b.x, b.y) if hasattr(b, 'x') else (b[0], b[1])
    return np.hypot(x1 - x2, y1 - y2)

def get_coord(node):
    """
    Get node goordinates.
    """
    return (node.x, node.y)

def is_collision_free(node, grid):
    """
    Check if node is inside the grid and not in collision with obstacles.
    """
    x_idx, y_idx = int((node.x - grid.x_min) / grid.resolution), int((node.y - grid.y_min) / grid.resolution)
    return 0 <= x_idx < grid.width and 0 <= y_idx < grid.height and grid.grid[x_idx, y_idx] < 0.3 # 0.3 threshold for collision

def steer(from_node, to_node, step_size):
    """
    Steer from one node to another with a given step size.
    """
    dx, dy = to_node.x - from_node.x, to_node.y - from_node.y
    theta = np.arctan2(dy, dx)  # Calculate the angle directly
    return from_node.x + step_size * np.cos(theta), from_node.y + step_size * np.sin(theta), theta