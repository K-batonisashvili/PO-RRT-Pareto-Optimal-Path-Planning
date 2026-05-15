import numpy as np
import math

# ----------------------- #
#     Helper Functions    #
# ----------------------- #
def distance_to(a, b):
    """Return Euclidean distance between two (x,y) points or Node objects.

    Use math.hypot for scalar speed and avoid NumPy overhead in hot loops.
    """
    # pull out (x,y) whether it's a Node or a bare tuple/list
    x1, y1 = (a.x, a.y) if hasattr(a, 'x') else (a[0], a[1])
    x2, y2 = (b.x, b.y) if hasattr(b, 'x') else (b[0], b[1])
    return math.hypot(x1 - x2, y1 - y2)


def distance_sq(a, b):
    """Return squared Euclidean distance (avoids sqrt when only comparisons are needed)."""
    x1, y1 = (a.x, a.y) if hasattr(a, 'x') else (a[0], a[1])
    x2, y2 = (b.x, b.y) if hasattr(b, 'x') else (b[0], b[1])
    dx = x1 - x2
    dy = y1 - y2
    return dx*dx + dy*dy

def get_coord(node):
    """
    Get node oordinates.
    """
    return (node.x, node.y)

def is_edge_collision_free(a, b, grid, num_samples=50, p_threshold=0.9):
    """Check if the straight edge from node a to node b is collision-free.

    This optimized version avoids creating intermediate NumPy arrays in hot
    loops and uses simple arithmetic to sample along the segment.
    """
    ax, ay = (a.x, a.y) if hasattr(a, "x") else (a[0], a[1])
    bx, by = (b.x, b.y) if hasattr(b, "x") else (b[0], b[1])

    # guard trivial case
    if ax == bx and ay == by:
        x_idx = int(ax)
        y_idx = int(ay)
        if not (0 <= x_idx < grid.width and 0 <= y_idx < grid.height):
            return False
        return grid.grid[x_idx, y_idx] < p_threshold

    dx = (bx - ax) / max(1, num_samples)
    dy = (by - ay) / max(1, num_samples)

    x = ax
    y = ay
    for i in range(num_samples + 1):
        x_idx = int(x)
        y_idx = int(y)

        if not (0 <= x_idx < grid.width and 0 <= y_idx < grid.height):
            return False
        if grid.grid[x_idx, y_idx] >= p_threshold:
            return False

        x += dx
        y += dy

    return True


def is_collision_free(node, grid):
    """
    Check if node is inside the grid and not in collision with obstacles.
    """
    x_idx, y_idx = int(node.x), int(node.y)
    return (
        0 <= x_idx < grid.width and
        0 <= y_idx < grid.height and
        grid.grid[x_idx, y_idx] < 0.9
    )

def steer(from_node, to_node, step_size):
    """
    Steer from one node to another with a given step size.
    """
    dx, dy = to_node.x - from_node.x, to_node.y - from_node.y
    theta = np.arctan2(dy, dx)  # Calculate the angle directly
    return from_node.x + step_size * np.cos(theta), from_node.y + step_size * np.sin(theta)

def accumulate_log_survival(parent, child, grid, num_samples=5):
    """Accumulate log survival across the edge from parent to child.

    Avoid NumPy allocations by doing simple arithmetic in the loop.
    """
    log_s_step = 0.0

    dx = (child.x - parent.x) / max(1, num_samples)
    dy = (child.y - parent.y) / max(1, num_samples)
    segment_length = (dx * dx + dy * dy) ** 0.5

    import math
    for i in range(num_samples):
        # midpoint sampling
        xm = parent.x + (i + 0.5) * dx
        ym = parent.y + (i + 0.5) * dy

        # Map continuous coordinates to grid indices (grid.grid shape is (width, height))
        xi = int(xm / grid.width * (grid.grid.shape[0] - 1))
        yi = int(ym / grid.height * (grid.grid.shape[1] - 1))

        xi = max(0, min(xi, grid.grid.shape[0] - 1))
        yi = max(0, min(yi, grid.grid.shape[1] - 1))

        raw_p = grid.grid[xi, yi]
        # clip to [0, 1 - eps] to avoid log(0)
        rp = raw_p
        if rp < 0.0:
            rp = 0.0
        elif rp >= 1.0:
            rp = 1.0 - 1e-12

        # log survival contribution (use math.log which is faster for scalars)
        log_s_step += math.log(1 - rp) * segment_length

    return log_s_step

def get_path_signature(node_list):
    return tuple((round(n.x, 2), round(n.y, 2)) for n in node_list)