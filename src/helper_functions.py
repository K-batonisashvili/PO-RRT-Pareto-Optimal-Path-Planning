import numpy as np

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
    Get node oordinates.
    """
    return (node.x, node.y)

def is_collision_free(node, grid):
    """
    Check if node is inside the grid and not in collision with obstacles.
    """
    x_idx, y_idx = int(node.x), int(node.y)
    return (
        0 <= x_idx < grid.width and
        0 <= y_idx < grid.height and
        grid.grid[x_idx, y_idx] < 0.3  # 0.3 threshold for collision
    )

def steer(from_node, to_node, step_size):
    """
    Steer from one node to another with a given step size.
    """
    dx, dy = to_node.x - from_node.x, to_node.y - from_node.y
    theta = np.arctan2(dy, dx)  # Calculate the angle directly
    return from_node.x + step_size * np.cos(theta), from_node.y + step_size * np.sin(theta), theta

def accumulate_log_survival(parent, child, grid, num_samples=5):
    """
    Accumulate log survival across the edge from parent to child,
    by sampling grid risk at points along the edge.
    """
    xs = np.linspace(parent.x, child.x, num_samples + 1)
    ys = np.linspace(parent.y, child.y, num_samples + 1)

    log_s_step = 0.0

    for i in range(num_samples):
        # midpoint sampling
        x = (xs[i] + xs[i + 1]) / 2
        y = (ys[i] + ys[i + 1]) / 2

        xi = int(x / grid.width * (grid.grid.shape[1] - 1))
        yi = int(y / grid.height * (grid.grid.shape[0] - 1))

        xi = np.clip(xi, 0, grid.grid.shape[1] - 1) 
        yi = np.clip(yi, 0, grid.grid.shape[0] - 1) 

        raw_p = grid.grid[xi, yi] 
        segment_length = distance_to((xs[i], ys[i]), (xs[i + 1], ys[i + 1]))
        log_s_step += np.log(1 - np.clip(raw_p, 0.0, 1.0)) * segment_length

    return log_s_step

    # for x, y in zip(xs, ys):
    #     xi = int(x / grid.width * (grid.grid.shape[0] - 1))
    #     yi = int(y / grid.height * (grid.grid.shape[1] - 1))
    #     if 0 <= xi < grid.grid.shape[0] and 0 <= yi < grid.grid.shape[1]:
    #         raw_p = grid.grid[xi, yi]
    #         p_clip = np.clip(raw_p, 0.0, 1.0)
    #         if p_clip > 0.0:
    #             log_s_step += np.log(1 - p_clip)

    # log_s_step *= distance_to(parent, child) / num_samples
    
    # return log_s_step

def get_path_signature(node_list):
    return tuple((round(n.x, 2), round(n.y, 2), round(n.theta, 2)) for n in node_list)