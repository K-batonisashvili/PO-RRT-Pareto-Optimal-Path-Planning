import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import PillowWriter
import math

from PO_RRT_Star_Theoretical import Node, Grid, GRID_WIDTH, GRID_HEIGHT
from helper_functions import steer, distance_to, distance_sq, is_edge_collision_free, accumulate_log_survival

# Constants
DEFAULT_STEP_SIZE = 10
COLLISION_SAMPLES = 20
RISK_SAMPLES = 10
TARGET_PATHS = 1 
CAPTURE_INTERVAL = 20

# SCALAR WEIGHTS
W1 = 0.05  # Weight for Distance
W2 = 0.95  # Weight for P_fail

def extract_baseline_segments(nodes):
    segments = []
    for n in nodes:
        if n.parent is not None:
            segments.append([(n.parent.x, n.parent.y, n.parent.p_fail), 
                             (n.x, n.y, n.p_fail)])
    return segments

def extract_baseline_goal(goal_node):
    segments = []
    curr = goal_node
    while curr.parent is not None:
        segments.append([(curr.parent.x, curr.parent.y, curr.parent.p_fail), 
                         (curr.x, curr.y, curr.p_fail)])
        curr = curr.parent
    return segments

def _is_ancestor(candidate, node):
    cur = node
    while cur is not None:
        if cur is candidate: return True
        cur = cur.parent
    return False

def _propagate(root, grid):
    """Propagate metric improvements down the baseline tree after a rewire."""
    stack = [root]
    while stack:
        curr = stack.pop()
        for child in getattr(curr, 'children', []):
            child.cost = curr.cost + distance_to(curr, child)
            child.log_survival = curr.log_survival + accumulate_log_survival(curr, child, grid, RISK_SAMPLES)
            child.p_fail = 1 - np.exp(child.log_survival)
            stack.append(child)

def main():
    start = (3, 99)
    goal = (80, 1)
    obstacles = [
        {"type": "circular", "center": (50, 65), "radius": 15, "safe_dist": 5},
        {"type": "rectangular", "x_range": (30, 70), "y_range": (20, 40), "probability": 0.07},
    ]
    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)
    
    start_node = Node(*start)
    start_node.is_start = True
    start_node.parent = None
    start_node.children = []
    start_node.cost = 0.0
    start_node.log_survival = 0.0
    start_node.p_fail = 0.0
    nodes = [start_node]
    
    goal_template = Node(*goal)
    best_goal_node = None

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, GRID_WIDTH); ax.set_ylim(0, GRID_HEIGHT); ax.set_zlim(0, 1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Probability of Failure')
    ax.set_title(f"Live Standard RRT* (Weighted Focus: W1={W1}, W2={W2})")
    ax.view_init(elev=25, azim=-90)

    ax.scatter(*start, 0, c='green', s=100, label="Start")
    ax.scatter(*goal, 0, c='red', s=150, marker='*', label="Goal")
    
    labeled_circle = False
    labeled_rect = False
    for obs in obstacles:
        if obs["type"] == "circular":
            cx, cy, r = obs["center"][0], obs["center"][1], obs["radius"]
            th = np.linspace(0, 2*np.pi, 50)
            lbl = "Obstacle" if not labeled_circle else ""
            ax.plot(cx + r*np.cos(th), cy + r*np.sin(th), np.zeros(50), c='orange', alpha=0.5, label=lbl)
            labeled_circle = True
        elif obs["type"] == "rectangular":
            x0, x1 = obs["x_range"]; y0, y1 = obs["y_range"]
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], np.zeros(5), c='orange', alpha=0.5)
            labeled_rect = True

    ax.legend(loc='upper right')

    lc_tree = Line3DCollection([], colors='gray', linewidths=0.8, alpha=0.5)
    lc_paths = Line3DCollection([], colors='blue', linewidths=3.0, alpha=1.0)
    ax.add_collection(lc_tree)
    ax.add_collection(lc_paths)

    print("Generating Weighted Standard RRT* Baseline GIF...")
    rng = np.random.default_rng(50) # EXACT SAME SEED
    writer = PillowWriter(fps=15)
    gif_filename = "baseline_weighted_rrt.gif"

    with writer.saving(fig, gif_filename, dpi=100):
        writer.grab_frame()
        
        for iteration in range(2500): 
            rx, ry = float(rng.uniform(0, grid.width)), float(rng.uniform(0, grid.height))
            rand_node = Node(rx, ry)
            
            x_idx, y_idx = int(rand_node.x), int(rand_node.y)
            if not (0 <= x_idx < grid.width and 0 <= y_idx < grid.height and grid.grid[x_idx, y_idx] < 0.9):
                continue
                
            nn = min(nodes, key=lambda n: distance_sq(n, rand_node))
            new_x, new_y = steer(nn, rand_node, DEFAULT_STEP_SIZE)
            new_node = Node(new_x, new_y)
            new_node.children = []
            
            x_idx, y_idx = int(new_node.x), int(new_node.y)
            if not (0 <= x_idx < grid.width and 0 <= y_idx < grid.height and grid.grid[x_idx, y_idx] < 0.9):
                continue

            radius = 10.0
            r2 = radius * radius
            znear = [n for n in nodes if distance_sq(n, new_node) <= r2]
            if not znear: znear = [nn]

            best_parent, best_scalar, best_cost, best_L, best_pfail = None, math.inf, None, None, None
            
            # --- EVALUATE WEIGHTED PARENTS ---
            for z in znear:
                if not is_edge_collision_free(z, new_node, grid, COLLISION_SAMPLES, 0.9): continue
                d_edge = distance_to(z, new_node)
                L_step = accumulate_log_survival(z, new_node, grid, RISK_SAMPLES)
                
                c = z.cost + d_edge
                L = z.log_survival + L_step
                pfail = 1 - np.exp(L)
                
                # Combine distance and risk into a single scalar value
                scalar = (W1 * c) + (W2 * pfail)
                
                if scalar < best_scalar:
                    best_scalar = scalar
                    best_cost = c
                    best_L = L
                    best_pfail = pfail
                    best_parent = z
                    
            if best_parent is None: continue

            new_node.parent = best_parent
            best_parent.children.append(new_node)
            new_node.cost = best_cost
            new_node.log_survival = best_L
            new_node.p_fail = best_pfail
            nodes.append(new_node)

            # --- WEIGHTED REWIRE ---
            for z in znear:
                if z is best_parent: continue
                if not is_edge_collision_free(new_node, z, grid, COLLISION_SAMPLES, 0.9): continue
                d_edge = distance_to(new_node, z)
                L_step = accumulate_log_survival(new_node, z, grid, RISK_SAMPLES)
                
                cand_cost = new_node.cost + d_edge
                cand_L = new_node.log_survival + L_step
                cand_pfail = 1 - np.exp(cand_L)
                cand_scalar = (W1 * cand_cost) + (W2 * cand_pfail)
                
                z_scalar = (W1 * z.cost) + (W2 * z.p_fail)
                
                if cand_scalar + 1e-9 < z_scalar and not _is_ancestor(z, new_node):
                    # Remove from old parent
                    if z.parent and z in z.parent.children:
                        z.parent.children.remove(z)
                        
                    # Reassign to new parent
                    z.parent = new_node
                    new_node.children.append(z)
                    z.cost = cand_cost
                    z.log_survival = cand_L
                    z.p_fail = cand_pfail
                    
                    # Propagate improvements downstream
                    _propagate(z, grid)

            # Goal check
            if distance_to(new_node, goal_template) <= DEFAULT_STEP_SIZE:
                if is_edge_collision_free(new_node, goal_template, grid, COLLISION_SAMPLES, 0.9):
                    gn = Node(*goal)
                    gn.parent = new_node
                    gn.cost = new_node.cost + distance_to(new_node, goal_template)
                    gn.log_survival = new_node.log_survival + accumulate_log_survival(new_node, goal_template, grid, RISK_SAMPLES)
                    gn.p_fail = 1 - np.exp(gn.log_survival)
                    best_goal_node = gn

            if iteration > 0 and iteration % CAPTURE_INTERVAL == 0:
                lc_tree.set_segments(extract_baseline_segments(nodes))
                
                if best_goal_node:
                    lc_paths.set_segments(extract_baseline_goal(best_goal_node))
                
                writer.grab_frame()
                print(f"Iter {iteration} | Tree size: {len(nodes)}")

            if best_goal_node is not None:
                print(f"Goal reached at iteration {iteration}!")
                break
        if best_goal_node:
            lc_tree.set_segments(extract_baseline_segments(nodes))
            lc_paths.set_color('red') # Match PO-RRT* style
            lc_paths.set_segments(extract_baseline_goal(best_goal_node))
                
        # Render Final
        lc_tree.set_alpha(0.4)
        for _ in range(60): 
            writer.grab_frame()

    print(f"Success! Baseline GIF saved as '{gif_filename}'")

if __name__ == '__main__':
    main()