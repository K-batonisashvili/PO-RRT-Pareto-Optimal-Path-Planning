import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.animation import PillowWriter
import time

# Import core architecture from your theoretical file
from PO_RRT_Star_Theoretical import Node, Tree, Grid, GRID_WIDTH, GRID_HEIGHT
from helper_functions import steer, distance_to, is_edge_collision_free, accumulate_log_survival
from visualization import plot_paths_summary
from PO_RRT_Star_Theoretical import Path, MockNode

# Constants
DEFAULT_STEP_SIZE = 10
COLLISION_SAMPLES = 20
RISK_SAMPLES = 10
TARGET_PATHS = 25
CAPTURE_INTERVAL = 20 # Capture a frame every X iterations

def flush_lazy_queues(tree):
    """Forces all nodes to process their inboxes."""
    for node in tree.node_list:
        if node.dirty:
            tree.process_node_queue(node)

def extract_tree_segments(tree):
    """Extracts all 3D line segments from the array-based lineage."""
    segments = []
    seen_edges = set()
    
    for node in tree.node_list:
        for i, (p_node, p_set_id) in enumerate(node.lineage):
            if p_node is None: continue
            
            idx = np.where(p_node.set_ids == p_set_id)[0]
            if len(idx) == 0: continue
            
            p_L = p_node.costs[idx[0], 1]
            L = node.costs[i, 1]
            
            p_pfail = 1.0 - np.exp(p_L)
            pfail = 1.0 - np.exp(L)
            
            edge_key = (id(p_node), id(node))
            if edge_key not in seen_edges:
                segments.append([(p_node.x, p_node.y, p_pfail), (node.x, node.y, pfail)])
                seen_edges.add(edge_key)
                
    return segments

def extract_goal_segments(goal_node):
    """Backtracks from the goal node to build 3D segments for completed paths."""
    highlight_segments = []
    for i in range(len(goal_node.costs)):
        curr_n = goal_node
        curr_pid = goal_node.set_ids[i]
        
        while curr_n is not None:
            idx = np.where(curr_n.set_ids == curr_pid)[0]
            if len(idx) == 0: break
            
            pn, pset = curr_n.lineage[idx[0]]
            if pn is None: break
            
            idx_pn = np.where(pn.set_ids == pset)[0]
            if len(idx_pn) == 0: break
            
            L_child = curr_n.costs[idx[0], 1]
            L_parent = pn.costs[idx_pn[0], 1]
            
            p1 = (pn.x, pn.y, 1.0 - np.exp(L_parent))
            p2 = (curr_n.x, curr_n.y, 1.0 - np.exp(L_child))
            highlight_segments.append([p1, p2])
            
            curr_n = pn
            curr_pid = pset
            
    return highlight_segments

def main():
    # 1. Replicate Environment
    start = (3, 99)
    goal = (80, 1)
    obstacles = [
        {"type": "circular", "center": (50, 65), "radius": 15, "safe_dist": 5},
        {"type": "rectangular", "x_range": (30, 70), "y_range": (20, 40), "probability": 0.07},
    ]
    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)
    
    # 2. Initialize Tree and Nodes
    p_fail_threshold = 0.95
    tree = Tree(grid, p_fail_threshold=p_fail_threshold)
    
    start_node = Node(*start)
    start_node.is_start = True
    start_node.costs = np.array([[0.0, 0.0]])
    start_node.set_ids = np.array([0], dtype=int)
    start_node.next_id = 1
    start_node.lineage = [(None, -1)]
    tree.add_node(start_node)
    
    goal_node = Node(*goal)
    goal_node.is_goal = True
    tree.add_node(goal_node)

    # 3. Setup Matplotlib 3D Canvas
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(0, GRID_WIDTH); ax.set_ylim(0, GRID_HEIGHT); ax.set_zlim(0, 1)
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Probability of Failure')
    ax.set_title("Live PO-RRT* Tree Generation")
    
    ax.view_init(elev=25, azim=-90)

    # Plot Start and Goal markers
    ax.scatter(*start, 0, c='green', s=100, label="Start")
    ax.scatter(*goal, 0, c='red', s=150, marker='*', label="Goal")
    
    # Draw Obstacles with Legend Labels
    labeled_circle = False
    labeled_rect = False
    for obs in obstacles:
        if obs["type"] == "circular":
            cx, cy, r = obs["center"][0], obs["center"][1], obs["radius"]
            theta = np.linspace(0, 2*np.pi, 50)
            lbl = "Obstacle" if not labeled_circle else ""
            ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), np.zeros(50), c='orange', alpha=0.5, label=lbl)
            labeled_circle = True
        elif obs["type"] == "rectangular":
            x0, x1 = obs["x_range"]; y0, y1 = obs["y_range"]
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], np.zeros(5), c='orange', alpha=0.5)
            labeled_rect = True

    ax.legend(loc='upper right')

    # Collections for tree edges (gray) and final paths (red)
    lc_tree = Line3DCollection([], colors='gray', linewidths=0.8, alpha=0.5)
    lc_paths = Line3DCollection([], colors='red', linewidths=3.0, alpha=1.0)
    ax.add_collection(lc_tree)
    ax.add_collection(lc_paths)

    # 4. Run Generation & Record GIF
    print(f"Starting simulation. Target: {TARGET_PATHS} paths. Generating GIF...")
    rng = np.random.default_rng(50)
    
    writer = PillowWriter(fps=10)
    gif_filename = "porrt_generation.gif"
    
    # Store highlighted segments permanently so they don't vanish if rewired
    cumulative_red_segments = []

    with writer.saving(fig, gif_filename, dpi=100):
        writer.grab_frame()
        
        for iteration in range(2500): 
            rx, ry = float(rng.uniform(0, grid.width)), float(rng.uniform(0, grid.height))
            rand_node = Node(rx, ry)
            nearest_node = tree.nearest(rand_node)
            new_x, new_y = steer(nearest_node, rand_node, DEFAULT_STEP_SIZE)
            new_node = Node(new_x, new_y)
            
            x_idx, y_idx = int(new_node.x), int(new_node.y)
            if not (0 <= x_idx < grid.width and 0 <= y_idx < grid.height and grid.grid[x_idx, y_idx] < 0.9):
                continue

            znear = tree.neighbors(new_node)
            for z in znear:
                if z.dirty: tree.process_node_queue(z)
                
            tree.choose_parents(znear, new_node, grid)
                
            if len(new_node.costs) > 0:
                tree.add_node(new_node)
                tree.rewire(znear, new_node, grid)

                if distance_to(new_node, goal_node) <= DEFAULT_STEP_SIZE:
                    if is_edge_collision_free(new_node, goal_node, grid, COLLISION_SAMPLES, 0.9):
                        d = distance_to(new_node, goal_node)
                        L = accumulate_log_survival(new_node, goal_node, grid, RISK_SAMPLES)
                        shifted_costs = new_node.costs.copy()
                        shifted_costs[:, 0] += d
                        shifted_costs[:, 1] += L
                        
                        valid_mask = (1.0 - np.exp(shifted_costs[:, 1])) <= tree.p_fail_threshold
                        if np.any(valid_mask):
                            goal_node.queue_update({
                                'type': 'add_batch', 'costs': shifted_costs[valid_mask], 
                                'p_node': new_node, 'p_set_ids': new_node.set_ids[valid_mask]
                            })
                            if goal_node.dirty: tree.process_node_queue(goal_node)

            # Capture frame periodically
            if iteration > 0 and iteration % CAPTURE_INTERVAL == 0:
                flush_lazy_queues(tree)
                
                # Update gray exploration tree
                segments = extract_tree_segments(tree)
                lc_tree.set_segments(segments)
                
                # Check for new red paths and accumulate them
                if len(goal_node.costs) > 0:
                    current_goal_segments = extract_goal_segments(goal_node)
                    for seg in current_goal_segments:
                        if seg not in cumulative_red_segments:
                            cumulative_red_segments.append(seg)
                    lc_paths.set_segments(cumulative_red_segments)

                writer.grab_frame()
                print(f"Iter {iteration} | Tree size: {len(tree.node_list)} | Paths reached: {len(goal_node.costs)}")

            # Stop condition
            if len(goal_node.costs) >= TARGET_PATHS:
                print(f"Reached target of {TARGET_PATHS} paths at iteration {iteration}. Rendering final sequence...")
                break
                
        # 5. Render Final Highlighted Sequence
        lc_tree.set_alpha(0.3) # Dim the exploration tree for the final hold
        for _ in range(60): 
            writer.grab_frame()

    # --- Extract Paths for Pareto Visualization ---
    multiple_paths = []
    for i in range(len(goal_node.costs)):
        path_nodes = []
        curr_n = goal_node
        curr_pid = goal_node.set_ids[i]
        c, L = goal_node.costs[i]
        
        valid_path = True
        while curr_n is not None:
            path_nodes.append(MockNode(curr_n.x, curr_n.y, c, 1 - np.exp(L)))
            
            idx = np.where(curr_n.set_ids == curr_pid)[0]
            if len(idx) == 0: 
                valid_path = False
                break
            
            pn, pset = curr_n.lineage[idx[0]]
            if pn is None: break
            
            idx_pn = np.where(pn.set_ids == pset)[0]
            if len(idx_pn) == 0: 
                valid_path = False
                break
                
            c, L = pn.costs[idx_pn[0]]
            curr_n = pn
            curr_pid = pset
            
        if valid_path and getattr(path_nodes[-1], 'cost', 1) == 0.0:
            path_nodes.reverse()
            p = Path()
            p.nodes = path_nodes
            multiple_paths.append({"path": p, "cost": path_nodes[-1].cost, "p_fail": path_nodes[-1].p_fail})

    # Show the Pareto scatter plot
    if multiple_paths:
        print("Opening Pareto Front Visualization...")
        plot_paths_summary(multiple_paths, obstacles=obstacles)

    print(f"Success! GIF saved as '{gif_filename}'")

if __name__ == '__main__':
    main()