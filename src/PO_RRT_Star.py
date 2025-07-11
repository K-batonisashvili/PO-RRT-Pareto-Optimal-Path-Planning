import numpy as np
import logging
import tkinter as tk
from tkinter import simpledialog
from helper_functions import distance_to, get_coord, is_collision_free, steer
from visualization import init_progress_plot_3d, update_progress_plot_3d, plot_paths_metrics, redraw_tree, plot_full_paths, plot_paths_summary

logging.basicConfig(level=logging.INFO)

# Define constants
GRID_WIDTH = 100
GRID_HEIGHT = 100
PARETO_RADIUS = 10
DEFAULT_STEP_SIZE = 7.5
PROBABILITY_THRESHOLD = 0.01

# ----------------------- #
#       Main Classes      #
# ----------------------- #

# --------------- Tree Class --------------- #
class Tree:
    """
    Main RRT Tree represented as a class.
    """
    def __init__(self, grid):
        """
        Initialize the tree with a root node and a root path.
        """
        self.paths = []
        self.rewire_counts = 0
        self.rewire_neighbors_count = 0
        self.grid = grid
        self.node_count = 0 # Debugger
        self.path_count = 0 # Debugger

    def add_node(self, node, multiple_children=False):
        """
        Add a node to the tree. If the node's parent has multiple children,
        create a new path for the node without duplicating nodes.
        """
        if node.parent is None:
        # This is the root node
            root_path = Path()
            self.path_count += 1
            root_path.add_node(node)
            self.paths.append(root_path)
        elif not multiple_children and node.parent.path is not None:
            # Continue on the parent's path
            node.parent.path.add_node(node)
        else:
            # Fork: create a new path
            new_path = Path()
            self.path_count += 1
            current = node.parent
            while current:
                new_path.nodes.insert(0, current)
                current = current.parent
            new_path.add_node(node)
            self.paths.append(new_path)
        node.added_to_tree = True
        self.node_count += 1
    
    def remove_node(self, node):
        """
        Completely remove a node from the tree, including all paths and references.
        """
        # Remove the node from all paths
        for path in self.paths:
            if node in path.nodes:
                path.nodes.remove(node)

        # Remove the node from its parent's children list
        if node.parent:
            node.parent.children.remove(node)

        # Remove the node's children (if any)
        for child in node.children:
            child.parent = None

        # Remove the node from the tree's node count
        self.node_count -= 1

        # Log the removal for debugging
        logging.info(f"Removed orphaned node: x={node.x}, y={node.y}, theta={node.theta}")

    def finalize_path(self, goal_node):
        path = Path()
        current = goal_node
        while current:
            path.nodes.insert(0, current)
            current = current.parent
        self.paths.append(path)
        self.path_count += 1


    def nearest(self, rand_node):
        """
        Find the nearest node in the tree to a random node, excluding the goal node.
        """
        all_nodes = [node for path in self.paths for node in path.nodes]
        return min(all_nodes, key=lambda n: distance_to(n, rand_node))
    
    def rewire(self, znear, new_node, grid, lc=None, edge_segments=None):
        """
        Rewire the tree to optimize paths based on cost and failure probability.
        """
        znear = [z for z in znear if distance_to(z, new_node) <= DEFAULT_STEP_SIZE]
        
        for neighbor in znear:
            xi = int(neighbor.x / GRID_WIDTH * (grid.shape[0] - 1))
            yi = int(neighbor.y / GRID_HEIGHT * (grid.shape[1] - 1))
            raw_p = grid[xi, yi]
            if raw_p <= 0.0:
                # no risk here → no change to log-survival
                log_s_step = 0.0
            else:
                # accumulate only in genuinely risky cells
                p_clip     = np.clip(raw_p, 0.0, 1.0)
                log_s_step = np.log(1 - p_clip)
            new_cost = new_node.cost + distance_to(new_node, neighbor)
            new_log_survival = new_node.log_survival + log_s_step
            new_p_fail     = 1 - np.exp(new_log_survival)

            # only rewire if (new_cost,new_p_fail) Pareto-dominates (old_cost,old_p_fail)
            if self.pareto_dominates(new_cost, new_p_fail, neighbor.cost, neighbor.p_fail):
                # detach neighbor from old parent and old path
                old_parent = neighbor.parent
                if old_parent:
                    # Remove the old edge from the plot
                    # update_progress_plot_3d(lc, edge_segments, old_parent, neighbor, remove=True)
                    old_parent.children.remove(neighbor)
                    
                # for child in neighbor.children[:]:
                #     # Remove the edge from the plot
                #     update_progress_plot_3d(lc, edge_segments, neighbor, child, remove=True)

                for path in self.paths:
                    if neighbor in path.nodes:
                        path.nodes.remove(neighbor)
                        break

                # attach neighbor under new_node
                neighbor.parent = new_node
                new_node.children.append(neighbor)

                # update the neighbor’s metrics
                neighbor.cost       = new_cost
                neighbor.log_survival = new_log_survival
                neighbor.p_fail     = new_p_fail

                # add neighbor into the same Path object as new_node
                for path in self.paths:
                    if new_node in path.nodes:
                        path.add_node(neighbor)
                        break

                # Add the new edge to the plot
                # update_progress_plot_3d(lc, edge_segments, new_node, neighbor)


                # propagate down the subtree
                self.propagate_cost(neighbor, grid, lc, edge_segments)

                self.rewire_counts += 1

    def pareto_dominates(self, cost1, fail1, cost2, fail2):
        """
        Check if node1 dominates node2 in terms of cost and failure probability.
        """
        return (cost1 <= cost2 and fail1 < fail2) or \
        (cost1 < cost2 and fail1  <= fail2)

    def propagate_cost(self, root, grid, lc=None, edge_segments=None):
        """
        Iteratively propagate cost & failure updates down the
        subtree of .children under `root`, avoiding recursion.
        """
        queue = [root]
        new_path = root.path
        while queue:
            node = queue.pop(0)
            for child in node.children:
                # 1) cost
                child.cost = node.cost + distance_to(node, child)

                # 2) risk in child's cell
                xi = int(child.x / GRID_WIDTH * (grid.shape[0] - 1))
                yi = int(child.y / GRID_HEIGHT * (grid.shape[1] - 1))
                raw_p = grid[xi, yi]
                if raw_p <= 0.0:
                    log_s_step = 0.0
                else:
                    p_clip     = np.clip(raw_p, 0.0, 1.0)
                    log_s_step = np.log(1 - p_clip)

                # 3) accumulate correctly
                child.log_survival = node.log_survival + log_s_step
                child.p_fail       = 1 - np.exp(child.log_survival)

                # 4) Update the path reference for the child
                child.path = new_path #

                # 5) guard against stray cycles: only follow actual parent→child links
                if child.parent is node:
                    queue.append(child)

                # # Draw the new edge with updated p_fail
                # if lc is not None and edge_segments is not None:
                #     update_progress_plot_3d(lc, edge_segments, node, child)
        # redraw_tree(self, lc, edge_segments)

    def neighbors(self, node):
        """
        Find all unique nodes within a static PARETO_RADIUS around the given node.
        A node is considered unique if it has distinct (x, y, theta, cost, p_fail).
        """
        neighbors = []
        seen = set()  # To track unique nodes based on their attributes
        x_min, x_max = node.x - PARETO_RADIUS, node.x + PARETO_RADIUS
        y_min, y_max = node.y - PARETO_RADIUS, node.y + PARETO_RADIUS

        for path in self.paths:
            for n in path.nodes:
                if n is not node and x_min <= n.x <= x_max and y_min <= n.y <= y_max:
                    if distance_to(n, node) < PARETO_RADIUS:
                        # Create a tuple of attributes to identify duplicates
                        node_signature = (n.x, n.y, n.theta, n.cost, n.p_fail)
                        if node_signature not in seen:
                            neighbors.append(n)
                            seen.add(node_signature)  # Mark this node as seen

        return neighbors

    def choose_parents(self, znear, x, y, theta, grid):
        """
        Instead of picking a single best parent, return a list of new Node()s—
        one for *each* neighbor in Znear that yields a Pareto‐optimal (cost, p_fail)
        pair at the same (x,y,theta).
        """
        test_node = Node(x, y, theta)
        znear = [z for z in znear if distance_to(z, test_node) <= DEFAULT_STEP_SIZE]
        
        # 1) gather all candidate (parent, cost, log_survival, p_fail)
        new_node_candidates = []
        xi = int(x / GRID_WIDTH * (grid.shape[0] - 1))
        yi = int(y / GRID_HEIGHT * (grid.shape[1] - 1))
        raw_p = grid[xi, yi]
        if raw_p <= 0.0:
            # no risk here → no change to log-survival
            log_s_step = 0.0
        else:
            # accumulate only in genuinely risky cells
            p_clip     = np.clip(raw_p, 0.0, 1.0)
            log_s_step = np.log(1 - p_clip)

        for potential_parent in znear:
            cost   = potential_parent.cost + distance_to(potential_parent, Node(x,y,theta))
            log_survival = potential_parent.log_survival + log_s_step
            prob_failure  = 1 - np.exp(log_survival)
            new_node_candidates.append((potential_parent, cost, log_survival, prob_failure))

        # 2) filter out dominated candidates
        pareto_dominant_nodes = []
        for pa in new_node_candidates:
            dominated = False
            for pb in new_node_candidates:
                if (pb is not pa) and self.pareto_dominates(pb[1], pb[3], pa[1], pa[3]):
                    dominated = True
                    break
            if not dominated:
                pareto_dominant_nodes.append(pa) # This ensures we do not lose any Pareto‐optimal candidates, even if they are not unique.

        # --- 3) Build exactly one new Node per remaining parent ---
        final_pareto_nodes = []
        for potential_parent, cost, log_surv, p_fail in pareto_dominant_nodes:
            new_node = Node(x, y, theta)
            new_node.parent         = potential_parent
            potential_parent.children.append(new_node)
            new_node.cost           = cost
            new_node.log_survival   = log_surv
            new_node.p_fail         = p_fail
            final_pareto_nodes.append(new_node)

        if len(final_pareto_nodes) > 8:
            logging.info(f"Found {len(final_pareto_nodes)} Pareto‐optimal nodes.")
            for node in final_pareto_nodes:
                logging.info(f"Node: {node}, Cost: {node.cost}, p_fail: {node.p_fail}")

        return final_pareto_nodes

    def get_path_to(self, goal_node):
        """
        Get the path from the start node to the goal node.
        Only returns a path if it starts at the root and ends at the specified goal_node.
        """
        for path in self.paths:
            if path.nodes and path.nodes[0].parent is None and path.nodes[-1] is goal_node:
                return path
        return None

    
# --------------- Tree Class --------------- #



# --------------- Path Class --------------- #
class Path:
    def __init__(self):
        self.nodes = []

    def add_node(self, node):
        self.nodes.append(node)
        node.path = self  # Set the path reference for the node
    
    @property
    def cost(self) -> float:
        # cost‐to‐come of the last node
        return self.nodes[-1].cost if self.nodes else 0.0

    @property
    def p_fail(self) -> float:
        # failure probability at the last node
        return self.nodes[-1].p_fail if self.nodes else 1.0
    
    @property
    def is_complete(self) -> bool:
        """
        Check if the path is complete, i.e., it starts at the root and ends at a goal node.
        """
        return self.nodes and self.nodes[0].is_start and self.nodes[-1].is_goal and \
               sum(1 for n in self.nodes if n.is_goal) == 1
    
    def __repr__(self):
        return f"Path(len={len(self.nodes)}, cost={self.cost:.2f}, p_fail={self.p_fail:.2f})"

# --------------- Path Class --------------- #








# --------------- Node Class --------------- #
class Node:
    """
    Class representing a node in the RRT tree.
    """
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.parent = None
        self.children = []
        self.cost = 0.0
        self.p_fail = 0.0
        self.log_survival = 0.0
        self.added_to_tree = False
        self.path = None  # Reference to the path this node belongs to
        self.is_goal = False  # Flag to indicate if this node is the goal node
        self.is_start = False  # Flag to indicate if this node is the start node

# --------------- Node Class --------------- #







# --------------- Grid Class --------------- #
class Grid:
    """
    Class representing the occupancy grid.
    """
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.obstacles = obstacles

        # Process all obstacles
        for obstacle in obstacles:
            if obstacle["type"] == "circular":
                self.add_circular_obstacle(
                    center=obstacle["center"],
                    radius=obstacle["radius"],
                    safe_dist=obstacle["safe_dist"]
                )
            elif obstacle["type"] == "rectangular":
                self.add_unknown_area(
                    x_range=obstacle["x_range"],
                    y_range=obstacle["y_range"],
                    probability=obstacle["probability"]
                )

    def add_circular_obstacle(self, center, radius, safe_dist):
        """
        Add a circular obstacle to the grid, prioritizing the highest probability.
        """
        cx = int(center[0] / self.width * (self.width - 1))
        cy = int(center[1] / self.height * (self.height - 1))
        rad_cells = int(radius / self.width * (self.width - 1))
        safe_cells = int(safe_dist / self.width * (self.width - 1))

        for x in range(cx - rad_cells - safe_cells, cx + rad_cells + safe_cells + 1):
            for y in range(cy - rad_cells - safe_cells, cy + rad_cells + safe_cells + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist <= rad_cells:
                        new_prob = 0.9
                    elif dist <= rad_cells + safe_cells:
                        new_prob = 0.9 * (1 - (dist - rad_cells) / safe_cells)
                    else:
                        new_prob = 0.0

                    # Update the grid only if the new probability is higher
                    self.grid[x][y] = max(self.grid[x][y], new_prob)

    def add_unknown_area(self, x_range, y_range, probability):
        """
        Add a rectangular unknown area to the grid, prioritizing the highest probability.
        """
        x_start = int(x_range[0] / self.width * (self.width - 1))
        x_end = int(x_range[1] / self.width * (self.width - 1))
        y_start = int(y_range[0] / self.height * (self.height - 1))
        y_end = int(y_range[1] / self.height * (self.height - 1))

        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[x][y] = max(self.grid[x][y], probability)

# --------------- Grid Class --------------- #








##################################
## CENTRAL PO_RRT_STAR FUNCTION ##
##################################
def PO_RRT_Star(start, goal, grid, failure_prob_values, max_iter=4500, step_size=DEFAULT_STEP_SIZE, threshold=PROBABILITY_THRESHOLD):
    # Initialize the tree and nodes
    start_node, goal_node = Node(*start), Node(*goal)
    tree = Tree(grid)
    tree.add_node(start_node)
    goal_node.is_goal = True
    start_node.is_start = True
    multiple_paths = []

    fig, ax, lc, edge_segments = init_progress_plot_3d(
    start,
    goal,
    x_lim=(0, grid.width),
    y_lim=(0, grid.height),
    obstacles=grid.obstacles,
    )   

    
    for current_iter in range(max_iter):
        if current_iter % 1000 == 0:
            # Update the plot every 1000 iterations
            redraw_tree(tree, lc, edge_segments)
        # Random node sample
        rand_node = Node(np.random.uniform(0, grid.width), np.random.uniform(0, grid.height), np.random.uniform(-np.pi, np.pi))
        if is_collision_free(rand_node, grid):  
            # Find the nearest node in the tree and steer to new node from it
            nearest_node = tree.nearest(rand_node)
            x, y, theta = steer(nearest_node, rand_node, step_size)
            new_node = Node(x, y, theta)
            if is_collision_free(new_node, grid):
                # Check if the new node is collision-free              
                znear = tree.neighbors(new_node)
                
                # Check if znear contains any nodes, if it doesn't, set the nearest node as znear
                # if not znear:
                #     znear = [nearest_node]

                # Exclude the goal node from znear to prevent rewiring through it
                znear = [n for n in znear if n.x != goal_node.x or n.y != goal_node.y]
                
                # if len(start_node.children) > 5:
                #     print(f"Start node has {len(start_node.children)} children")
                #     print(f"Znear: {new_nodes}")   

                new_nodes = tree.choose_parents(
                        znear, 
                        new_node.x, 
                        new_node.y, 
                        new_node.theta, 
                        grid.grid
                    )   
                    
                if len(new_nodes) > 1:
                    # multiple possible parents 
                    for nn in new_nodes:
                        multiple_children = True if len(nn.parent.children) > 1 else False
                        # if len(nn.parent.children) > 10:
                        #     print(f"This is the parent: {nn.parent}")
                        #     print(f"Multiple children: {len(nn.parent.children)}")                        # 1) goal check per branch
                        if distance_to(nn, goal) < step_size:
                            if not any(child.is_goal for child in nn.children): # Safeguard against adding multiple goal nodes
                            # connect to goal exactly once per branch
                                goal_instance = Node(goal[0], goal[1], goal[2])
                                goal_instance.is_goal = True
                                goal_instance.parent = nn
                                nn.children.append(goal_instance)
                                goal_instance.cost = nn.cost + distance_to(nn, goal_instance)
                                goal_instance.log_survival = nn.log_survival
                                goal_instance.p_fail = 1 - np.exp(goal_instance.log_survival)
                                
                                if not nn.added_to_tree: # safeguard against adding the same node multiple times
                                    tree.add_node(nn, multiple_children=multiple_children)
                                tree.finalize_path(goal_instance)
                                print(f"Goal node added to tree with cost: {goal_instance.cost}, p_fail: {goal_instance.p_fail}")

                                # raw_path = tree.get_path_to(goal_instance)
                                # if raw_path is not None:
                                #     multiple_paths.append({
                                #         "path": raw_path,
                                #         "cost": raw_path.cost,
                                #         "p_fail": raw_path.p_fail
                                #     })
                                # redraw_tree(tree, lc, edge_segments)
                        else:
                            tree.add_node(nn, multiple_children=multiple_children)
                            tree.rewire(tree.neighbors(nn), nn, grid.grid, lc, edge_segments)
                            # update_progress_plot_3d(lc, edge_segments, nn.parent, nn)
                else:
                    # single child branch
                    for nn in new_nodes:
                        multiple_children = True if len(nn.parent.children) > 1 else False
                        if distance_to(nn, goal) < step_size:
                            if not any(child.is_goal for child in nn.children): # Safeguard against adding multiple goal nodes
                                # ─── goal handling ─────────────────────────
                                goal_instance = Node(goal[0], goal[1], goal[2])
                                goal_instance.is_goal = True
                                goal_instance.parent = nn
                                nn.children.append(goal_instance)
                                goal_instance.cost = nn.cost + distance_to(nn, goal_instance)
                                goal_instance.log_survival = nn.log_survival
                                goal_instance.p_fail = 1 - np.exp(goal_instance.log_survival)

                                if not nn.added_to_tree: # safeguard against adding the same node multiple times
                                    tree.add_node(nn, multiple_children=multiple_children)
                                tree.finalize_path(goal_instance)
                                print(f"Goal node added to tree with cost: {goal_instance.cost}, p_fail: {goal_instance.p_fail}, current_iter: {current_iter}")

                                # raw_path = tree.get_path_to(goal_instance)
                                # if raw_path is not None:
                                #     multiple_paths.append({
                                #         "path": raw_path,
                                #         "cost": raw_path.cost,
                                #         "p_fail": raw_path.p_fail
                                #     })
                                # redraw_tree(tree, lc, edge_segments)
                        # 2) Not near goal, so add the new node to the tree
                        else:
                            # Add the new node to the tree
                            # multiple_children = True if len(nn.parent.children) > 1 else False

                            tree.add_node(nn, multiple_children=multiple_children)
                            tree.rewire(tree.neighbors(nn), nn, grid.grid, lc, edge_segments)
                            # update_progress_plot_3d(lc, edge_segments, nn.parent, nn)

    # Extract multiple paths from the tree
    multiple_paths = [
                        {
                            "path": p,
                            "cost": p.cost,
                            "p_fail": p.p_fail
                        }
                        for p in tree.paths if p.is_complete
                    ]
    
    
    print("\n--- Debug: Paths from start to goal ---")
    for idx, entry in enumerate(multiple_paths):
        path = entry["path"]
        if hasattr(path, "nodes"):
            nodes = path.nodes
        else:
            nodes = path
        print(f"Path {idx+1}:")
        for node in nodes:
            print(f"  (x={node.x:.2f}, y={node.y:.2f}, theta={node.theta:.2f}, cost={node.cost:.2f}, p_fail={node.p_fail:.4f})"
            + (" [START]" if getattr(node, "is_start", False) else "")
            + (" [GOAL]" if getattr(node, "is_goal", False) else ""))
        print(f"Path {idx+1} ends at: (x={path.nodes[-1].x}, y={path.nodes[-1].y}) is_goal={getattr(path.nodes[-1], 'is_goal', False)}")
        print("-" * 40)

    # After collecting multiple_paths, filter out dominated paths using pareto_dominates

    def pareto_filter(paths):
        non_dominated = []
        for i, entry_i in enumerate(paths):
            dominated = False
            for j, entry_j in enumerate(paths):
                if i != j:
                    # If entry_j dominates entry_i, mark as dominated
                    if (entry_j["cost"] <= entry_i["cost"] and entry_j["p_fail"] < entry_i["p_fail"]) or \
                    (entry_j["cost"] < entry_i["cost"] and entry_j["p_fail"] <= entry_i["p_fail"]):
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(entry_i)
        return non_dominated

    filtered_paths = pareto_filter(multiple_paths)

    return filtered_paths, multiple_paths 

##################################
## CENTRAL PO_RRT_STAR FUNCTION ##
##################################



# Main code
def main():
    
    # Create the main application window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    obstacles = []

    
    start, goal = (10, 70, 0), (75, 5, 0)

    # Obstacle dictionary
    obstacles = [
        # {"type": "circular", "center": (50, 80), "radius": 10, "safe_dist": 5},
        {"type": "rectangular", "x_range": (15, 45), "y_range": (10, 40), "probability": 0.05},
        {"type": "rectangular", "x_range": (50, 70), "y_range": (45, 60), "probability": 0.05},
        {"type": "circular", "center": (60, 30), "radius": 10, "safe_dist": 5},
    ]

    
    failure_prob_values = simpledialog.askstring("Input", "Enter the failure probabilities (comma-separated):")
    if failure_prob_values:
        failure_prob_values = [float(x.strip()) for x in failure_prob_values.split(',')]
    else:
        failure_prob_values = [0.1, 0.2, 0.3]  # Default values if none provided

    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

    filtered_paths, multiple_paths = PO_RRT_Star(start, goal, grid, failure_prob_values)
    # plot_paths_metrics(multiple_paths)
    # plot_full_paths(multiple_paths)
    plot_paths_summary(filtered_paths, obstacles=obstacles)
    plot_paths_summary(multiple_paths, obstacles=obstacles)
    

if __name__ == '__main__':
    main()