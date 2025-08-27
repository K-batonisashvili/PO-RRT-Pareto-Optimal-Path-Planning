import numpy as np
import logging
import tkinter as tk
from tkinter import simpledialog
from helper_functions import distance_to, get_coord, is_collision_free, steer, accumulate_log_survival, get_path_signature
from visualization import init_progress_plot_3d, update_progress_plot_3d, plot_paths_metrics, redraw_tree, plot_full_paths, plot_paths_summary
from scipy.spatial import cKDTree

logging.basicConfig(level=logging.INFO)

# Define constants
GRID_WIDTH = 100
GRID_HEIGHT = 100
PARETO_RADIUS = 8
DEFAULT_STEP_SIZE = 8
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
        self.node_list = []
        self.rewire_counts = 0
        self.additional_rewire_nodes = 0  # Additional rewire nodes        
        self.rewire_neighbors_count = 0
        self.grid = grid
        self.node_count = 0 # Debugger
        self.path_count = 0 # Debugger
        self.start_node = None

    def add_node(self, node, multiple_children=False):
        """
        Add a node to the tree. If the node's parent has multiple children,
        create a new path for the node without duplicating nodes.
        """
        if node.parent is not None:
            d = distance_to(node, node.parent)
            if d > DEFAULT_STEP_SIZE + 1e-3:
                print(f" [add_node] Illegal jump: {d:.2f} from parent at ({node.parent.x:.2f}, {node.parent.y:.2f}) "
                                f"to child at ({node.x:.2f}, {node.y:.2f})")
        if node not in self.node_list:
            self.node_list.append(node)
        if node.parent is None:
        # This is the root node
            root_path = Path()
            self.path_count += 1
            root_path.add_node(node)
            self.paths.append(root_path)
        elif not multiple_children and node.parent.path is not None:
            # Continue on the parent's path
            node.parent.path.add_node(node)
            node.added_to_tree = True
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
        stack = []
        while current:
            stack.append(current)
            current = current.parent

        # Ensure path starts at the actual start node
        if stack and stack[-1].is_start:
            path_nodes = list(reversed(stack))
            signature = get_path_signature(path_nodes)

            # Check for duplicate signature
            if hasattr(self, "_finalized_signatures") and signature in self._finalized_signatures:
                return  # Already finalized
            else:
                if not hasattr(self, "_finalized_signatures"):
                    self._finalized_signatures = set()
                self._finalized_signatures.add(signature)

            # Build and store path
            for node in path_nodes:
                path.add_node(node)
            self.paths.append(path)
            self.path_count += 1


    def nearest(self, rand_node):
        if not self.node_list:
            return None

        coords = [(n.x, n.y) for n in self.node_list]
        kdtree = cKDTree(coords)
        _, idx = kdtree.query([rand_node.x, rand_node.y], k=1)
        return self.node_list[idx]
    
    
    def pareto_dominates(self, cost1, fail1, cost2, fail2):
        """
        Check if node1 dominates node2 in terms of cost and failure probability.
        """
        return (cost1 <= cost2 and fail1 < fail2) or \
        (cost1 < cost2 and fail1  <= fail2)
    
    def neighbors(self, node):
        """
        Efficiently find all unique nodes within PARETO_RADIUS using cKDTree.
        """
        if not hasattr(self, '_kdtree') or len(self.node_list) != getattr(self, '_kdtree_node_count', -1):
            coords = [(n.x, n.y) for n in self.node_list]
            self._kdtree = cKDTree(coords)
            self._kdtree_node_count = len(self.node_list)

        # Query neighbors within radius
        idxs = self._kdtree.query_ball_point([node.x, node.y], r=PARETO_RADIUS)

        neighbors = []
        seen = set()

        for idx in idxs:
            n = self.node_list[idx]
            if n is node:
                continue

            node_signature = (n.x, n.y, n.theta, round(n.cost, 3), round(n.p_fail, 5))
            if node_signature not in seen:
                neighbors.append(n)
                seen.add(node_signature)

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
        
        for potential_parent in znear:
            log_s_step = accumulate_log_survival(potential_parent, test_node, grid)
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
            # Check if the potential parent is too far from the new node
            if distance_to(potential_parent, Node(x, y, theta)) > DEFAULT_STEP_SIZE + 1e-3:
                print(f" [choose_parents] Illegal parent assignment: jump from "
                                f"({potential_parent.x:.2f}, {potential_parent.y:.2f}) → ({x:.2f}, {y:.2f})")

            new_node = Node(x, y, theta)
            new_node.parent         = potential_parent
            potential_parent.children.append(new_node)
            new_node.cost           = cost
            new_node.log_survival   = log_surv
            new_node.p_fail         = p_fail
            final_pareto_nodes.append(new_node)
            # --- DEBUG: Check p_fail monotonicity ---
            if new_node.parent and new_node.p_fail < new_node.parent.p_fail - 1e-8:
                print(f"DEBUG BREAK: p_fail decreased from parent to child during node creation!")
                print(f"  Parent p_fail: {new_node.parent.p_fail:.6f}, Child p_fail: {new_node.p_fail:.6f}")
                import pdb; pdb.set_trace()  # <-- This is a breakpoint for debugging
        # if len(final_pareto_nodes) > 8:
        #     logging.info(f"Found {len(final_pareto_nodes)} Pareto‐optimal nodes.")
        #     for node in final_pareto_nodes:
        #         logging.info(f"Node: {node}, Cost: {node.cost}, p_fail: {node.p_fail}")

        return final_pareto_nodes
    
    def rewire(self, znear, nn, grid, lc=None, edge_segments=None):
        """
        Rewire the tree to optimize paths based on cost and failure probability.
        Now allows all non-dominated (Pareto-optimal) rewires, not just strictly dominating ones.
        """
        znear = [z for z in znear if distance_to(z, nn) <= DEFAULT_STEP_SIZE]
        # 1) Gather all candidate rewires
        rewire_candidates = []
        for z in znear:
            if z is nn or nn in z.children:
                continue  # skip self-loop or cycle
            if distance_to(nn, z) > DEFAULT_STEP_SIZE + 1e-3:
                print(f" [rewire] Illegal rewire: distance = {distance_to(nn, z):.2f} "
                      f"from ({nn.x:.2f}, {nn.y:.2f}) to ({z.x:.2f}, {z.y:.2f})")
            log_s_step = accumulate_log_survival(nn, z, grid)
            new_log_survival = nn.log_survival + log_s_step
            new_cost = nn.cost + distance_to(nn, z)
            new_p_fail = 1 - np.exp(new_log_survival)
            rewire_candidates.append((z, new_cost, new_log_survival, new_p_fail))

        # 2) Pareto filter the candidates
        pareto_rewires = []
        for ca in rewire_candidates:
            dominated = False
            for cb in rewire_candidates:
                if (ca is not cb) and self.pareto_dominates(cb[1], cb[3], ca[1], ca[3]):
                    dominated = True
                    break
            if not dominated:
                pareto_rewires.append(ca)

        # 3) For each non-dominated candidate, perform the rewire
        for z, new_cost, new_log_survival, new_p_fail in pareto_rewires:
            # Check if this is a strictly dominant rewire
            strictly_dominant = self.pareto_dominates(new_cost, new_p_fail, z.cost, z.p_fail)

            if strictly_dominant:
                # Detach neighbor from old parent and old path
                old_parent = z.parent
                if old_parent:
                    if z in old_parent.children:
                        old_parent.children.remove(z)
                    z.parent = None

                for path in self.paths:
                    if z in path.nodes:
                        path.nodes.remove(z)
                        break

                # Attach neighbor under new_node
                z.parent = nn
                nn.children.append(z)

                # Update the neighbor’s metrics
                z.cost = new_cost
                z.log_survival = new_log_survival
                z.p_fail = new_p_fail

                # Add neighbor into the same Path object as new_node
                z.path = nn.path
                nn.path.add_node(z)

                # Propagate down the subtree
                self.propagate_cost(z, grid, lc, edge_segments)
                self.rewire_counts += 1
            else:
                # Non-dominated but not strictly dominant = new node/branch
                new_z = Node(z.x, z.y, z.theta)
                new_z.parent = nn
                nn.children.append(new_z)
                new_z.cost = new_cost
                new_z.log_survival = new_log_survival
                new_z.p_fail = new_p_fail
                new_z.path = nn.path
                self.add_node(new_z, multiple_children=True) 
                new_z.is_additional_rewire = True  # rewire tracking
                # self.propagate_cost(new_z, grid, lc, edge_segments)
                self.rewire_counts += 1

    def propagate_cost(self, root, grid, lc=None, edge_segments=None):
        """
        Iteratively propagate cost & failure updates down the subtree.
        """
        queue = [root]
        new_path = root.path
        while queue:
            node = queue.pop(0)
            for child in node.children:

                new_cost = node.cost + distance_to(node, child)
                log_s_step = accumulate_log_survival(node, child, grid)
                new_log_survival = node.log_survival + log_s_step
                new_p_fail = 1 - np.exp(new_log_survival)

                
                child.cost = new_cost
                child.log_survival = new_log_survival
                child.p_fail = new_p_fail
                child.path = new_path
                queue.append(child)  # only propagate forward

                # # Draw the new edge with updated p_fail
                # if lc is not None and edge_segments is not None:
                #     update_progress_plot_3d(lc, edge_segments, node, child)

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
        self.is_additional_rewire = False  # Flag for additional rewire nodes

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
def PO_RRT_Star(start, goal, grid, failure_prob_values, max_iter=5000):
    # Initialize the tree and nodes
    start_node, goal_node = Node(*start), Node(*goal)
    tree = Tree(grid)
    tree.add_node(start_node)
    tree.start_node = start_node
    goal_node.is_goal = True
    start_node.is_start = True
    multiple_paths = []
    goal_tracker = set()

    fig, ax, lc, edge_segments = init_progress_plot_3d(
    start,
    goal,
    x_lim=(0, grid.width),
    y_lim=(0, grid.height),
    obstacles=grid.obstacles,
    )   

    
    for current_iter in range(max_iter):
        # Random node sample
        rand_node = Node(np.random.uniform(0, grid.width), np.random.uniform(0, grid.height), np.random.uniform(-np.pi, np.pi))
        if is_collision_free(rand_node, grid):  
            # Find the nearest node in the tree and steer to new node from it
            nearest_node = tree.nearest(rand_node)
            x, y, theta = steer(nearest_node, rand_node, DEFAULT_STEP_SIZE)
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
                        grid
                    )   
                    
                if len(new_nodes) > 1:
                    # multiple possible parents 
                    for nn in new_nodes:
                        multiple_children = True if len(nn.parent.children) > 1 else False
                        # 1) goal check per branch
                        if distance_to(nn, goal) <= DEFAULT_STEP_SIZE:
                            # if not any(child.is_goal for child in nn.children): # Safeguard against adding multiple goal nodes
                            # connect to goal exactly once per branch
                                goal_instance = Node(goal[0], goal[1], goal[2])
                                goal_instance.is_goal = True
                                if distance_to(nn, goal_instance) > DEFAULT_STEP_SIZE + 1e-3:
                                    print(f"❌ [goal connection] Goal jump too long: {distance_to(nn, goal_instance):.2f} "
                                                    f"from ({nn.x:.2f}, {nn.y:.2f}) to ({goal_instance.x:.2f}, {goal_instance.y:.2f})")
                                goal_instance.parent = nn
                                nn.children.append(goal_instance)
                                # if not nn.added_to_tree: # safeguard against adding the same node multiple times
                                # Make sure nn is in the tree first
                                tree.add_node(nn, multiple_children=multiple_children)
                                tree.rewire(tree.neighbors(nn), nn, grid, lc, edge_segments)


                                # ─── Cost and P_Fail for goal ─────────────────────────
                                goal_instance.cost = nn.cost + distance_to(nn, goal_instance)
                                log_s_step = accumulate_log_survival(nn, goal_instance, grid)
                                goal_instance.log_survival = nn.log_survival + log_s_step
                                goal_instance.p_fail = 1 - np.exp(goal_instance.log_survival)
                                
                                # Deduplication: check if this goal node is worse than one already added
                                key = (round(goal_instance.p_fail, 5), round(goal_instance.cost, 2))
                                if key in goal_tracker:
                                    continue  # Skip adding this goal node — dominated or duplicate
                                goal_tracker.add(key)

                                # Add goal_instance to tree
                                tree.add_node(goal_instance) 
                                print(f"Goal node added to tree with cost: {goal_instance.cost}, p_fail: {goal_instance.p_fail}")
                                # redraw_tree(tree, lc, edge_segments)
                        else:
                            tree.add_node(nn, multiple_children=multiple_children)
                            tree.rewire(tree.neighbors(nn), nn, grid, lc, edge_segments)
                            # update_progress_plot_3d(lc, edge_segments, nn.parent, nn)
                else:
                    # single child branch
                    for nn in new_nodes:
                        multiple_children = True if len(nn.parent.children) > 1 else False
                        if distance_to(nn, goal) <= DEFAULT_STEP_SIZE:
                            # if not any(child.is_goal for child in nn.children): # Safeguard against adding multiple goal nodes
                                # ─── goal handling ─────────────────────────
                                goal_instance = Node(goal[0], goal[1], goal[2])
                                goal_instance.is_goal = True
                                if distance_to(nn, goal_instance) > DEFAULT_STEP_SIZE + 1e-3:
                                    print(f"❌ [goal connection] Goal jump too long: {distance_to(nn, goal_instance):.2f} "
                                                    f"from ({nn.x:.2f}, {nn.y:.2f}) to ({goal_instance.x:.2f}, {goal_instance.y:.2f})")
                                goal_instance.parent = nn
                                nn.children.append(goal_instance)
                                # if not nn.added_to_tree: # safeguard against adding the same node multiple times
                                # Make sure nn is in the tree first
                               
                                tree.add_node(nn, multiple_children=multiple_children)
                                tree.rewire(tree.neighbors(nn), nn, grid, lc, edge_segments)


                                # ─── Cost and P_Fail for goal ─────────────────────────
                                goal_instance.cost = nn.cost + distance_to(nn, goal_instance)
                                log_s_step = accumulate_log_survival(nn, goal_instance, grid)
                                goal_instance.log_survival = nn.log_survival + log_s_step
                                goal_instance.p_fail = 1 - np.exp(goal_instance.log_survival)

                                # Deduplication: check if this goal node is worse than one already added
                                key = (round(goal_instance.p_fail, 5), round(goal_instance.cost, 2))
                                if key in goal_tracker:
                                    continue  # Skip adding this goal node — dominated or duplicate
                                goal_tracker.add(key)

                                # Add goal_instance to tree
                                tree.add_node(goal_instance) 
                                print(f"Goal node added to tree with cost: {goal_instance.cost}, p_fail: {goal_instance.p_fail}")

                                # redraw_tree(tree, lc, edge_segments)
                        # 2) Not near goal, so add the new node to the tree
                        else:
                            # Add the new node to the tree
                            # multiple_children = True if len(nn.parent.children) > 1 else False

                            tree.add_node(nn, multiple_children=multiple_children)
                            tree.rewire(tree.neighbors(nn), nn, grid, lc, edge_segments)
                            # update_progress_plot_3d(lc, edge_segments, nn.parent, nn)
                # redraw_tree(tree, lc, edge_segments)

    # 1. Collect all goal nodes in the tree
    goal_nodes = []
    for path in tree.paths:
        for node in path.nodes:
            if getattr(node, "is_goal", False):
                goal_nodes.append(node)

    # 2. Finalize unique root-to-goal paths
    seen_path_signatures = set()
    for g in goal_nodes:
        tree.finalize_path(g)  # will build and append new Path() object from g.parent chain

    # 3. Filter down to only complete, non-dominated paths
    # Extract multiple paths from the tree
    multiple_paths = [
                        {
                            "path": p,
                            "cost": p.cost,
                            "p_fail": p.p_fail
                        }
                        for p in tree.paths if p.is_complete
                    ]
    
    
    # Debugging output for paths
    # redraw_tree(tree, lc, edge_segments)

    MAX_ALLOWED_STEP = DEFAULT_STEP_SIZE + 1e-3  # Small epsilon

    print("\n--- Debug: Paths from start to goal ---")
    for idx, entry in enumerate(multiple_paths):
        path = entry["path"]
        nodes = path.nodes

        print(f"\nPath {idx+1} (Cost: {entry['cost']:.2f}, P_fail: {entry['p_fail']:.4f}):")
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i+1]
            step_dist = distance_to(a, b)

            # # DEBUG JUMP
            # if step_dist > MAX_ALLOWED_STEP:
            #     print(f"    ILLEGAL JUMP DETECTED between nodes {i} and {i+1}:")
            #     print(f"    From: (x={a.x:.2f}, y={a.y:.2f})")
            #     print(f"    To:   (x={b.x:.2f}, y={b.y:.2f})")
            #     print(f"    Distance: {step_dist:.2f} > allowed {MAX_ALLOWED_STEP:.2f}")
            #     print("Illegal jump in path — investigate tree structure or rewire logic.")

            # DEBUG P_FAIL MONOTONICITY
            if b.p_fail < a.p_fail - 1e-8:  # Allow for tiny floating point error
                print(f"    WARNING: p_fail decreased from parent to child at nodes {i}->{i+1}:")
                print(f"    Parent p_fail: {a.p_fail:.6f}, Child p_fail: {b.p_fail:.6f}")
            elif a.p_fail < 0 or b.p_fail < 0:
                print(f"    ERROR: Negative p_fail detected at node {i} or {i+1}.")

            # Normal print
            print(f"  (x={b.x:.2f}, y={b.y:.2f}, theta={b.theta:.2f}, cost={b.cost:.2f}, p_fail={b.p_fail:.4f})" +
                (" [GOAL]" if getattr(b, "is_goal", False) else ""))
            
    num_additional_in_tree = sum(1 for n in tree.node_list if getattr(n, "is_additional_rewire", False))
    print(f"\nAdditional rewire nodes currently in tree: {num_additional_in_tree}")
    # print("\n--- Debug: Paths from start to goal ---")
    # for idx, entry in enumerate(multiple_paths):
    #     path = entry["path"]
    #     if hasattr(path, "nodes"):
    #         nodes = path.nodes
    #     else:
    #         nodes = path
    #     print(f"Path {idx+1}:")
    #     for node in nodes:
    #         print(f"  (x={node.x:.2f}, y={node.y:.2f}, theta={node.theta:.2f}, cost={node.cost:.2f}, p_fail={node.p_fail:.4f})"
    #         + (" [START]" if getattr(node, "is_start", False) else "")
    #         + (" [GOAL]" if getattr(node, "is_goal", False) else ""))
    #     print(f"Path {idx+1} ends at: (x={path.nodes[-1].x}, y={path.nodes[-1].y}) path length={len(path)} is_goal={getattr(path.nodes[-1], 'is_goal', False)}")
    #     print("-" * 40)

    # After collecting multiple_paths, filter out dominated paths using pareto_dominates

    def pareto_filter(paths):
        non_dominated = []
        for i, entry_i in enumerate(paths):
            dominated = False
            for j, entry_j in enumerate(paths):
                if i != j:
                    if ((entry_j["cost"] <= entry_i["cost"] and
                        entry_j["p_fail"] < entry_i["p_fail"]) or
                        (entry_j["cost"] < entry_i["cost"] and
                        entry_j["p_fail"] <= entry_i["p_fail"])):
                        dominated = True
                        break
            if not dominated:
                non_dominated.append(entry_i)
        return non_dominated


    filtered_paths = pareto_filter(multiple_paths)
    # Print each unique path's cost and p_fail
    print("\nPre-Filtered unique paths (cost, p_fail):")
    for entry in filtered_paths:
        print(f"  cost={entry['cost']:.6f}, p_fail={entry['p_fail']:.8f}")

    # Remove duplicates by (cost, p_fail), keeping the first occurrence
    unique_filtered = []
    seen = set()
    for entry in filtered_paths:
        key = (round(entry["cost"], 6), round(entry["p_fail"], 8))  # Use rounding to avoid floating point issues
        if key not in seen:
            seen.add(key)
            unique_filtered.append(entry)

    # Print each unique path's cost and p_fail
    print("\nFiltered unique paths (cost, p_fail):")
    for entry in unique_filtered:
        print(f"  cost={entry['cost']:.6f}, p_fail={entry['p_fail']:.8f}")

    # Use unique_filtered as your filtered_paths from now on
    filtered_paths = unique_filtered

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
        {"type": "circular", "center": (10, 58), "radius": 8, "safe_dist": 5},
        {"type": "circular", "center": (30, 60), "radius": 10, "safe_dist": 5},
        {"type": "circular", "center": (60, 20), "radius": 18, "safe_dist": 5}
    ]

    
    # failure_prob_values = simpledialog.askstring("Input", "Enter the failure probabilities (comma-separated):")
    # if failure_prob_values:
    #     failure_prob_values = [float(x.strip()) for x in failure_prob_values.split(',')]
    # else:
    #     failure_prob_values = [0.1, 0.2, 0.3]  # Default values if none provided

    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

    filtered_paths, multiple_paths = PO_RRT_Star(start, goal, grid, failure_prob_values=[0.1])
    # plot_paths_metrics(multiple_paths)
    # plot_full_paths(multiple_paths)
    plot_paths_summary(filtered_paths, obstacles=obstacles)
    plot_paths_summary(multiple_paths, obstacles=obstacles)
    

if __name__ == '__main__':
    main()