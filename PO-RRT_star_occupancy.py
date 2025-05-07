import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
# from sklearn.linear_model import LinearRegression
from matplotlib.animation import FuncAnimation
import copy
import math
import logging
import tkinter as tk
from tkinter import simpledialog
from mpl_toolkits.mplot3d.art3d import Line3DCollection


logging.basicConfig(level=logging.INFO)


# Define constants
GRID_RESOLUTION = 0.05
X_MIN, X_MAX = 0.0, 4.0
Y_MIN, Y_MAX = 0.0, 4.0
GRID_WIDTH = int((X_MAX - X_MIN) / GRID_RESOLUTION)
GRID_HEIGHT = int((Y_MAX - Y_MIN) / GRID_RESOLUTION)
MAX_STEERING_ANGLE = np.pi / 3
PARETO_RADIUS = 0.25
DEFAULT_STEP_SIZE = 0.1
GOAL_SAMPLE_RATE = 0.1
PROBABILITY_THRESHOLD = 0.01

# Define the unknown area bounds and obstacles
unknown_area_bounds = ((1.5, 2.5), (0.3, 1.7))
obstacle_centers = [(2.0, 4), (2.0, 2), (2.0, 0)]
obstacle_radius = 0.4

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
    x_idx, y_idx = int((node.x - X_MIN) / GRID_RESOLUTION), int((node.y - Y_MIN) / GRID_RESOLUTION)
    return 0 <= x_idx < GRID_WIDTH and 0 <= y_idx < GRID_HEIGHT and grid[x_idx, y_idx] < 0.3 # 0.3 threshold for collision

def steer(from_node, to_node, step_size):
    """
    Steer from one node to another with a given step size.
    """
    dx, dy = to_node.x - from_node.x, to_node.y - from_node.y
    theta = np.arctan2(dy, dx)  # Calculate the angle directly
    return Node(from_node.x + step_size * np.cos(theta), from_node.y + step_size * np.sin(theta), theta)


# ----------------------- #
#       Main Classes      #
# ----------------------- #
class Tree:
    """
    Main RRT Tree represented as a class.
    """


    def __init__(self, root_node, grid):
        """
        Initialize the tree with a root node and a root path.
        """
        self.paths = [Path(root_node)]  # Start with a single path containing the root node
        self.rewire_counts = 0
        self.rewire_neighbors_count = 0
        self.grid = grid

    def add_node(self, node, parent=None):
        """
        Add a node to the tree. If the node has a parent, add it to the parent's path.
        Otherwise, create a new path for the node.
        """
        if node.x == 3.5 and node.y == 0.5:
            print("Goal node added to the tree.")
        if parent:
            # Find the path containing the parent and add the node to it
            for path in self.paths:
                if parent in path.nodes:
                    path.add_node(node)
                    node.added_to_tree = True
                    break
        else:
            # Create a new path for the node
            new_path = Path(node)
            self.paths.append(new_path)

    def nearest(self, rand_node):
        """
        Find the nearest node in the tree to a random node, excluding the goal node.
        """
        all_nodes = [node for path in self.paths for node in path.nodes]
        return min(all_nodes, key=lambda n: distance_to(n, rand_node))
    
    def rewire(self, znear, new_node, grid):
        """
        Rewire the tree to optimize paths based on cost and failure probability.
        """
        for neighbor in znear:
            #compute what cost + p_fail would be if we attached via new_node
            xi = int((neighbor.x - X_MIN)/GRID_RESOLUTION)
            yi = int((neighbor.y - Y_MIN)/GRID_RESOLUTION)
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
            new_p_fail     = np.exp(new_log_survival)

            # only rewire if (new_cost,new_p_fail) Pareto-dominates (old_cost,old_p_fail)
            if self.pareto_dominates(new_cost, new_p_fail, neighbor.cost, neighbor.p_fail):
                # detach neighbor from old parent and old path
                old_parent = neighbor.parent
                if old_parent and neighbor in old_parent.children:
                    old_parent.children.remove(neighbor)

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

                # propagate down the subtree
                self.propagate_cost(neighbor, grid)
                self.rewire_counts += 1

    def pareto_dominates(self, cost1, fail1, cost2, fail2):
        """
        Check if node1 dominates node2 in terms of cost and failure probability.
        """
        return (cost1 < cost2 and fail1 < fail2) or (cost1 <= cost2 and fail1 < fail2) or (cost1 < cost2 and fail1 <= fail2)

    def propagate_cost(self, root, grid):
        """
        Iteratively propagate cost & failure updates down the
        subtree of .children under `root`, avoiding recursion.
        """
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in node.children.copy():    
                # 1) cost
                child.cost = node.cost + distance_to(node, child)

                # 2) risk in child's cell
                xi = int((child.x - X_MIN) / GRID_RESOLUTION)
                yi = int((child.y - Y_MIN) / GRID_RESOLUTION)
                raw_p = grid[xi, yi]
                if raw_p <= 0.0:
                    log_s_step = 0.0
                else:
                    p_clip     = np.clip(raw_p, 0.0, 1.0)
                    log_s_step = np.log(1 - p_clip)

                # 3) accumulate correctly
                child.log_survival = node.log_survival + log_s_step
                child.p_fail       = 1 - np.exp(child.log_survival)

                # 4) guard against stray cycles: only follow actual parent→child links
                if child.parent is node:
                    queue.append(child)

    def neighbors(self, node):
        """
        Find all nodes within a static PARETO_RADIUS around the given node.
        """
        neighbors = []
        raw = []
        x_min, x_max = node.x - PARETO_RADIUS, node.x + PARETO_RADIUS
        y_min, y_max = node.y - PARETO_RADIUS, node.y + PARETO_RADIUS

        for path in self.paths:
            for n in path.nodes:
                if n is not node and x_min <= n.x <= x_max and y_min <= n.y <= y_max:
                    if distance_to(n, node) < PARETO_RADIUS:
                        neighbors.append(n)

        return neighbors

    def choose_parents(self, znear, x, y, theta, grid):
        """
        Instead of picking a single best parent, return a list of new Node()s—
        one for *each* neighbor in Znear that yields a Pareto‐optimal (cost, p_fail)
        pair at the same (x,y,theta).
        """
        # 1) gather all candidate (parent, cost, log_survival, p_fail)
        candidates = []
        # step‐failure from grid cell
        xi = int((x - X_MIN)/GRID_RESOLUTION)
        yi = int((y - Y_MIN)/GRID_RESOLUTION)
        raw_p = grid[xi, yi]
        if raw_p <= 0.0:
            # no risk here → no change to log-survival
            log_s_step = 0.0
        else:
            # accumulate only in genuinely risky cells
            p_clip     = np.clip(raw_p, 0.0, 1.0)
            log_s_step = np.log(1 - p_clip)

        for neighbor in znear:
            cost   = neighbor.cost + distance_to(neighbor, Node(x,y,theta))
            log_survival = neighbor.log_survival + log_s_step
            prob_failure  = 1 - np.exp(log_survival)
            candidates.append((neighbor, cost, log_survival, prob_failure))
            if prob_failure > 0.90:
                flag = True


        # 2) filter out dominated candidates
        pareto = []
        for pa in candidates:
            dominated = False
            for pb in candidates:
                if (pb is not pa) and self.pareto_dominates(pb[1], pb[3], pa[1], pa[3]):
                    dominated = True
                    break
            if not dominated:
                pareto.append(pa) # This ensures we do not lose any Pareto‐optimal candidates, even if they are not unique.

        # 3) build one new Node per Pareto‐optimal parent
        # if pareto:
        #     best = min(pareto, key=lambda t: t[3])   # minimal pfail
        #     pareto = [best]

        # --- 4) Build exactly one new Node per remaining parent ---
        pareto_nodes = []
        for parent, cost, log_surv, p_fail in pareto:
            node = Node(x, y, theta)
            node.parent         = parent
            parent.children.append(node)
            node.cost           = cost
            node.log_survival   = log_surv
            node.p_fail         = p_fail
            pareto_nodes.append(node)

        return pareto_nodes

    def get_path_to(self, goal_node):
        """
        Get the path from the start node to the goal node.
        """
        for path in self.paths:
            if goal_node in path.nodes:
                return path
        return None

    def plot_progress(self, start, goal, ax, path=[]):
        """
        Plot RRT* progress.
        """
        ax.clear()
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.plot(start[0], start[1], 'go')  # Start point
        ax.plot(goal[0], goal[1], 'bo')   # Goal point

        # Plot obstacles
        for center in obstacle_centers:
            circle = plt.Circle(center, obstacle_radius, color='red', alpha=0.5, label='Obstacle')
            ax.add_patch(circle)

        # Plot unknown area
        x_bounds = [unknown_area_bounds[0][0], unknown_area_bounds[0][1], unknown_area_bounds[0][1], unknown_area_bounds[0][0], unknown_area_bounds[0][0]]
        y_bounds = [unknown_area_bounds[1][0], unknown_area_bounds[1][0], unknown_area_bounds[1][1], unknown_area_bounds[1][1], unknown_area_bounds[1][0]]
        ax.fill(x_bounds, y_bounds, color='orange', alpha=0.3, label='Unknown Area')

        # Plot tree edges
        for path in self.paths:
            for node in path.nodes:
                if node.parent:
                    ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 'g-')

        # Highlight the path
        for node in path:
            if node.parent:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y], 'r-', linewidth=2)

        plt.draw()
        # plt.pause(0.01)

    def plot_progress_3d(self, start, goal, ax, complete_path=[]):
        """
        Plot RRT* progress in 3D. Z-axis represents p_fail.
        """
        ax.clear()
        ax.set_xlim(X_MIN, X_MAX)
        ax.set_ylim(Y_MIN, Y_MAX)
        ax.set_zlim(0, 1)  # p_fail ranges from 0 to 1
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('p_fail')
        ax.set_title('RRT* Progress in 3D')

        # Plot start and goal
        ax.scatter(start[0], start[1], 0, c='green', label='Start', s=50)
        ax.scatter(goal[0], goal[1], 0, c='blue', label='Goal', s=50)

        # Plot circular obstacles
        for center in obstacle_centers:
            cx, cy = center
            theta = np.linspace(0, 2 * np.pi, 100)  # Angle for the circle
            x = cx + obstacle_radius * np.cos(theta)  # X-coordinates of the circle
            y = cy + obstacle_radius * np.sin(theta)  # Y-coordinates of the circle
            z = np.zeros_like(x)  # Z-coordinates (flat on the ground)
            ax.plot(x, y, z, color='red', alpha=0.7, label='Obstacle')

        # Plot unknown area
        x_bounds = [unknown_area_bounds[0][0], unknown_area_bounds[0][1], unknown_area_bounds[0][1], unknown_area_bounds[0][0], unknown_area_bounds[0][0]]
        y_bounds = [unknown_area_bounds[1][0], unknown_area_bounds[1][0], unknown_area_bounds[1][1], unknown_area_bounds[1][1], unknown_area_bounds[1][0]]
        ax.plot_trisurf(
            x_bounds,
            y_bounds,
            np.zeros(len(x_bounds)),
            color='orange',
            alpha=0.3
        )

        # Plot tree edges and nodes
        for path in self.paths:
            for node in path.nodes:
                if node.parent:
                    # Draw the edge between the node and its parent
                    ax.plot(
                        [node.x, node.parent.x],
                        [node.y, node.parent.y],
                        [node.p_fail, node.parent.p_fail],
                        'g-', alpha=0.7
                    )
                # Draw the node itself
                # ax.scatter(node.x, node.y, node.p_fail, c='black', s=10)

        # Highlight complete path in red
        for node in complete_path:
            if node.parent:
                ax.plot(
                    [node.x, node.parent.x],
                    [node.y, node.parent.y],
                    [node.p_fail, node.parent.p_fail],
                    'r-', linewidth=2, alpha=0.7
                )


        ax.legend()
        plt.draw()
        plt.pause(0.001)


class Path:
    def __init__(self, root_node=None):
        self.nodes = [root_node] if root_node else []

    def add_node(self, node):
        self.nodes.append(node)

    @property
    def cost(self) -> float:
        # cost‐to‐come of the last node
        return self.nodes[-1].cost if self.nodes else 0.0

    @property
    def p_fail(self) -> float:
        # failure probability at the last node
        return self.nodes[-1].p_fail if self.nodes else 1.0

    def __repr__(self):
        return f"Path(len={len(self.nodes)}, cost={self.cost:.2f}, p_fail={self.p_fail:.2f})"


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

class Grid:
    """
    Class representing the occupancy grid.
    """
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))

    def add_circular_obstacle(self, center, radius, safe_dist):
        """
        Adding a circular obstacle to the grid.
        """
        cx, cy = int((center[0] - X_MIN) / GRID_RESOLUTION), int((center[1] - Y_MIN) / GRID_RESOLUTION)
        rad_cells, safe_cells = int(radius / GRID_RESOLUTION), int(safe_dist / GRID_RESOLUTION)

        for x in range(cx - rad_cells - safe_cells, cx + rad_cells + safe_cells + 1):
            for y in range(cy - rad_cells - safe_cells, cy + rad_cells + safe_cells + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                    if dist <= rad_cells:
                        self.grid[x][y] = 0.9
                    elif dist <= rad_cells + safe_cells:
                        self.grid[x][y] = 0.9 * (1 - (dist - rad_cells) / safe_cells)

    def add_unknown_area(self, x_range, y_range, probability):
        """
        Adding a square unknown area to the grid.
        """
        x_start, x_end = int((x_range[0] - X_MIN) / GRID_RESOLUTION), int((x_range[1] - X_MIN) / GRID_RESOLUTION)
        y_start, y_end = int((y_range[0] - Y_MIN) / GRID_RESOLUTION), int((y_range[1] - Y_MIN) / GRID_RESOLUTION)
        self.grid[x_start:x_end+1, y_start:y_end+1] = probability

###############################
## CENTRAL RRT STAR FUNCTION ##
###############################
def rrt_star(start, goal, grid, failure_prob_values, max_iter=5000, step_size=DEFAULT_STEP_SIZE, threshold=PROBABILITY_THRESHOLD, goal_sample_rate=GOAL_SAMPLE_RATE):
    # Initialize the tree and nodes
    start_node, goal_node = Node(*start), Node(*goal)
    tree = Tree(start_node, grid)
    multiple_paths = []
    # goal_clone = None

    fig, ax, lc, edge_segments = init_progress_plot_3d(
    start=(start_node.x, start_node.y),
    goal=(goal_node.x,  goal_node.y),
    x_lim=(X_MIN, X_MAX),
    y_lim=(Y_MIN, Y_MAX)
    )   

    
    for current_iter in range(max_iter):
        # Random node sample or goal-biased sample
        # if np.random.rand() > goal_sample_rate:
        rand_node = Node(np.random.uniform(X_MIN, X_MAX), np.random.uniform(Y_MIN, Y_MAX), np.random.uniform(-np.pi, np.pi))
        # else:
        #     rand_node = Node(goal[0], goal[1], np.random.uniform(-np.pi, np.pi))
        
        if is_collision_free(rand_node, grid.grid):  
            # Find the nearest node in the tree and steer to new node from it
            nearest_node = tree.nearest(rand_node)
            new_node = steer(nearest_node, rand_node, step_size)
            if is_collision_free(new_node, grid.grid):
                # Check if the new node is collision-free              
                znear = tree.neighbors(new_node)
                
                # Check if znear contains any nodes, if it doesn't, set the nearest node as znear
                if not znear:
                    znear = [nearest_node]

                # Exclude the goal node from znear to prevent rewiring through it
                znear = [n for n in znear if n is not goal_node]

                new_nodes = tree.choose_parents(
                        znear, 
                        new_node.x, 
                        new_node.y, 
                        new_node.theta, 
                        grid.grid
                    )
                for nn in new_nodes:
                    # If the new node is close enough to the goal, connect it directly to the goal
                    if distance_to(nn, goal) < step_size:
                        if multiple_paths:
                            tree.rewire(tree.neighbors(nn), nn, grid.grid)
                        # goal_clone = Node(goal[0], goal[1], nn.theta)
                        goal_node.parent = nn.parent
                        nn.children.append(goal_node)

                        # Update cost and failure metrics for the goal node
                        goal_node.cost = nn.cost + distance_to(nn, goal_node)
                        goal_node.log_survival = nn.log_survival
                        goal_node.p_fail = 1 - np.exp(goal_node.log_survival)
                        
                        tree.add_node(goal_node, parent=nn.parent)
                        raw_path = tree.get_path_to(goal_node).nodes[:]  # Shallow copy of the node list
                        path_cost = goal_node.cost
                        path_pfail = goal_node.p_fail
                        print(f"Path found with cost: {path_cost:.2f}, p_fail: {path_pfail:.2f}")
                        update_progress_plot_3d(lc, edge_segments, nn.parent, goal_node)

                        multiple_paths.append((raw_path, path_cost, path_pfail))
                    else:
                        # Add each distinct Node to the tree
                        tree.add_node(nn, parent=nn.parent)
                        tree.rewire(tree.neighbors(nn), nn, grid.grid)
                        parent = nn.parent
                        update_progress_plot_3d(lc, edge_segments, parent, nn)

    return multiple_paths 

def init_progress_plot_3d(start, goal, x_lim, y_lim, z_lim=(0.0,1.0)):
    """
    Initialize a 3D progress plot for RRT*.

    Returns: (fig, ax, lc, edge_segments)
      - fig: matplotlib Figure
      - ax:  Axes3D
      - lc:  Line3DCollection for edges
      - edge_segments: list that will hold all edge segments
    """
    fig = plt.figure()
    ax  = fig.add_subplot(projection='3d')

    # Static axes limits & labels
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_zlim(*z_lim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Probability of Failure')

    # Plot start & goal
    ax.scatter(start[0], start[1], 0.0, c='green', s=50, label='Start')
    ax.scatter(goal[0],  goal[1],  0.0, c='blue',  s=50, label='Goal')
    ax.legend()

    for center in obstacle_centers:
            cx, cy = center
            theta = np.linspace(0, 2 * np.pi, 100)  # Angle for the circle
            x = cx + obstacle_radius * np.cos(theta)  # X-coordinates of the circle
            y = cy + obstacle_radius * np.sin(theta)  # Y-coordinates of the circle
            z = np.zeros_like(x)  # Z-coordinates (flat on the ground)
            ax.plot(x, y, z, color='red', alpha=0.7, label='Obstacle')

        # Plot unknown area
    x_bounds = [unknown_area_bounds[0][0], unknown_area_bounds[0][1], unknown_area_bounds[0][1], unknown_area_bounds[0][0], unknown_area_bounds[0][0]]
    y_bounds = [unknown_area_bounds[1][0], unknown_area_bounds[1][0], unknown_area_bounds[1][1], unknown_area_bounds[1][1], unknown_area_bounds[1][0]]
    ax.plot_trisurf(
        x_bounds,
        y_bounds,
        np.zeros(len(x_bounds)),
        color='orange',
        alpha=0.3
    )

    # Prepare empty Line3DCollection for edges
    edge_segments = []
    lc = Line3DCollection(edge_segments, linewidths=1.5, alpha=0.7, color='C0')
    ax.add_collection(lc)

    plt.ion()  # turn on interactive mode
    plt.show()

    return fig, ax, lc, edge_segments

def update_progress_plot_3d(lc, edge_segments, parent_node, new_node, pause_time=0.001):
    """
    Append the new edge from parent_node to new_node, update the Line3DCollection,
    and redraw the figure.

    Arguments:
      - lc: the Line3DCollection returned by init_progress_plot_3d
      - edge_segments: the list returned by init_progress_plot_3d
      - parent_node, new_node: Node objects with .x, .y, .p_fail
      - pause_time: how long to pause (seconds) after redraw
    """
    # Add the new segment
    edge_segments.append([
        (parent_node.x,  parent_node.y,  parent_node.p_fail),
        (new_node.x,     new_node.y,     new_node.p_fail)
    ])

    # Bulk‐update and redraw
    lc.set_segments(edge_segments)
    plt.draw()
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

# Main code

def main():
    
    # Create the main application window
    root = tk.Tk()
    root.withdraw()  # Hide the root window

    failure_prob_values = simpledialog.askstring("Input", "Enter the failure probabilities (comma-separated):")
    if failure_prob_values:
        failure_prob_values = [float(x.strip()) for x in failure_prob_values.split(',')]
    else:
        failure_prob_values = [0.1, 0.2, 0.3]  # Default values if none provided

    start, goal = (0.3, 2, 0), (3.5, 0.5, 0)
    grid = Grid(GRID_WIDTH, GRID_HEIGHT)
    for center in obstacle_centers:
        grid.add_circular_obstacle(center, obstacle_radius, 0.2)
    grid.add_unknown_area(unknown_area_bounds[0], unknown_area_bounds[1], 0.05)

    multiple_paths = rrt_star(start, goal, grid, failure_prob_values)
    plot_paths_metrics(multiple_paths)

    

if __name__ == '__main__':
    main()