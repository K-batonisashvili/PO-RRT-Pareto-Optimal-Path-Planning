# naive_rrt_baselines.py

import math
import numpy as np

from helper_functions import (
    distance_to,
    distance_sq,
    is_collision_free,
    is_edge_collision_free,
    steer,
    accumulate_log_survival,
)
from PO_RRT_Star import (
    Node,
    Grid,
    GRID_WIDTH,
    GRID_HEIGHT,
    DEFAULT_STEP_SIZE,
)


def _connection_radius(num_nodes, grid):
    n = max(num_nodes, 2)
    d = 2
    zeta_d = math.pi
    free_volume = float(grid.width * grid.height)

    gamma_rrt = 2.0 * ((1.0 + 1.0/d) ** (1.0/d)) * ((free_volume / zeta_d) ** (1.0/d))
    base_radius = gamma_rrt * ((math.log(n) / n) ** (1.0 / d))

    eta = DEFAULT_STEP_SIZE
    # return min(base_radius, eta)
    return 10
    



def _is_ancestor(candidate_ancestor, node):
    """
    Return True if candidate_ancestor is on the parent chain of node.
    Used to prevent cycles during rewiring.
    """
    cur = node
    while cur is not None:
        if cur is candidate_ancestor:
            return True
        cur = cur.parent
    return False


def _propagate_subtree_costs(root, grid, mode, w1, w2):
    """
    After rewiring, recompute (cost, log_survival, p_fail) for root
    and its descendants under the scalar mode.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        parent = node.parent
        if parent is None:
            # root – cost already set by caller
            pass
        else:
            d_edge = distance_to(parent, node)
            log_s_step = accumulate_log_survival(parent, node, grid, num_samples=3)
            node.cost = parent.cost + d_edge
            node.log_survival = parent.log_survival + log_s_step
            node.p_fail = 1 - np.exp(node.log_survival)

        # scalar objective (not strictly needed unless you want to store it)
        if mode == "distance":
            node.scalar = node.cost
        elif mode == "risk":
            node.scalar = node.p_fail
        elif mode == "weighted":
            node.scalar = w1 * node.cost + w2 * node.p_fail

        stack.extend(node.children)


def run_scalar_rrt(
    start,
    goal,
    grid,
    max_iter,
    mode="distance",
    w1=1.0,
    w2=1.0,
    rng_seed=None,
    sample_sequence=None,
):
    """
    Naive scalar RRT* baseline.

    Parameters
    ----------
    start, goal : (x, y)
    grid        : your Grid instance
    max_iter    : number of samples
    mode        : "distance", "risk", or "weighted"
    w1          : weight for cost in weighted mode
    w2          : weight for p_fail in weighted mode
    rng_seed    : optional seed for reproducibility

    Returns
    -------
    result : dict with keys:
        - "success": bool
        - "path": list[Node]  (from start to goal, including goal)
        - "cost": float       (total Euclidean distance)
        - "p_fail": float     (total probability of failure)
        - "scalar": float     (scalar objective value at goal)
        - "num_nodes": int
        - "mode": str
        - "w1": float
        - "w2": float
    """

    assert mode in ("distance", "risk", "weighted")

    # If we are *not* given explicit samples, fall back to our own RNG.
    if sample_sequence is None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = None

    start_node = Node(*start)
    start_node.is_start = True
    start_node.parent = None
    start_node.children = []
    start_node.cost = 0.0
    start_node.log_survival = 0.0
    start_node.p_fail = 0.0
    start_node.scalar = 0.0

    nodes = [start_node]
    goal_cfg = goal

    def nearest(node_like):
        # use squared distance to avoid sqrt in tight loops
        return min(nodes, key=lambda n: distance_sq(n, node_like))

    def neighbors(node_like, radius):
        r2 = radius * radius
        rx = radius
        # fast axis-aligned bounding box pre-filter to avoid many distance_sq calls
        res = []
        nx = node_like.x if hasattr(node_like, 'x') else node_like[0]
        ny = node_like.y if hasattr(node_like, 'y') else node_like[1]
        for n in nodes:
            if n is node_like:
                continue
            if abs(n.x - nx) > rx or abs(n.y - ny) > rx:
                continue
            if distance_sq(n, node_like) <= r2:
                res.append(n)
        return res

    def scalar_from(cost_val, p_fail_val):
        if mode == "distance":
            return cost_val
        elif mode == "risk":
            return p_fail_val
        elif mode == "weighted":
            return w1*cost_val + w2 * p_fail_val

    best_goal = None  # (scalar, cost, p_fail, connecting_node, goal_node)

     # ------------------ main loop ------------------ #
    for it in range(max_iter):
        # 1) sample random configuration
        if sample_sequence is not None:
            if it >= len(sample_sequence):
                break  # no more samples available
            sx, sy = sample_sequence[it]
            rand_node = Node(sx, sy)
        else:
            rand_node = Node(
                rng.uniform(0, grid.width),
                rng.uniform(0, grid.height),
            )

        if not is_collision_free(rand_node, grid):
            continue

        # 2) nearest and steer
        nn = nearest(rand_node)
        x, y = steer(nn, rand_node, DEFAULT_STEP_SIZE)
        new_node = Node(x, y)
        new_node.children = []
        if not is_collision_free(new_node, grid):
            continue

        # 3) select parent among neighbors (scalar RRT*)
        radius = _connection_radius(len(nodes), grid)
        znear = neighbors(new_node, radius)
        if not znear:
            znear = [nn]

        best_parent = None
        best_cost = None
        best_log_surv = None
        best_p_fail = None
        best_scalar = math.inf

        for z in znear:
            if not is_edge_collision_free(z, new_node, grid, num_samples=50, p_threshold=0.9):
                continue

            d_edge = distance_to(z, new_node)
            log_s_step = accumulate_log_survival(z, new_node, grid)
            new_cost = z.cost + d_edge
            new_log_surv = z.log_survival + log_s_step
            new_p_fail = 1 - np.exp(new_log_surv)
            new_scalar = scalar_from(new_cost, new_p_fail)

            if new_scalar < best_scalar:
                best_scalar = new_scalar
                best_parent = z
                best_cost = new_cost
                best_log_surv = new_log_surv
                best_p_fail = new_p_fail

        if best_parent is None:
            continue

        # attach new node
        new_node.parent = best_parent
        best_parent.children.append(new_node)
        new_node.cost = best_cost
        new_node.log_survival = best_log_surv
        new_node.p_fail = best_p_fail
        new_node.scalar = best_scalar
        nodes.append(new_node)

        # 4) rewiring
        for z in znear:
            if z is best_parent or z is new_node:
                continue
            if not is_edge_collision_free(new_node, z, grid, num_samples=10, p_threshold=0.9):
                continue

            d_edge = distance_to(new_node, z)
            log_s_step = accumulate_log_survival(new_node, z, grid, num_samples=3)
            cand_cost = new_node.cost + d_edge
            cand_log_surv = new_node.log_survival + log_s_step
            cand_p_fail = 1 - np.exp(cand_log_surv)

            cand_scalar = scalar_from(cand_cost, cand_p_fail)
            cur_scalar = scalar_from(z.cost, z.p_fail)

            # prevent cycles: don't rewire an ancestor to its descendant
            if cand_scalar + 1e-9 < cur_scalar and not _is_ancestor(z, new_node):
                old_parent = z.parent
                if old_parent is not None and z in old_parent.children:
                    old_parent.children.remove(z)
                z.parent = new_node
                new_node.children.append(z)
                # propagate metric changes down the subtree
                _propagate_subtree_costs(z, grid, mode, w1, w2)

        # 5) (optional) opportunistic connection to goal
        #    -> we only check scalar value at goal after the loop,
        #       but you can put an early success condition here if desired.

    # ------------------ post-run: connect to goal ------------------ #
    goal_node_template = Node(*goal_cfg)  # used only for cost/risk integration

    for z in nodes:
        # Only consider nodes reasonably near the goal to avoid crazy detours
        if distance_to(z, goal_cfg) > DEFAULT_STEP_SIZE:
            continue
        if not is_edge_collision_free(z, goal_node_template, grid, num_samples=10, p_threshold=0.9):
            continue

        d_edge = distance_to(z, goal_cfg)
        log_s_step = accumulate_log_survival(z, goal_node_template, grid, num_samples=3)
        total_cost = z.cost + d_edge
        total_log_surv = z.log_survival + log_s_step
        total_p_fail = 1 - np.exp(total_log_surv)
        total_scalar = scalar_from(total_cost, total_p_fail)

        if best_goal is None or total_scalar < best_goal[0]:
            # build a concrete goal node so we can output a full path
            goal_node = Node(*goal_cfg)
            goal_node.is_goal = True
            goal_node.parent = z
            z.children.append(goal_node)
            goal_node.cost = total_cost
            goal_node.log_survival = total_log_surv
            goal_node.p_fail = total_p_fail
            goal_node.scalar = total_scalar
            best_goal = (total_scalar, total_cost, total_p_fail, goal_node)

    if best_goal is None:
        # No feasible connection to goal
        return {
            "success": False,
            "path": [],
            "cost": math.inf,
            "p_fail": 1.0,
            "scalar": math.inf,
            "num_nodes": len(nodes),
            "mode": mode,
            "w1": w1,
            "w2": w2,
        }

    # Reconstruct path from start to goal
    _, best_cost, best_p_fail, best_goal_node = best_goal
    path_nodes = []
    cur = best_goal_node
    while cur is not None:
        path_nodes.append(cur)
        cur = cur.parent
    path_nodes.reverse() 
    
    return {
        "success": True,
        "path": path_nodes,
        "cost": best_cost,
        "p_fail": best_p_fail,
        "scalar": scalar_from(best_cost, best_p_fail),
        "num_nodes": len(nodes),
        "mode": mode,
        "w1": w1,
        "w2": w2,
        "nodes": nodes,       
    }
