import time
import numpy as np
import logging
import tkinter as tk
import math
from tkinter import simpledialog

import matplotlib.pyplot as plt
from visualization import init_progress_plot_3d, init_progress_plot_2d, _occupancy_color


from helper_functions import (
    is_edge_collision_free, distance_to, steer, 
    accumulate_log_survival
)
from visualization import (
    plot_paths_summary, interactive_spectral_cluster_plot
)
import json

logging.basicConfig(level=logging.INFO)

# define constants
GRID_WIDTH = 100
GRID_HEIGHT = 100
DEFAULT_STEP_SIZE = 10
COLLISION_SAMPLES = 20   
RISK_SAMPLES = 10        

# quick export functions
def _pack_paths_for_json(paths):
    out = []
    for entry in paths:
        p = entry["path"]
        nodes = p.nodes
        out.append({
            "cost": float(entry["cost"]),
            "p_fail": float(entry["p_fail"]),
            "nodes": [
                {
                    "x": float(n.x),
                    "y": float(n.y),
                    "cost": float(getattr(n, "cost", 0.0)),
                    "p_fail": float(getattr(n, "p_fail", 0.0)),
                } for n in nodes
            ],
        })
    return out

def save_run_json(filename, start, goal, grid, filtered_paths, multiple_paths, edge_segments):
    def _as_seq(x): return x.tolist() if isinstance(x, np.ndarray) else x
    def _norm_point(pt):
        pt = _as_seq(pt)
        if len(pt) == 2: return [float(pt[0]), float(pt[1]), 0.0]
        elif len(pt) == 3: return [float(pt[0]), float(pt[1]), float(pt[2])]
        return None

    edges = []
    if edge_segments:
        for item in edge_segments:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                p1, p2 = _norm_point(item[0]), _norm_point(item[1])
                if p1 and p2: edges.append([p1, p2])

    payload = {
        "meta": {
            "saved_at_unix": time.time(),
            "start": list(start),
            "goal": list(goal),
            "grid_size": [int(grid.width), int(grid.height)],
        },
        "obstacles": getattr(grid, "obstacles", []),
        "paths": {
            "filtered": _pack_paths_for_json(filtered_paths),
            "multiple": _pack_paths_for_json(multiple_paths),
        },
        "tree": {"edges": edges},
    }
    with open(filename, "w") as f:
        json.dump(payload, f)
    print(f"[OK] exported -> {filename}")

def draw_array_tree_3d_live(tree, lc):
    # translates array-based lineage into 3D Matplotlib segments
    segments = []
    for node in tree.node_list:
        for i, (p_node, p_set_id) in enumerate(node.lineage):
            if p_node is None: continue
            
            # extract current node z-axis (p_fail)
            L = node.costs[i, 1]
            p_fail = 1.0 - np.exp(L)
            
            # extract parent node z-axis
            p_idx = np.where(p_node.set_ids == p_set_id)[0]
            if len(p_idx) == 0: continue
            p_L = p_node.costs[p_idx[0], 1]
            p_parent_fail = 1.0 - np.exp(p_L)
            
            segments.append([
                (p_node.x, p_node.y, p_parent_fail),
                (node.x, node.y, p_fail)
            ])
            
    lc.set_segments(segments)
    lc.set_color(['gray'] * len(segments))
    plt.pause(0.001)

def draw_array_tree_2d_live(tree, lc):
    # flattens array lineage into 2d spatial segments
    segments = []
    for node in tree.node_list:
        for p_node, _ in node.lineage:
            if p_node is not None:
                segments.append([(p_node.x, p_node.y), (node.x, node.y)])
                
    lc.set_segments(segments)
    lc.set_color(['gray'] * len(segments))
    plt.pause(0.001)

def render_2d_array_tree(tree, grid, filtered_paths, start, goal):
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, grid.width)
    ax.set_ylim(0, grid.height)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("2D PORRT* Tree (Array Architecture)")
    ax.grid(True, alpha=0.3)

    # Draw Obstacles
    if grid.obstacles:
        for obs in grid.obstacles:
            if obs["type"] == "circular":
                cx, cy = obs["center"]
                r = obs["radius"]
                theta = np.linspace(0, 2*np.pi, 100)
                prob = obs.get("probability", 1.0)
                ax.plot(cx + r*np.cos(theta), cy + r*np.sin(theta), '--', color=_occupancy_color(prob))
            elif obs["type"] == "rectangular":
                x0, x1 = obs["x_range"]
                y0, y1 = obs["y_range"]
                prob = obs.get("probability", 0.05)
                ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], '--', color=_occupancy_color(prob))

    # Draw Array Tree Edges
    for node in tree.node_list:
        for pn, _ in node.lineage:
            if pn is not None:
                ax.plot([pn.x, node.x], [pn.y, node.y], color='gray', alpha=0.4, lw=0.5)

    # Overlay Filtered Paths
    cmap = cm.get_cmap('tab10', max(1, len(filtered_paths)))
    for idx, entry in enumerate(filtered_paths):
        nodes = entry["path"].nodes
        xs = [n.x for n in nodes]
        ys = [n.y for n in nodes]
        ax.plot(xs, ys, color=cmap(idx), lw=2.5, label=f'Path {idx+1}')

    ax.scatter(start[0], start[1], c='green', s=80, zorder=5, label='Start')
    ax.scatter(goal[0], goal[1], c='red', marker='*', s=120, zorder=5, label='Goal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show(block=True)


def render_3d_array_tree(tree, grid, filtered_paths, start, goal):
    fig, ax, lc, _ = init_progress_plot_3d(
        start, goal, (0, grid.width), (0, grid.height), grid.obstacles
    )
    
    segments = []
    colors = []
    linewidths = []

    # Draw Array Tree Branches
    for node in tree.node_list:
        for i, (p_node, p_set_id) in enumerate(node.lineage):
            if p_node is None: continue
            
            p_fail = 1.0 - np.exp(node.costs[i, 1])
            p_idx = np.where(p_node.set_ids == p_set_id)[0]
            if len(p_idx) == 0: continue
            
            p_parent_fail = 1.0 - np.exp(p_node.costs[p_idx[0], 1])
            
            segments.append([(p_node.x, p_node.y, p_parent_fail), (node.x, node.y, p_fail)])
            colors.append('gray')
            linewidths.append(0.8)

    # Highlight Filtered Paths
    cmap = cm.get_cmap('tab10', max(1, len(filtered_paths)))
    for idx, entry in enumerate(filtered_paths):
        nodes = entry["path"].nodes
        for i in range(1, len(nodes)):
            n1, n2 = nodes[i-1], nodes[i]
            segments.append([(n1.x, n1.y, n1.p_fail), (n2.x, n2.y, n2.p_fail)])
            colors.append(cmap(idx))
            linewidths.append(2.5)

    lc.set_segments(segments)
    lc.set_color(colors)
    lc.set_linewidths(linewidths)
    plt.show(block=True)

# ----------------------- #
#       Main Classes      #
# ----------------------- #




######################################################
#####                                           ###### 
#####               PATH CLASS                  ######
#####                                           ######
######################################################
class Path:
    def __init__(self):
        self.nodes = []
    @property
    def cost(self) -> float:
        return self.nodes[-1].cost if self.nodes else 0.0
    @property
    def p_fail(self) -> float:
        return self.nodes[-1].p_fail if self.nodes else 1.0

class MockNode:
    # proxy for visualization compat
    def __init__(self, x, y, cost, p_fail):
        self.x = x
        self.y = y
        self.cost = cost
        self.p_fail = p_fail




######################################################
#####                                           ###### 
#####               NODE CLASS                  ######
#####                                           ######
######################################################
class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        
        # nx2 array -> [cost, log_surv]
        self.costs = np.empty((0, 2), dtype=float)
        # track uuid for lineage mapping
        self.set_ids = np.empty(0, dtype=int)
        self.next_id = 0
        # maps local row -> (parent_node, parent_set_id)
        self.lineage = []
        # forward spatial pointers
        self.children = []
        self.child_edges = {}
        
        # lazy eval triggers
        self.dirty = False
        self.pending_updates = []
        self.is_goal = False 
        self.is_start = False 

    def queue_update(self, update_dict):
        # queues lazy tasks
        self.pending_updates.append(update_dict)
        self.dirty = True

class Grid:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.obstacles = obstacles
        for obstacle in obstacles:
            if obstacle["type"] == "circular":
                self.add_circular_obstacle(obstacle["center"], obstacle["radius"], obstacle["safe_dist"])
            elif obstacle["type"] == "rectangular":
                self.add_unknown_area(obstacle["x_range"], obstacle["y_range"], obstacle["probability"])
                
    def add_circular_obstacle(self, center, radius, safe_dist):
        cx = int(center[0] / self.width * (self.width - 1))
        cy = int(center[1] / self.height * (self.height - 1))
        rad_cells = int(radius / self.width * (self.width - 1))
        safe_cells = int(safe_dist / self.width * (self.width - 1))
        for x in range(cx - rad_cells - safe_cells, cx + rad_cells + safe_cells + 1):
            for y in range(cy - rad_cells - safe_cells, cy + rad_cells + safe_cells + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
                    if dist <= rad_cells:
                        new_prob = 0.99
                    elif dist <= rad_cells + safe_cells:
                        new_prob = 0.99 * (1 - (dist - rad_cells) / safe_cells)
                    else:
                        new_prob = 0.0
                    self.grid[x][y] = max(self.grid[x][y], new_prob)
                    
    def add_unknown_area(self, x_range, y_range, probability):
        x_start = int(x_range[0] / self.width * (self.width - 1))
        x_end = int(x_range[1] / self.width * (self.width - 1))
        y_start = int(y_range[0] / self.height * (self.height - 1))
        y_end = int(y_range[1] / self.height * (self.height - 1))
        for x in range(x_start, x_end + 1):
            for y in range(y_start, y_end + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    self.grid[x][y] = max(self.grid[x][y], probability)




######################################################
#####                                           ###### 
#####               TREE CLASS                  ######
#####                                           ######
######################################################
class Tree:
    def __init__(self, grid, p_fail_threshold=1.0):
        self.node_list = []
        self.grid = grid
        self.start_node = None
        self.p_fail_threshold = p_fail_threshold

    def add_node(self, node):
        if node not in self.node_list:
            self.node_list.append(node)
    
    def connection_radius(self):
        # standard dynamic search radius
        n = max(len(self.node_list), 2)
        d = 2
        zeta_d = math.pi
        free_vol = float(self.grid.width * self.grid.height)
        gamma = 2.0 * ((1.0 + 1.0/d)**(1.0/d)) * ((free_vol/zeta_d)**(1.0/d))
        base_radius = gamma * ((math.log(n) / n)**(1.0/d))
        return min(base_radius, 10.0)

    def nearest(self, node):
        nx, ny = node.x, node.y
        best_node = None
        best_dist_sq = float('inf')
        for n in self.node_list:
            d_sq = (n.x - nx)**2 + (n.y - ny)**2
            if d_sq < best_dist_sq:
                best_dist_sq = d_sq
                best_node = n
        return best_node

    def neighbors(self, node):
        r = self.connection_radius()
        r2 = r * r
        nx, ny = node.x, node.y
        res = []
        for n in self.node_list:
            if n is node: continue
            if abs(n.x - nx) > r or abs(n.y - ny) > r: continue
            if (n.x - nx)**2 + (n.y - ny)**2 <= r2:
                res.append(n)
        return res

    def is_ancestor(self, candidate, node):
        # traverses lineage pointers back to root
        queue = [node]
        visited = set()
        while queue:
            curr = queue.pop(0)
            if curr in visited: continue
            visited.add(curr)
            if curr is candidate: return True
            for p_node, _ in curr.lineage:
                if p_node: queue.append(p_node)
        return False
   
    def pareto_filter_vectors(self, costs):
        # vectorized dominance check
        n = costs.shape[0]
        mask = np.ones(n, dtype=bool)
        
        # Bypass math entirely for single-trajectory nodes
        if n <= 1: 
            return mask
            
        for i in range(n):
            if not mask[i]: continue
            
            c_lesseq = costs[i, 0] <= costs[:, 0]
            l_greateq = costs[i, 1] >= costs[:, 1]
            c_less = costs[i, 0] < costs[:, 0]
            l_great = costs[i, 1] > costs[:, 1]
            
            dom_by_i = (c_lesseq & l_great) | (c_less & l_greateq)
            eps_dup = (np.abs(costs[i, 0] - costs[:, 0]) < 0.1) & \
                      (np.abs(costs[i, 1] - costs[:, 1]) < 0.005)
            
            kill_mask = dom_by_i | eps_dup
            kill_mask[i] = False 
            mask[kill_mask] = False
            
            c_lesseq_inv = costs[:, 0] <= costs[i, 0]
            l_greateq_inv = costs[:, 1] >= costs[i, 1]
            c_less_inv = costs[:, 0] < costs[i, 0]
            l_great_inv = costs[:, 1] > costs[i, 1]
            
            dom_i = (c_lesseq_inv & l_great_inv) | (c_less_inv & l_greateq_inv)
            if np.any(dom_i & mask):
                mask[i] = False
        return mask
    
    def choose_parents(self, znear, new_node, grid):
        # evaluates neighbors to find optimal pareto parents for new_node
        candidates = []
        physics_cache = {}
        
        for z in znear:
            if getattr(z, 'is_goal', False): continue
            
            if is_edge_collision_free(z, new_node, grid, COLLISION_SAMPLES, 0.9):
                d = distance_to(z, new_node)
                L = accumulate_log_survival(z, new_node, grid, RISK_SAMPLES)
                
                # temporarily store edge physics so we don't recalculate if accepted
                physics_cache[z] = (d, L)
                
                for i in range(len(z.costs)):
                    new_L = z.costs[i,1] + L
                    if 1.0 - np.exp(new_L) <= self.p_fail_threshold:
                        candidates.append([z.costs[i,0]+d, new_L, z, z.set_ids[i]])
                        
        if candidates:
            # pre-filter raw candidates
            cand_matrix = np.array([[c[0], c[1]] for c in candidates])
            mask = self.pareto_filter_vectors(cand_matrix)
            
            valid_cands = [c for i, c in enumerate(candidates) if mask[i]]
            
            if valid_cands:
                n_new = len(valid_cands)
                
                # directly initialize the arrays without queueing
                new_node.costs = np.array([[c[0], c[1]] for c in valid_cands])
                new_node.set_ids = np.arange(0, n_new)
                new_node.next_id = n_new
                
                for c in valid_cands:
                    p_node = c[2]
                    p_set_id = c[3]
                    new_node.lineage.append((p_node, p_set_id))
                    
                    if new_node not in p_node.children:
                        p_node.children.append(new_node)
                        # pull from physics cache
                        p_node.child_edges[new_node] = physics_cache[p_node]

    def rewire(self, znear, new_node, grid):
        # attempts to optimize nearby nodes by routing them through new_node
        if len(new_node.costs) == 0: 
            return
            
        for z in znear:
            if getattr(z, 'is_goal', False) or getattr(z, 'is_start', False): continue
            if self.is_ancestor(z, new_node): continue
            
            if is_edge_collision_free(new_node, z, grid, COLLISION_SAMPLES, 0.9):
                d = distance_to(new_node, z)
                L = accumulate_log_survival(new_node, z, grid, RISK_SAMPLES)
                
                # vectorized batch generation for all new_node costs
                shifted_costs = new_node.costs.copy()
                shifted_costs[:, 0] += d
                shifted_costs[:, 1] += L
                
                # pre-filter hard constraints before queueing
                pfails = 1.0 - np.exp(shifted_costs[:, 1])
                valid_mask = pfails <= self.p_fail_threshold
                
                if np.any(valid_mask):
                    valid_costs = shifted_costs[valid_mask]
                    valid_ids = new_node.set_ids[valid_mask]
                    
                    z.queue_update({
                        'type': 'add_batch', 
                        'costs': valid_costs, 
                        'p_node': new_node, 
                        'p_set_ids': valid_ids
                    })
                    

    def process_node_queue(self, node):
        # 1. cyclical protec
        if not node.dirty or getattr(node, 'processing', False):
            return
            
        # Lock this node so bidirectional lineages don't cause infinite loops
        node.processing = True
        
        try:
            # 2. Recursive Upstream Check
            for (p_node, _) in node.lineage:
                if p_node and p_node.dirty:
                    self.process_node_queue(p_node)
                    
            # Re-check dirty flag in case upstream resolved it
            if not node.dirty:
                return
                
            new_costs_raw, new_lin_raw = [], []
            kill_set = set() 
            delta_map = {} 
            
            # 3. Parse updates and cache edge physics
            for t in node.pending_updates:
                if t['type'] == 'add':
                    new_costs_raw.append([t['c'], t['L']])
                    new_lin_raw.append((t['p_node'], t['p_set_id']))
                    if node not in t['p_node'].children:
                        t['p_node'].children.append(node)
                        edge_d = distance_to(t['p_node'], node)
                        edge_L = accumulate_log_survival(t['p_node'], node, self.grid, RISK_SAMPLES)
                        t['p_node'].child_edges[node] = (edge_d, edge_L)
                elif t['type'] == 'add_batch':
                    new_costs_raw.extend(t['costs'].tolist())
                    for pid in t['p_set_ids']:
                        new_lin_raw.append((t['p_node'], pid))
                    if node not in t['p_node'].children:
                        t['p_node'].children.append(node)
                        edge_d = distance_to(t['p_node'], node)
                        edge_L = accumulate_log_survival(t['p_node'], node, self.grid, RISK_SAMPLES)
                        t['p_node'].child_edges[node] = (edge_d, edge_L)
                elif t['type'] == 'kill':
                    kill_set.add((id(t['p_node']), t['p_set_id']))
                elif t['type'] == 'kill_batch':
                    for pid in t['p_set_ids']:
                        kill_set.add((id(t['p_node']), pid))
                elif t['type'] == 'delta':
                    key = (id(t['p_node']), t['p_set_id'])
                    if key not in delta_map: delta_map[key] = [0.0, 0.0]
                    delta_map[key][0] += t['dc']
                    delta_map[key][1] += t['dl']
                elif t['type'] == 'delta_batch':
                    for pid, dc, dl in t['deltas']:
                        key = (id(t['p_node']), pid)
                        if key not in delta_map: delta_map[key] = [0.0, 0.0]
                        delta_map[key][0] += dc
                        delta_map[key][1] += dl
                        
            node.pending_updates.clear()
            
            # 4. Track existing lineages against Kills and Deltas
            surviving_idx = []
            cascaded_kills = []
            cascaded_deltas = []
            
            for i, (pn, pid) in enumerate(node.lineage):
                key = (id(pn), pid)
                local_id = node.set_ids[i]
                if key in kill_set:
                    cascaded_kills.append(local_id)
                else:
                    if key in delta_map:
                        dc, dl = delta_map[key]
                        node.costs[i, 0] += dc
                        node.costs[i, 1] += dl
                        cascaded_deltas.append((local_id, dc, dl))
                    surviving_idx.append(i)
            
            # Collapse arrays locally
            if surviving_idx:
                node.costs = node.costs[surviving_idx]
                node.set_ids = node.set_ids[surviving_idx]
                node.lineage = [node.lineage[i] for i in surviving_idx]
            else:
                node.costs = np.empty((0,2))
                node.set_ids = np.empty(0, dtype=int)
                node.lineage = []
            
            # 5. Append new incoming arrays
            new_ids = []
            if new_costs_raw:
                n_new = len(new_costs_raw)
                new_ids = np.arange(node.next_id, node.next_id + n_new)
                node.next_id += n_new
                
                node.costs = np.vstack([node.costs, np.array(new_costs_raw)])
                node.set_ids = np.concatenate([node.set_ids, new_ids])
                node.lineage.extend(new_lin_raw)
                
            if len(node.costs) == 0:
                # Node is dead, push upstream kills down
                if cascaded_kills:
                    for child in node.children:
                        child.queue_update({'type': 'kill_batch', 'p_node': node, 'p_set_ids': cascaded_kills})
                node.dirty = False
                return
            
            # 6. Apply hard constraints
            pfails = 1.0 - np.exp(node.costs[:, 1])
            valid_mask = pfails <= self.p_fail_threshold
            
            # 7. Apply pareto filter
            pareto_mask = self.pareto_filter_vectors(node.costs)
            
            # Combine valid survivors
            keep_mask = valid_mask & pareto_mask
            
            # Identify locally pruned arrays to trigger kill cascade
            pruned_idx = np.where(~keep_mask)[0]
            for i in pruned_idx:
                cascaded_kills.append(node.set_ids[i])
                
            # Execute pruning
            node.costs = node.costs[keep_mask]
            node.set_ids = node.set_ids[keep_mask]
            node.lineage = [node.lineage[i] for i in range(len(node.lineage)) if keep_mask[i]]
            
            # 8. CASCADE ONLY SURVIVORS (Queueing down)
            if node.children:
                # Send bulk kill signals
                if cascaded_kills:
                    for child in node.children:
                        child.queue_update({'type': 'kill_batch', 'p_node': node, 'p_set_ids': cascaded_kills})
                        
                surviving_set_ids = set(node.set_ids)
                
                # Send bulk deltas
                valid_deltas = [(lid, dc, dl) for lid, dc, dl in cascaded_deltas if lid in surviving_set_ids]
                if valid_deltas:
                    for child in node.children:
                        child.queue_update({'type': 'delta_batch', 'p_node': node, 'deltas': valid_deltas})
                            
                # Cascade strictly dominant new trade-offs via batching
                if len(new_ids) > 0:
                    surviving_new_indices = [np.where(node.set_ids == nid)[0][0] for nid in new_ids if nid in surviving_set_ids]
                    if surviving_new_indices:
                        bulk_costs = node.costs[surviving_new_indices]
                        bulk_ids = node.set_ids[surviving_new_indices]
                        
                        for child in node.children:
                            if child not in node.child_edges:
                                node.child_edges[child] = (
                                    distance_to(node, child), 
                                    accumulate_log_survival(node, child, self.grid, RISK_SAMPLES)
                                )
                            edge_d, edge_L = node.child_edges[child]
                            
                            shifted_costs = bulk_costs.copy()
                            shifted_costs[:, 0] += edge_d
                            shifted_costs[:, 1] += edge_L
                            
                            child.queue_update({
                                'type': 'add_batch', 
                                'costs': shifted_costs, 
                                'p_node': node, 
                                'p_set_ids': bulk_ids
                            })

            node.dirty = False
            
        finally:
            # 9. Release the recursion lock even if an error occurs
            node.processing = False









def PO_RRT_Star(start, goal, grid, max_iter, p_fail_threshold=0.95):
    tree = Tree(grid, p_fail_threshold=p_fail_threshold)
    start_node = Node(*start)
    start_node.is_start = True
    start_node.costs = np.array([[0.0, 0.0]])
    start_node.set_ids = np.array([0], dtype=int)
    start_node.next_id = 1
    start_node.lineage = [(None, -1)]
    tree.add_node(start_node)
    tree.start_node = start_node
    
    goal_node = Node(*goal)
    goal_node.is_goal = True
    tree.add_node(goal_node)

    fig, ax, lc, _ = init_progress_plot_2d(
        start, goal, 
        x_lim=(0, grid.width), 
        y_lim=(0, grid.height), 
        obstacles=grid.obstacles
    )

    rng = np.random.default_rng(None)

    for current_iter in range(max_iter):
        if current_iter > 0 and current_iter % 250 == 0:
            draw_array_tree_2d_live(tree, lc)
            logging.info(f"Iteration {current_iter}/{max_iter} - Tree size: {len(tree.node_list)}")
            time.sleep(0.1)  # brief pause for visualization update

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
            if z.dirty:
                tree.process_node_queue(z)
        
        # 1. choose optimal pareto parents
        tree.choose_parents(znear, new_node, grid)
            
        # 2. rewire neighbors if node successfully connected
        if len(new_node.costs) > 0:
            tree.add_node(new_node)
            tree.rewire(znear, new_node, grid)

            # 3. bind to goal 
            if distance_to(new_node, goal_node) <= DEFAULT_STEP_SIZE:
                if is_edge_collision_free(new_node, goal_node, grid, COLLISION_SAMPLES, 0.9):
                    d = distance_to(new_node, goal_node)
                    L = accumulate_log_survival(new_node, goal_node, grid, RISK_SAMPLES)
                    
                    shifted_costs = new_node.costs.copy()
                    shifted_costs[:, 0] += d
                    shifted_costs[:, 1] += L
                    
                    pfails = 1.0 - np.exp(shifted_costs[:, 1])
                    valid_mask = pfails <= tree.p_fail_threshold
                    
                    if np.any(valid_mask):
                        goal_node.queue_update({
                            'type': 'add_batch', 
                            'costs': shifted_costs[valid_mask], 
                            'p_node': new_node, 
                            'p_set_ids': new_node.set_ids[valid_mask]
                        })


    if goal_node.dirty: 
        tree.process_node_queue(goal_node)
    # extract backwards 
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

    # pareto filter complete paths
    non_dominated = []
    for i, entry_i in enumerate(multiple_paths):
        dominated = False
        for j, entry_j in enumerate(multiple_paths):
            if i != j:
                if ((entry_j["cost"] <= entry_i["cost"] and entry_j["p_fail"] < entry_i["p_fail"]) or
                    (entry_j["cost"] < entry_i["cost"] and entry_j["p_fail"] <= entry_i["p_fail"])):
                    dominated = True
                    break
        if not dominated:
            non_dominated.append(entry_i)

    # gen structural edges
    edge_segments2d = []
    for n in tree.node_list:
        for child in n.children:
            edge_segments2d.append([(n.x, n.y), (child.x, child.y)])

    return non_dominated, multiple_paths, tree, edge_segments2d

def main():
    root = tk.Tk()
    root.withdraw()
    
    start = (3, 99)
    goal = (80, 1)

    obstacles = [
        {"type": "circular", "center": (50, 65), "radius": 15, "safe_dist": 5},
        {"type": "rectangular", "x_range": (30, 70), "y_range": (20, 40), "probability": 0.07},
        # {"type": "rectangular", "x_range": (30, 70), "y_range": (50, 80), "probability": 0.05},
    ]

    sample_count = simpledialog.askinteger("Input", "Enter how many samples:", initialvalue=2500, minvalue=1)
    PFAILTHRESHOLD = simpledialog.askfloat("Input", "Enter P_FAIL threshold:", initialvalue=0.95, minvalue=0, maxvalue=1)
    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

    filtered_paths, multiple_paths, tree, edge_segments = PO_RRT_Star(start, goal, grid, sample_count, p_fail_threshold=PFAILTHRESHOLD)

    do_plot = input("Plot filtered paths and all paths? (filtered/all/both/none) [filtered]: ").strip().lower() or 'filtered'
    if do_plot in ('filtered', 'both'):
        plot_paths_summary(filtered_paths, obstacles=obstacles)
    if do_plot in ('all', 'both'):
        plot_paths_summary(multiple_paths, obstacles=obstacles)

    do_spec = input("Open spectral clustering visualization? (y/N): ").strip().lower() == 'y'
    if do_spec:
        from PO_RRT_Star_EXACT import spectral_cluster_paths
        interactive_spectral_cluster_plot(filtered_paths, spectral_cluster_paths, obstacles=obstacles)

    do_save = input("Export run to JSON for replay? (y/N): ").strip().lower() == 'y'
    if do_save:
        default_name = f"porrt_export_{int(time.time())}.json"
        fname = input(f"Output filename [{default_name}]: ").strip()
        outfile = fname or default_name
        save_run_json(outfile, start, goal, grid, filtered_paths, multiple_paths, edge_segments)

if __name__ == '__main__':
    main()