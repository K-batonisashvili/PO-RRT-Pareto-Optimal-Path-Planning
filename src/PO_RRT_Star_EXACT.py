import time
from tracemalloc import start
import numpy as np
import logging
import tkinter as tk
import math
from tkinter import simpledialog
from helper_functions import is_edge_collision_free, distance_to, get_coord, is_collision_free, steer, accumulate_log_survival, get_path_signature
from visualization import plot_final_tree_2d,init_progress_plot_3d, update_progress_plot_3d, plot_paths_metrics, redraw_tree, plot_full_paths, plot_paths_summary, init_progress_plot_2d, redraw_tree_2d, interactive_spectral_cluster_plot
from scipy.spatial import cKDTree as ckdtree
import json, time


logging.basicConfig(level=logging.INFO)
# Define constants
GRID_WIDTH = 100
GRID_HEIGHT = 100
PARETO_RADIUS = 10
DEFAULT_STEP_SIZE = 10
PROBABILITY_THRESHOLD = 0.01

# --- Standardized Physics Constants ---
COLLISION_SAMPLES = 20   
RISK_SAMPLES = 10        


# quick import/export functions for tree+paths
def _pack_paths_for_json(paths):
    out = []
    for entry in paths:
        p = entry["path"]
        nodes = p.nodes if hasattr(p, "nodes") else p
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

def save_tree_debug_json(filename, tree):
    node_id = {}
    nodes_out = []

    for idx, n in enumerate(tree.node_list):
        node_id[n] = idx
        nodes_out.append({
            "id": idx,
            "x": float(n.x),
            "y": float(n.y),
            "cost": float(getattr(n, "cost", 0.0)),
            "p_fail": float(getattr(n, "p_fail", 0.0)),
            "log_survival": float(getattr(n, "log_survival", 0.0)),
            "is_start": bool(getattr(n, "is_start", False)),
            "is_goal": bool(getattr(n, "is_goal", False)),
        })

    edges_out = []
    for n in tree.node_list:
        pid = node_id.get(n.parent) if getattr(n, "parent", None) is not None else None
        cid = node_id[n]
        if pid is not None:
            edges_out.append([int(pid), int(cid)])

    payload = {
        "meta": {
            "saved_at_unix": time.time(),
            "num_nodes": len(nodes_out),
            "num_edges": len(edges_out),
        },
        "nodes": nodes_out,
        "edges": edges_out,
    }

    with open(filename, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[OK] exported tree debug → {filename}")


def save_run_json(filename, start, goal, grid, filtered_paths, multiple_paths, edge_segments):
    def _as_seq(x):
        return x.tolist() if isinstance(x, np.ndarray) else x

    def _norm_point(pt):
        pt = _as_seq(pt)
        if len(pt) == 2:
            x, y = pt
            z = 0.0
        elif len(pt) == 3:
            x, y, z = pt
        else:
            return None
        return [float(x), float(y), float(z)]

    def _norm_segment(item):
        seg = item
        if isinstance(seg, (list, tuple)) and len(seg) == 2:
            a, b = seg
            if isinstance(b, str):
                seg = a 
        if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
            return None
        p1, p2 = seg[0], seg[1]
        P1 = _norm_point(p1)
        P2 = _norm_point(p2)
        if P1 is None or P2 is None:
            return None
        return [P1, P2]

    edges = []
    if edge_segments:
        for item in edge_segments:
            e = _norm_segment(item)
            if e:
                edges.append(e)

    if not edges:
        def add_from(entries):
            for entry in entries:
                p = entry["path"]
                nodes = p.nodes if hasattr(p, "nodes") else p
                for i in range(1, len(nodes)):
                    n1, n2 = nodes[i-1], nodes[i]
                    edges.append([
                        [float(n1.x), float(n1.y), float(getattr(n1, "p_fail", 0.0))],
                        [float(n2.x), float(n2.y), float(getattr(n2, "p_fail", 0.0))]
                    ])
        add_from(filtered_paths)
        add_from(multiple_paths)

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
        "tree": {
            "edges": edges 
        },
    }
    with open(filename, "w") as f:
        json.dump(payload, f)
    print(f"[OK] exported → {filename}")


def _resample_polyline_xy(nodes, m=64):
    xs = np.asarray([n.x for n in nodes], dtype=float)
    ys = np.asarray([n.y for n in nodes], dtype=float)
    if len(xs) < 2:
        out = np.tile(np.array([xs[0], ys[0]], dtype=float), (m, 1))
        return out

    seg_dx = np.diff(xs)
    seg_dy = np.diff(ys)
    seg_len = np.hypot(seg_dx, seg_dy)
    total = float(seg_len.sum())
    if total <= 1e-12:
        out = np.tile(np.array([xs[0], ys[0]], dtype=float), (m, 1))
        return out

    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    t = np.linspace(0.0, total, m)
    out = np.zeros((m, 2), dtype=float)

    j = 0
    for i, ti in enumerate(t):
        while j+1 < len(cum) and ti > cum[j+1]:
            j += 1
        if j+1 >= len(cum):
            out[i] = [xs[-1], ys[-1]]
        else:
            if cum[j+1] - cum[j] <= 1e-12:
                alpha = 0.0
            else:
                alpha = (ti - cum[j]) / (cum[j+1] - cum[j])
            x = xs[j] + alpha * (xs[j+1] - xs[j])
            y = ys[j] + alpha * (ys[j+1] - ys[j])
            out[i] = [x, y]
    return out


def _pairwise_path_distance(entries, m_points=64, w_xy=1.0, w_cost=0.25, w_pfail=0.75):
    N = len(entries)
    if N == 0:
        return np.zeros((0, 0))
    xy = []
    costs = np.zeros(N, dtype=float)
    pf   = np.zeros(N, dtype=float)
    for i, e in enumerate(entries):
        nodes = e["path"].nodes if hasattr(e["path"], "nodes") else e["path"]
        xy.append(_resample_polyline_xy(nodes, m=m_points))
        costs[i] = float(e["cost"])
        pf[i]    = float(e["p_fail"])
    xy = np.stack(xy, axis=0)  

    c_scale = np.std(costs) if np.std(costs) > 1e-12 else 1.0
    p_scale = np.std(pf)    if np.std(pf)    > 1e-12 else 1.0

    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Xi = xy[i]
        for j in range(i+1, N):
            Xj = xy[j]
            dxy = np.sqrt(np.mean((Xi[:, 0] - Xj[:, 0])**2 + (Xi[:, 1] - Xj[:, 1])**2))
            dcost = abs(costs[i] - costs[j]) / c_scale
            dpf   = abs(pf[i]    - pf[j])    / p_scale
            d = np.sqrt((w_xy * dxy)**2 + (w_cost * dcost)**2 + (w_pfail * dpf)**2)
            D[i, j] = D[j, i] = d
    return D


def _self_tuning_affinity(D, neighbor_k=7):
    N = D.shape[0]
    if N == 0:
        return np.zeros((0, 0))
    sortD = np.sort(D, axis=1)
    k = max(1, min(neighbor_k, max(1, N-1)))
    sig = sortD[:, k] 
    sig[sig < 1e-12] = np.median(sig[sig > 0]) if np.any(sig > 0) else 1.0
    S = sig[:, None] * sig[None, :]
    A = np.exp(-(D**2) / (S + 1e-12))
    np.fill_diagonal(A, 0.0)
    return 0.5 * (A + A.T)


def _normalized_laplacian(A):
    d = A.sum(axis=1)
    d[d < 1e-12] = 1e-12
    Dmh = np.diag(1.0 / np.sqrt(d))
    I = np.eye(A.shape[0])
    return I - Dmh @ A @ Dmh


def _eigengap_k_from_A(A, k_max=10):
    N = A.shape[0]
    if N <= 2:
        return max(1, N)
    k_max = int(max(2, min(k_max, N)))
    L = _normalized_laplacian(A)
    vals, _ = np.linalg.eigh(L)
    vals = np.clip(vals, 0.0, None)[:k_max] 
    if len(vals) < 3:
        return min(2, N)
    gaps = vals[1:] - vals[:-1]
    idx = 1 + int(np.argmax(gaps[1:])) 
    k = idx + 1
    return int(max(2, min(k, k_max)))


def _kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    centers = [rng.randint(0, n)]
    d2 = np.full(n, np.inf)
    for _ in range(1, k):
        d2 = np.minimum(d2, np.sum((X - X[centers[-1]])**2, axis=1))
        probs = d2 / d2.sum()
        centers.append(rng.choice(n, p=probs))
    return np.array(centers, dtype=int)


def _kmeans(X, k, n_init=10, max_iter=100, random_state=0):
    rng = np.random.RandomState(random_state)
    best_inertia = np.inf
    best_labels  = None
    for _ in range(n_init):
        centers_idx = _kmeans_pp_init(X, k, rng)
        centers = X[centers_idx].copy()
        labels = np.zeros(X.shape[0], dtype=int)
        for _it in range(max_iter):
            d2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if not np.any(mask):
                    centers[j] = X[rng.randint(0, X.shape[0])]
                else:
                    centers[j] = X[mask].mean(axis=0)
        inertia = np.min(((X[:, None, :] - centers[None, :, :])**2).sum(axis=2), axis=1).sum()
        if inertia < best_inertia:
            best_inertia = inertia
            best_labels = labels.copy()
    return best_labels

def spectral_cluster_paths(
    path_entries,
    n_clusters="auto",
    m_points=64,
    w_xy=1.0,
    w_cost=0.25,
    w_pfail=0.75,
    neighbor_k=7,
    prefer_sklearn=True,
    random_state=0,
):
    N = len(path_entries)
    if N == 0:
        return [], np.zeros(0, dtype=int), {}
    if N == 1:
        return [{"members":[path_entries[0]], "representative": path_entries[0]}], np.array([0], int), {}

    D = _pairwise_path_distance(
        path_entries, m_points=m_points, w_xy=w_xy, w_cost=w_cost, w_pfail=w_pfail
    )
    A = _self_tuning_affinity(D, neighbor_k=neighbor_k)

    if isinstance(n_clusters, str) and n_clusters.lower() == "auto":
        k = _eigengap_k_from_A(A, k_max=min(10, N))
        if k < 2:  
            k = min(2, N)
    else:
        k = int(max(1, min(int(n_clusters), N)))

    labels = None
    used_sklearn = False
    if prefer_sklearn:
        try:
            from sklearn.cluster import SpectralClustering
            sc = SpectralClustering(
                n_clusters=k, affinity="precomputed", assign_labels="kmeans", random_state=random_state
            )
            labels = sc.fit_predict(A)
            used_sklearn = True
        except Exception:
            labels = None

    if labels is None:
        L = _normalized_laplacian(A)
        vals, vecs = np.linalg.eigh(L)
        order = np.argsort(vals)
        U = vecs[:, order[:k]]
        row_norm = np.linalg.norm(U, axis=1, keepdims=True)
        row_norm[row_norm < 1e-12] = 1.0
        Z = U / row_norm
        labels = _kmeans(Z, k, n_init=10, max_iter=100, random_state=random_state)

    clusters = []
    for lab in sorted(set(labels.tolist())):
        members = [path_entries[i] for i in range(N) if labels[i] == lab]
        rep = min(members, key=lambda e: (e["cost"], e["p_fail"]))
        clusters.append({"members": members, "representative": rep})

    debug = {"D": D, "A": A, "k": k, "used_sklearn": used_sklearn}
    return clusters, labels, debug


# ----------------------- #
#       Main Classes      #
# ----------------------- #

class Tree:
    def __init__(self, grid):
        self.node_list = []
        self.rewire_counts = 0
        self.rewire_parent_changes = 0      
        self.alternative_branch_creations = 0 
        self.grid = grid
        self.start_node = None
        self.total_nodes_added = 0

    def add_node(self, node):
        if node not in self.node_list:
            self.node_list.append(node)
            self.total_nodes_added += 1
    
    def connection_radius(self):
        """
        Standard RRT* neighbor radius:
            r_n = min{ gamma * (log n / n)^(1/d), eta }

        d=2 because the KDTree is built in (x,y).
        free_volume is approximated by width*height.
        eta is the steering limit (DEFAULT_STEP_SIZE).
        """
        n = max(len(self.node_list), 2)
        d = 2

        zeta_d = math.pi  # volume of unit ball in R^2
        free_volume = float(self.grid.width * self.grid.height)

        gamma_rrt = 2.0 * ((1.0 + 1.0 / d) ** (1.0 / d)) * ((free_volume / zeta_d) ** (1.0 / d))
        base_radius = gamma_rrt * ((math.log(n) / n) ** (1.0 / d))

        eta = float(DEFAULT_STEP_SIZE)
        r = min(base_radius, eta)

        # tiny safety (optional)
        # return min(base_radius, eta)
        return 10

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
        radius = self.connection_radius()
        r2 = radius * radius
        rx = radius
        res = []
        nx, ny = node.x, node.y
        for n in self.node_list:
            if n is node:
                continue
            # Fast axis-aligned bounding box pre-filter
            if abs(n.x - nx) > rx or abs(n.y - ny) > rx:
                continue
            # Actual distance check
            if (n.x - nx)**2 + (n.y - ny)**2 <= r2:
                res.append(n)
        return res

    def is_ancestor(self, candidate_ancestor, node):
        """Safely traverse UP the tree to prevent cycle loops."""
        cur = node
        while cur is not None:
            if cur is candidate_ancestor:
                return True
            cur = cur.parent
        return False
   
    def pareto_dominates(self, cost1, fail1, cost2, fail2):
        return (cost1 <= cost2 and fail1 < fail2) or \
        (cost1 < cost2 and fail1  <= fail2)
    
    def choose_parents(self, znear, x, y, grid):
        test_node = Node(x, y)
        new_node_candidates = []
        
        edge_cache = {}
        for potential_parent in znear:
            spatial_key = (potential_parent.x, potential_parent.y)
            if spatial_key not in edge_cache:
                free = is_edge_collision_free(potential_parent, test_node, grid, num_samples=COLLISION_SAMPLES, p_threshold=0.9)
                log_s = accumulate_log_survival(potential_parent, test_node, grid, num_samples=RISK_SAMPLES) if free else 0
                edge_cache[spatial_key] = (free, log_s)
                
            free, log_s_step = edge_cache[spatial_key]
            
            if not free:
                continue
                
            cost   = potential_parent.cost + distance_to(potential_parent, Node(x,y))
            log_survival = potential_parent.log_survival + log_s_step
            prob_failure  = 1 - np.exp(log_survival)
            new_node_candidates.append((potential_parent, cost, log_survival, prob_failure))

        pareto_dominant_nodes = []
        for pa in new_node_candidates:
            dominated = False
            for pb in new_node_candidates:
                if (pb is not pa) and self.pareto_dominates(pb[1], pb[3], pa[1], pa[3]):
                    dominated = True
                    break
            
            if not dominated:
                is_duplicate = False
                for acc in pareto_dominant_nodes:
                    if abs(acc[1] - pa[1]) < 0.1 and abs(acc[3] - pa[3]) < 0.005:
                        is_duplicate = True
                        break
                
                if not is_duplicate:
                    pareto_dominant_nodes.append(pa)

        final_pareto_nodes = []
        for potential_parent, cost, log_surv, p_fail in pareto_dominant_nodes:
            new_node = Node(x, y)
            new_node.parent = potential_parent
            potential_parent.children.append(new_node)
            new_node.cost = cost
            new_node.log_survival = log_surv
            new_node.p_fail = p_fail
            final_pareto_nodes.append(new_node)
        return final_pareto_nodes
    
    # def rewire(self, znear, nn, grid):
    #     edge_cache = {}
        
    #     for z in znear:
    #         if z is nn or nn in z.children:
    #             continue 
    #         if self.is_ancestor(z, nn):
    #             continue
            
    #         spatial_key = (round(z.x, 3), round(z.y, 3))
    #         if spatial_key not in edge_cache:
    #             free = is_edge_collision_free(nn, z, grid, num_samples=COLLISION_SAMPLES, p_threshold=0.9)
    #             log_s = accumulate_log_survival(nn, z, grid, num_samples=RISK_SAMPLES) if free else 0
    #             edge_cache[spatial_key] = (free, log_s)
                
    #         free, log_s_step = edge_cache[spatial_key]
            
    #         if not free:
    #             continue

    #         new_log_survival = nn.log_survival + log_s_step
    #         new_cost = nn.cost + distance_to(nn, z)
    #         new_p_fail = 1 - np.exp(new_log_survival)

    #         if self.pareto_dominates(z.cost, z.p_fail, new_cost, new_p_fail):
    #             continue
                
    #         if abs(z.cost - new_cost) < 0.1 and abs(z.p_fail - new_p_fail) < 0.005:
    #             continue

    #         if self.pareto_dominates(new_cost, new_p_fail, z.cost, z.p_fail):
    #             old_parent = z.parent
    #             if old_parent and z in old_parent.children:
    #                 old_parent.children.remove(z)
    #             z.parent = nn
    #             nn.children.append(z)
    #             z.cost = new_cost
    #             z.log_survival = new_log_survival
    #             z.p_fail = new_p_fail
                
    #             self.propagate_cost(z, grid)
    #             self.rewire_counts += 1
    #             self.rewire_parent_changes += 1   
    #         else:
    #             if abs(z.cost - new_cost) > 0.1 or abs(z.p_fail - new_p_fail) > 0.005:
    #                 new_z = Node(z.x, z.y)
    #                 new_z.parent = nn
    #                 nn.children.append(new_z)
    #                 new_z.cost = new_cost
    #                 new_z.log_survival = new_log_survival
    #                 new_z.p_fail = new_p_fail
                    
    #                 self.add_node(new_z) 
    #                 new_z.is_additional_rewire = True 
                    
    #                 self.rewire_counts += 1
    #                 self.alternative_branch_creations += 1

    # def rewire(self, znear, nn, grid):
    #     # 1. Group all neighbors by their spatial coordinate to prevent clone explosions
    #     spatial_groups = {}
    #     for z in znear:
    #         if z is nn or nn in z.children or self.is_ancestor(z, nn):
    #             continue
    #         key = (z.x, 3, z.y, 3)
    #         if key not in spatial_groups:
    #             spatial_groups[key] = []
    #         spatial_groups[key].append(z)

    #     # 2. Evaluate the proposed path from 'nn' to each spatial location EXACTLY ONCE
    #     for spatial_key, nodes_at_loc in spatial_groups.items():
    #         rep_node = nodes_at_loc[0] # Representative node for geometry math
            
    #         # Math is only done once per spatial group!
    #         free = is_edge_collision_free(nn, rep_node, grid, num_samples=COLLISION_SAMPLES, p_threshold=0.9)
    #         if not free:
    #             continue

    #         log_s_step = accumulate_log_survival(nn, rep_node, grid, num_samples=RISK_SAMPLES)
    #         new_log_survival = nn.log_survival + log_s_step
    #         new_cost = nn.cost + distance_to(nn, rep_node)
    #         new_p_fail = 1 - np.exp(new_log_survival)

    #         # 3. Check the new path against ALL existing Pareto nodes at this location
    #         is_dominated = False
    #         is_epsilon_duplicate = False
    #         nodes_to_rewire = []

    #         for z in nodes_at_loc:
    #             # If the new path is worse than ANY existing node here, reject it entirely
    #             if self.pareto_dominates(z.cost, z.p_fail, new_cost, new_p_fail):
    #                 is_dominated = True
    #                 break
                
    #             # If the new path is epsilon-identical to ANY existing node here, reject it
    #             if abs(z.cost - new_cost) < 0.1 and abs(z.p_fail - new_p_fail) < 0.005:
    #                 is_epsilon_duplicate = True
    #                 break
                
    #             # If the new path strictly dominates an existing node, mark that node for rewiring
    #             if self.pareto_dominates(new_cost, new_p_fail, z.cost, z.p_fail):
    #                 nodes_to_rewire.append(z)

    #         if is_dominated or is_epsilon_duplicate:
    #             continue

    #         # 4. Execute Strict Rewires
    #         for z in nodes_to_rewire:
    #             old_parent = z.parent
    #             if old_parent and z in old_parent.children:
    #                 old_parent.children.remove(z)
                
    #             z.parent = nn
    #             if z not in nn.children:
    #                 nn.children.append(z)
                
    #             z.cost = new_cost
    #             z.log_survival = new_log_survival
    #             z.p_fail = new_p_fail
                
    #             self.propagate_cost(z, grid)
    #             self.rewire_counts += 1
    #             self.rewire_parent_changes += 1   

    #         # 5. Execute Tradeoff Branching (Spawn MAX ONE branch per location!)
    #         if len(nodes_to_rewire) < len(nodes_at_loc):
    #             new_z = Node(rep_node.x, rep_node.y)
    #             new_z.parent = nn
    #             nn.children.append(new_z)
    #             new_z.cost = new_cost
    #             new_z.log_survival = new_log_survival
    #             new_z.p_fail = new_p_fail
                
    #             self.add_node(new_z) 
    #             new_z.is_additional_rewire = True 
                
    #             self.rewire_counts += 1
    #             self.alternative_branch_creations += 1

    def rewire(self, znear, nn, grid):
        # 1. Group all neighbors by their spatial coordinate to prevent clone explosions
        spatial_groups = {}
        for z in znear:
            if z is nn or nn in z.children or self.is_ancestor(z, nn):
                continue
            
            key = (z.x, z.y)
            if key not in spatial_groups:
                spatial_groups[key] = []
            spatial_groups[key].append(z)

        # 2. Evaluate the proposed path from 'nn' to each spatial location EXACTLY ONCE
        for spatial_key, nodes_at_loc in spatial_groups.items():
            rep_node = nodes_at_loc[0] 
            
            free = is_edge_collision_free(nn, rep_node, grid, num_samples=COLLISION_SAMPLES, p_threshold=0.9)
            if not free:
                continue

            log_s_step = accumulate_log_survival(nn, rep_node, grid, num_samples=RISK_SAMPLES)
            new_log_survival = nn.log_survival + log_s_step
            new_cost = nn.cost + distance_to(nn, rep_node)
            new_p_fail = 1 - np.exp(new_log_survival)

            # 3. Check the new path against ALL existing Pareto nodes at this location
            is_dominated_by_any = False
            is_tradeoff_duplicate = False
            nodes_to_rewire = []

            for z in nodes_at_loc:
                # Does the existing node completely dominate the new path?
                if self.pareto_dominates(z.cost, z.p_fail, new_cost, new_p_fail):
                    is_dominated_by_any = True
                    break
                
                # Does the new path strictly dominate this existing node?
                if self.pareto_dominates(new_cost, new_p_fail, z.cost, z.p_fail):
                    nodes_to_rewire.append(z)
                else:
                    # It's a non-dominated tradeoff. Check if it's too similar to warrant a new branch.
                    if abs(z.cost - new_cost) < 0.1 and abs(z.p_fail - new_p_fail) < 0.005:
                        is_tradeoff_duplicate = True

            if is_dominated_by_any:
                continue

            # 4. Execute Strict Rewires (Straightens out paths down to the decimal!)
            for z in nodes_to_rewire:
                old_parent = z.parent
                if old_parent and z in old_parent.children:
                    old_parent.children.remove(z)
                
                z.parent = nn
                if z not in nn.children:
                    nn.children.append(z)
                
                z.cost = new_cost
                z.log_survival = new_log_survival
                z.p_fail = new_p_fail
                
                self.propagate_cost(z, grid)
                self.rewire_counts += 1
                self.rewire_parent_changes += 1   

            # 5. Execute Tradeoff Branching
            # CRITICAL: Only spawn a new branch if we didn't just inject the exact same cost 
            # by strictly rewiring an existing node above!
            # 5. Execute Tradeoff Branching
            if not is_tradeoff_duplicate and len(nodes_to_rewire) == 0:
                new_z = Node(rep_node.x, rep_node.y)
                new_z.parent = nn
                nn.children.append(new_z)
                new_z.cost = new_cost
                new_z.log_survival = new_log_survival
                new_z.p_fail = new_p_fail
                
                self.add_node(new_z) 
                new_z.is_additional_rewire = True 
                
                self.rewire_counts += 1
                self.alternative_branch_creations += 1  

                # ---> NEW: Propagate the tradeoff down the existing subtree! <---
                self.clone_and_propagate_subtree(z, new_z, grid)

    def clone_and_propagate_subtree(self, old_node, new_node, grid):
        """
        Recursively clones the children of 'old_node' and attaches them 
        to 'new_node', propagating the new trade-off costs down the branch.
        """
        for child in old_node.children:
            # 1. Create a spatial clone of the child
            child_clone = Node(child.x, child.y)
            child_clone.parent = new_node
            new_node.children.append(child_clone)
            
            # 2. Calculate the cost from the new parent
            d_edge = distance_to(new_node, child_clone)
            log_s_step = accumulate_log_survival(new_node, child_clone, grid, num_samples=RISK_SAMPLES)
            
            child_clone.cost = new_node.cost + d_edge
            child_clone.log_survival = new_node.log_survival + log_s_step
            child_clone.p_fail = 1 - np.exp(child_clone.log_survival)
            
            # 3. Add to the tree 
            self.add_node(child_clone)
            child_clone.is_additional_rewire = True
            self.alternative_branch_creations += 1
            
            # 4. Recurse deeper into the tree
            self.clone_and_propagate_subtree(child, child_clone, grid)

    def propagate_cost(self, root, grid):
        """Propagates updated costs recursively down to all children"""
        queue = [root]
        while queue:
            node = queue.pop(0)
            for child in node.children:
                new_cost = node.cost + distance_to(node, child)
                log_s_step = accumulate_log_survival(node, child, grid, num_samples=RISK_SAMPLES)
                new_log_survival = node.log_survival + log_s_step
                
                child.cost = new_cost
                child.log_survival = new_log_survival
                child.p_fail = 1 - np.exp(new_log_survival)
                queue.append(child)

class Path:
    def __init__(self):
        self.nodes = []
    
    @property
    def cost(self) -> float:
        return self.nodes[-1].cost if self.nodes else 0.0
    
    @property
    def p_fail(self) -> float:
        return self.nodes[-1].p_fail if self.nodes else 1.0

class Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.children = []
        self.cost = 0.0
        self.p_fail = 0.0
        self.log_survival = 0.0
        self.is_goal = False 
        self.is_start = False 
        self.is_additional_rewire = False

class Grid:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.grid = np.zeros((width, height))
        self.obstacles = obstacles
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
        cx = int(center[0] / self.width * (self.width - 1))
        cy = int(center[1] / self.height * (self.height - 1))
        rad_cells = int(radius / self.width * (self.width - 1))
        safe_cells = int(safe_dist / self.width * (self.width - 1))
        for x in range(cx - rad_cells - safe_cells, cx + rad_cells + safe_cells + 1):
            for y in range(cy - rad_cells - safe_cells, cy + rad_cells + safe_cells + 1):
                if 0 <= x < self.width and 0 <= y < self.height:
                    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
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


def PO_RRT_Star(start, goal, grid, max_iter, sample_sequence=None, rng_seed=None):
    start_node, goal_node = Node(*start), Node(*goal)
    tree = Tree(grid)
    tree.add_node(start_node)
    tree.start_node = start_node
    goal_node.is_goal = True
    start_node.is_start = True

    if sample_sequence is None:
        rng = np.random.default_rng(rng_seed)
    else:
        rng = None

    for current_iter in range(max_iter):
        if sample_sequence is not None:
            if current_iter >= len(sample_sequence): break  
            sx, sy = sample_sequence[current_iter]
            rand_node = Node(float(sx), float(sy))
        else:
            rand_node = Node(float(rng.uniform(0, grid.width)), float(rng.uniform(0, grid.height)))

        # if is_collision_free(rand_node, grid):  
        #     nearest_node = tree.nearest(rand_node)
        #     x, y = steer(nearest_node, rand_node, DEFAULT_STEP_SIZE)
        #     new_node = Node(x, y)
        #     if is_collision_free(new_node, grid):              
        #         znear = tree.neighbors(new_node)
        #         znear = [n for n in znear if n.x != goal_node.x or n.y != goal_node.y]
                
        #         new_nodes = tree.choose_parents(znear, new_node.x, new_node.y, grid)   
                    
        #         for nn in new_nodes:
        #             if distance_to(nn, goal) <= DEFAULT_STEP_SIZE:
        #                     goal_instance = Node(goal[0], goal[1])
        #                     goal_instance.is_goal = True
        #                     goal_instance.parent = nn
        #                     nn.children.append(goal_instance)
                            
        #                     tree.add_node(nn)
        #                     tree.rewire(tree.neighbors(nn), nn, grid)
                            
        #                     goal_instance.cost = nn.cost + distance_to(nn, goal_instance)
        #                     log_s_step = accumulate_log_survival(nn, goal_instance, grid, num_samples=RISK_SAMPLES)
        #                     goal_instance.log_survival = nn.log_survival + log_s_step
        #                     goal_instance.p_fail = 1 - np.exp(goal_instance.log_survival)
        #                     tree.add_node(goal_instance) 
        #             else:
        #                 tree.add_node(nn)
        #                 tree.rewire(tree.neighbors(nn), nn, grid)

        if is_collision_free(rand_node, grid):  
            nearest_node = tree.nearest(rand_node)
            x, y = steer(nearest_node, rand_node, DEFAULT_STEP_SIZE)
            new_node = Node(x, y)
            
            if is_collision_free(new_node, grid):              
                znear = tree.neighbors(new_node)
                znear = [n for n in znear if n.x != goal_node.x or n.y != goal_node.y]
                
                new_nodes = tree.choose_parents(znear, new_node.x, new_node.y, grid)   
                    
                for nn in new_nodes:
                    # REMOVED: Mid-loop Goal Injection. 
                    # We now grow the tree identically to Standard RRT*
                    tree.add_node(nn)
                    tree.rewire(tree.neighbors(nn), nn, grid)

    # --- Post-Run Exhaustive Goal Connection ---
    goal_template = Node(*goal)
    post_run_goals = 0
    for node in tree.node_list:
        if not getattr(node, "is_goal", False) and distance_to(node, goal_template) <= DEFAULT_STEP_SIZE:
            if not any(getattr(child, "is_goal", False) for child in node.children):
                if is_edge_collision_free(node, goal_template, grid, num_samples=COLLISION_SAMPLES, p_threshold=0.9):
                    g = Node(goal[0], goal[1])
                    g.is_goal = True
                    g.parent = node
                    node.children.append(g)
                    g.cost = node.cost + distance_to(node, g)
                    log_s = accumulate_log_survival(node, g, grid, num_samples=RISK_SAMPLES)
                    g.log_survival = node.log_survival + log_s
                    g.p_fail = 1 - np.exp(g.log_survival)
                    tree.node_list.append(g)
                    post_run_goals += 1
    
    if post_run_goals > 0:
        print(f"PO-RRT* Post-Run: Connected {post_run_goals} additional nodes to goal.")

    # --- TRACE BACKWARDS TO BUILD PATHS ---
    multiple_paths = []
    goal_nodes_in_tree = [n for n in tree.node_list if getattr(n, "is_goal", False)]
    
    for g in goal_nodes_in_tree:
        stack = []
        cur = g
        while cur is not None:
            stack.append(cur)
            cur = cur.parent
        
        # If it reached the start root successfully
        if stack and getattr(stack[-1], "is_start", False):
            stack.reverse()
            p = Path()
            p.nodes = stack
            multiple_paths.append({"path": p, "cost": g.cost, "p_fail": g.p_fail})
    
    # Pareto Filter
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

    # Unique Filter
    unique_filtered = []
    seen = set()
    for entry in non_dominated:
        key = (round(entry["cost"], 6), round(entry["p_fail"], 8))  
        if key not in seen:
            seen.add(key)
            unique_filtered.append(entry)

    # Package edge segments cleanly for visualizer
    edge_segments2d = []
    for n in tree.node_list:
        if n.parent:
            edge_segments2d.append([(n.parent.x, n.parent.y), (n.x, n.y)])

    return unique_filtered, multiple_paths, tree, edge_segments2d

def main():
    root = tk.Tk()
    root.withdraw()
    obstacles = []
    
    # start, goal = (3, 95), (80, 50)

    start = (3, 99)
    goal = (80, 1)
    # obstacles = [
    #     {"type": "rectangular", "x_range": (10, 20), "y_range": (60, 80), "probability": 1.0},
    #     {"type": "rectangular", "x_range": (40, 50), "y_range": (30, 70), "probability": 1.0},
    #     {"type": "rectangular", "x_range": (50, 60), "y_range": (0, 20), "probability": 1.0},
    #     {"type": "rectangular", "x_range": (40, 70), "y_range": (30, 35), "probability": 1.0},
    #     {"type": "rectangular", "x_range": (70, 75), "y_range": (20, 35), "probability": 1.0},
    #     {"type": "rectangular", "x_range": (60, 70), "y_range": (60, 80), "probability": 1.0},
    #     {"type": "rectangular", "x_range": (70, 80), "y_range": (80, 95), "probability": 1.0},
    #     {"type": "rectangular", "x_range": (60, 80), "y_range": (75, 80), "probability": 1.0},
    #     {"type": "circular", "center": (20, 40), "radius": 5, "safe_dist": 2},
    # ]

    obstacles = [
        {"type": "circular", "center": (50, 65), "radius": 15, "safe_dist": 5},
        {"type": "rectangular", "x_range": (30, 70), "y_range": (20, 40), "probability": 0.05},
    ]

    sample_count = simpledialog.askinteger("Input", "Enter how many samples:", initialvalue=2500, minvalue=1)
    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

    filtered_paths, multiple_paths, tree, edge_segments = PO_RRT_Star(start, goal, grid, sample_count)
    
    do_plot = input("Plot filtered paths and all paths? (filtered/all/both/none) [filtered]: ").strip().lower() or 'filtered'
    if do_plot in ('filtered', 'both'):
        try:
            plot_paths_summary(filtered_paths, obstacles=obstacles)
        except Exception as e:
            print(f"Failed to plot filtered paths: {e}")
    if do_plot in ('all', 'both'):
        try:
            plot_paths_summary(multiple_paths, obstacles=obstacles)
        except Exception as e:
            print(f"Failed to plot all paths: {e}")

    do_spec = input("Open spectral clustering visualization? (y/N): ").strip().lower() == 'y'
    if do_spec:
        try:
            clusters, labels, dbg = spectral_cluster_paths(filtered_paths, n_clusters="auto")
            print(f"[spectral] chose k={dbg.get('k')} (sklearn={dbg.get('used_sklearn')})")
            interactive_spectral_cluster_plot(filtered_paths, spectral_cluster_paths, obstacles=obstacles)
        except Exception as e:
            print(f"Spectral visualization failed: {e}")

    do_save = input("Export run to JSON for replay? (y/N): ").strip().lower() == 'y'
    if do_save:
        default_name = f"porrt_export_{int(time.time())}.json"
        fname = input(f"Output filename [{default_name}]: ").strip()
        outfile = fname or default_name
        save_run_json(outfile, start, goal, grid, filtered_paths, multiple_paths, edge_segments)

if __name__ == '__main__':
    main()