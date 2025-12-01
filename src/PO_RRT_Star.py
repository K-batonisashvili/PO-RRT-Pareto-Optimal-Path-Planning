import time
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
                    "theta": float(getattr(n, "theta", 0.0)),
                    "cost": float(getattr(n, "cost", 0.0)),
                    "p_fail": float(getattr(n, "p_fail", 0.0)),
                } for n in nodes
            ],
        })
    return out

def save_tree_debug_json(filename, tree):
    """
    Export the entire tree structure for debugging:
    - Every node with its coordinates, cost, p_fail, flags
    - Parent-child relationships as IDs

    This uses tree.node_list as the canonical set of nodes.
    """
    # Assign an integer ID to every node
    node_id = {}
    nodes_out = []

    for idx, n in enumerate(tree.node_list):
        node_id[n] = idx
        nodes_out.append({
            "id": idx,
            "x": float(n.x),
            "y": float(n.y),
            "theta": float(getattr(n, "theta", 0.0)),
            "cost": float(getattr(n, "cost", 0.0)),
            "p_fail": float(getattr(n, "p_fail", 0.0)),
            "log_survival": float(getattr(n, "log_survival", 0.0)),
            "is_start": bool(getattr(n, "is_start", False)),
            "is_goal": bool(getattr(n, "is_goal", False)),
        })

    # Build parent-child edges as pairs of node IDs
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
        # pt can be (x,y) or (x,y,z)
        if len(pt) == 2:
            x, y = pt
            z = 0.0
        elif len(pt) == 3:
            x, y, z = pt
        else:
            return None
        return [float(x), float(y), float(z)]

    def _norm_segment(item):
        # Accept shapes:
        #   [(x,y,z), (x2,y2,z2)]
        #   [(x,y),   (x2,y2)]
        #   ([(...),(...)], "green")
        seg = item
        if isinstance(seg, (list, tuple)) and len(seg) == 2:
            a, b = seg
            # colored form?
            if isinstance(b, str):
                seg = a  # first element is the actual segment
        # now seg should be a pair of points
        if not (isinstance(seg, (list, tuple)) and len(seg) >= 2):
            return None
        p1, p2 = seg[0], seg[1]
        P1 = _norm_point(p1)
        P2 = _norm_point(p2)
        if P1 is None or P2 is None:
            return None
        return [P1, P2]

    # 1) Normalize any collected edge segments
    edges = []
    if edge_segments:
        for item in edge_segments:
            e = _norm_segment(item)
            if e:
                edges.append(e)

    # 2) Fallback: build edges from paths if nothing got collected
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
            "edges": edges  # always [[x,y,z],[x2,y2,z2]]
        },
    }
    with open(filename, "w") as f:
        json.dump(payload, f)
    print(f"[OK] exported → {filename}")


def _resample_polyline_xy(nodes, m=64):
    """
    Resample a polyline (list of Nodes) to m equally spaced points along arc length.
    Returns (m,2) array of xy points.
    """
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
    # target arc-lengths
    t = np.linspace(0.0, total, m)
    out = np.zeros((m, 2), dtype=float)

    # walk segments once
    j = 0
    for i, ti in enumerate(t):
        # advance until cum[j] <= ti <= cum[j+1]
        while j+1 < len(cum) and ti > cum[j+1]:
            j += 1
        if j+1 >= len(cum):
            out[i] = [xs[-1], ys[-1]]
        else:
            # local interpolation parameter
            if cum[j+1] - cum[j] <= 1e-12:
                alpha = 0.0
            else:
                alpha = (ti - cum[j]) / (cum[j+1] - cum[j])
            x = xs[j] + alpha * (xs[j+1] - xs[j])
            y = ys[j] + alpha * (ys[j+1] - ys[j])
            out[i] = [x, y]
    return out


def _pairwise_path_distance(entries, m_points=64, w_xy=1.0, w_cost=0.25, w_pfail=0.75):
    """
    Build an NxN distance matrix between completed paths combining:
      - RMS XY distance of resampled polylines (weight w_xy)
      - normalized Δcost   (weight w_cost)
      - normalized Δp_fail (weight w_pfail)
    entries: list of {"path": Path, "cost": float, "p_fail": float}
    """
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
    xy = np.stack(xy, axis=0)  # (N, m, 2)

    # scales for normalization (avoid div by zero)
    c_scale = np.std(costs) if np.std(costs) > 1e-12 else 1.0
    p_scale = np.std(pf)    if np.std(pf)    > 1e-12 else 1.0

    D = np.zeros((N, N), dtype=float)
    for i in range(N):
        Xi = xy[i]
        for j in range(i+1, N):
            Xj = xy[j]
            # RMS pointwise XY distance
            dxy = np.sqrt(np.mean((Xi[:, 0] - Xj[:, 0])**2 + (Xi[:, 1] - Xj[:, 1])**2))
            # normalized metric deltas
            dcost = abs(costs[i] - costs[j]) / c_scale
            dpf   = abs(pf[i]    - pf[j])    / p_scale
            d = np.sqrt((w_xy * dxy)**2 + (w_cost * dcost)**2 + (w_pfail * dpf)**2)
            D[i, j] = D[j, i] = d
    return D


def _self_tuning_affinity(D, neighbor_k=7):
    """
    Zelnik-Manor & Perona self-tuning kernel:
        A_ij = exp( - D_ij^2 / (σ_i σ_j) )
    with σ_i = distance to k-th nearest neighbor of i.
    """
    N = D.shape[0]
    if N == 0:
        return np.zeros((0, 0))
    # sort distances row-wise (exclude self at 0)
    sortD = np.sort(D, axis=1)
    k = max(1, min(neighbor_k, max(1, N-1)))
    sig = sortD[:, k]  # distance to k-th neighbor (0-based; self is 0)
    sig[sig < 1e-12] = np.median(sig[sig > 0]) if np.any(sig > 0) else 1.0
    S = sig[:, None] * sig[None, :]
    A = np.exp(-(D**2) / (S + 1e-12))
    np.fill_diagonal(A, 0.0)
    # symmetrize (should already be)
    return 0.5 * (A + A.T)


def _normalized_laplacian(A):
    d = A.sum(axis=1)
    d[d < 1e-12] = 1e-12
    Dmh = np.diag(1.0 / np.sqrt(d))
    # L_sym = I - D^-1/2 A D^-1/2
    I = np.eye(A.shape[0])
    return I - Dmh @ A @ Dmh


def _eigengap_k_from_A(A, k_max=10):
    """
    Choose k using eigengap heuristic on normalized Laplacian.
    Ensure k in [2, min(k_max, N)].
    """
    N = A.shape[0]
    if N <= 2:
        return max(1, N)
    k_max = int(max(2, min(k_max, N)))
    L = _normalized_laplacian(A)
    # small eigenvalues first
    vals, _ = np.linalg.eigh(L)
    vals = np.clip(vals, 0.0, None)[:k_max]  # first k_max smallest
    # ignore gap at index 0 (between λ0≈0 and λ1)
    if len(vals) < 3:
        return min(2, N)
    gaps = vals[1:] - vals[:-1]
    # pick the largest gap starting from index 1
    idx = 1 + int(np.argmax(gaps[1:]))  # 1-based gap -> k = idx+1
    k = idx + 1
    return int(max(2, min(k, k_max)))


def _kmeans_pp_init(X, k, rng):
    n = X.shape[0]
    # pick first center uniformly
    centers = [rng.randint(0, n)]
    # distances to nearest center
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
            # assign
            d2 = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
            new_labels = np.argmin(d2, axis=1)
            if np.array_equal(new_labels, labels):
                break
            labels = new_labels
            # update
            for j in range(k):
                mask = labels == j
                if not np.any(mask):
                    # re-seed empty cluster
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
    """
    Cluster completed paths using spectral clustering on a self-tuned affinity.

    Returns:
      clusters: list of {"members":[...], "representative": entry}
      labels:   np.ndarray shape (N,)
      debug:    dict with matrices used
    """
    N = len(path_entries)
    if N == 0:
        return [], np.zeros(0, dtype=int), {}
    if N == 1:
        return [{"members":[path_entries[0]], "representative": path_entries[0]}], np.array([0], int), {}

    # pairwise distances and affinity
    D = _pairwise_path_distance(
        path_entries, m_points=m_points, w_xy=w_xy, w_cost=w_cost, w_pfail=w_pfail
    )
    A = _self_tuning_affinity(D, neighbor_k=neighbor_k)

    # cluster count
    if isinstance(n_clusters, str) and n_clusters.lower() == "auto":
        k = _eigengap_k_from_A(A, k_max=min(10, N))
        if k < 2:  # safety
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
        # NumPy-only fallback: spectral embedding + kmeans
        L = _normalized_laplacian(A)
        vals, vecs = np.linalg.eigh(L)
        # take k smallest eigenvectors (skip the first if val ~ 0)
        order = np.argsort(vals)
        U = vecs[:, order[:k]]
        # row normalize
        row_norm = np.linalg.norm(U, axis=1, keepdims=True)
        row_norm[row_norm < 1e-12] = 1.0
        Z = U / row_norm
        labels = _kmeans(Z, k, n_init=10, max_iter=100, random_state=random_state)

    # build clusters with representatives
    clusters = []
    for lab in sorted(set(labels.tolist())):
        members = [path_entries[i] for i in range(N) if labels[i] == lab]
        rep = min(members, key=lambda e: (e["cost"], e["p_fail"]))
        clusters.append({"members": members, "representative": rep})

    debug = {"D": D, "A": A, "k": k, "used_sklearn": used_sklearn}
    return clusters, labels, debug



# ----------------------- #
#       Main Classes      #
# ----------------------- #t

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
            self.kdtree = None  # Invalidate KDTree cache
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

    def build_kdtree(self):
        """
        Build a ckdtree for efficient nearest neighbor searches.
        """
        self.kdtree = ckdtree([(n.x, n.y) for n in self.node_list])

    def connection_radius(self):
        """
        Dynamic RRT* neighbor radius:
            r_n = min{ gamma * (log n / n)^(1/d), eta }

        - We use d = 2 (x, y only, matching the kd-tree).
        - Approximate free space volume by grid.width * grid.height.
        - eta is chosen as a multiple of DEFAULT_STEP_SIZE.
        """
        # Number of nodes in the tree (avoid log(1) / log(0))
        n = max(len(self.node_list), 2)
        d = 2  # 2D (x,y) space for the kd-tree

        # Volume of unit ball in R^2
        zeta_d = math.pi

        # Approximate free volume by the grid dimensions
        free_volume = float(self.grid.width * self.grid.height)

        # Gamma_RRT* per Karaman & Frazzoli (up to a constant factor)
        gamma_rrt = 2.0 * ((1.0 + 1.0 / d) ** (1.0 / d)) * (
            (free_volume / zeta_d) ** (1.0 / d)
        )

        # Base theoretical radius
        base_radius = gamma_rrt * ((math.log(n) / n) ** (1.0 / d))

        # Cap radius by a multiple of the step size (eta)
        eta = DEFAULT_STEP_SIZE * 2.0  # tune this multiplier as desired
        r = min(base_radius, eta)

        # Safety: avoid zero / extremely tiny radius
        return max(r, 0.5 * DEFAULT_STEP_SIZE)


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


    def nearest(self, node):
        """Return nearest node using cached KDTree."""
        if self.kdtree is None:
            self.build_kdtree()
        _, idx = self.kdtree.query([node.x, node.y], k=1)
        return self.node_list[int(idx)]
    
        
    # def neighbors(self, node):
    #     """
    #     Efficiently find all unique nodes within PARETO_RADIUS using ckdtree.
    #     """
    #     if self.kdtree is None:
    #         self.build_kdtree()

    #     # Query neighbors within radius
    #     idxs = self.kdtree.query_ball_point([node.x, node.y], r=PARETO_RADIUS)

    #     neighbors = []
    #     seen = set()

    #     for idx in idxs:
    #         n = self.node_list[idx]
    #         if n is node:
    #             continue

    #         node_signature = (n.x, n.y, n.theta, round(n.cost, 3), round(n.p_fail, 5))
    #         if node_signature not in seen:
    #             neighbors.append(n)
    #             seen.add(node_signature)

    #     return neighbors

    def neighbors(self, node):
        """
        Efficiently find all unique nodes within a dynamic RRT* radius
        using ckdtree.

        Radius:
            r_n = min{ gamma * (log n / n)^(1/d), eta }
        with d = 2 and eta tied to DEFAULT_STEP_SIZE.
        """
        if self.kdtree is None:
            self.build_kdtree()

        # Dynamic radius per RRT* theory
        radius = self.connection_radius()

        # Query neighbors within radius
        idxs = self.kdtree.query_ball_point([node.x, node.y], r=radius)

        neighbors = []
        seen = set()

        for idx in idxs:
            n = self.node_list[idx]
            if n is node:
                continue

            node_signature = (n.x, n.y, n.theta,
                              round(n.cost, 3),
                              round(n.p_fail, 5))
            if node_signature not in seen:
                neighbors.append(n)
                seen.add(node_signature)

        return neighbors

    
    def is_descendant(self, ancestor, node):
        """Return True if node is in the subtree rooted at ancestor."""
        stack = [ancestor]
        while stack:
            cur = stack.pop()
            if cur is node:
                return True
            stack.extend(cur.children)
        return False
    
    def rebuild_path_for_node(self, node):
        """
        Rebuild the path from root to this node and update references.
        Ensures that nodes on the new path are removed from any old paths
        and that the new path is tracked in self.paths.
        """
        # 1) Collect nodes from root → node
        stack = []
        cur = node
        while cur:
            stack.append(cur)
            cur = cur.parent
        new_nodes = list(reversed(stack))  # root -> node

        # 2) Remove these nodes from any existing paths
        for n in new_nodes:
            old_path = n.path
            if old_path is not None and old_path in self.paths:
                if n in old_path.nodes:
                    old_path.nodes.remove(n)

        # 3) Build a new path and re-assign .path
        new_path = Path()
        for n in new_nodes:
            new_path.add_node(n)  # also sets n.path = new_path

        # 4) Register this path in the tree
        self.paths.append(new_path)
        self.path_count += 1

        return new_path


    def pareto_dominates(self, cost1, fail1, cost2, fail2):
        """
        Check if node1 dominates node2 in terms of cost and failure probability.
        """
        return (cost1 <= cost2 and fail1 < fail2) or \
        (cost1 < cost2 and fail1  <= fail2)
    
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
            if not is_edge_collision_free(potential_parent, test_node, grid, num_samples=10, p_threshold=0.9):
                continue
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
    
    def rewire(self, znear, nn, grid):
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
            if not is_edge_collision_free(nn, z, grid, num_samples=10, p_threshold=0.9):
                continue
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
            
            if self.is_descendant(z, nn):
                continue  # Prevent cycles
            
            strictly_dominant = self.pareto_dominates(new_cost, new_p_fail, z.cost, z.p_fail)

            if strictly_dominant:
                # Detach neighbor from old parent and old path
                old_parent = z.parent
                if old_parent:
                    if z in old_parent.children:
                        old_parent.children.remove(z)
                    z.parent = None

                # Attach neighbor under new_node
                z.parent = nn
                nn.children.append(z)

                # Update the neighbor’s metrics
                z.cost = new_cost
                z.log_survival = new_log_survival
                z.p_fail = new_p_fail

                # Rebuild ordered path
                self.rebuild_path_for_node(z)

                # Propagate down the subtree
                self.propagate_cost(z, grid)
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
                self.propagate_cost(new_z, grid)
                self.rewire_counts += 1

    def propagate_cost(self, root, grid):
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
        return (
            bool(self.nodes)
            and self.nodes[0].is_start
            and self.nodes[-1].is_goal
            and sum(1 for n in self.nodes if n.is_goal) == 1
        )
    
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
def PO_RRT_Star(start, goal, grid, max_iter):
    # Initialize the tree and nodes
    start_node, goal_node = Node(*start), Node(*goal)
    tree = Tree(grid)
    tree.add_node(start_node)
    tree.start_node = start_node
    goal_node.is_goal = True
    start_node.is_start = True
    multiple_paths = []
    goal_tracker = set()

    
    # 3D figure
    fig3d, ax3d, lc3d, edge_segments3d = init_progress_plot_3d(
        start,  
        goal,
        x_lim=(0, grid.width),
        y_lim=(0, grid.height),
        obstacles=grid.obstacles,
        z_lim=(0.0, 1.0),
    )

    # 2D figure
    fig2d, ax2d, lc2d, edge_segments2d = init_progress_plot_2d(
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
            if is_collision_free(new_node, grid): # Check if the new node is collision-free              
                znear = tree.neighbors(new_node)

                # Exclude the goal node from znear to prevent rewiring through it
                znear = [n for n in znear if n.x != goal_node.x or n.y != goal_node.y]
                
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
                            # connect to goal exactly once per branch
                                goal_instance = Node(goal[0], goal[1], goal[2])
                                goal_instance.is_goal = True
                                goal_instance.parent = nn
                                nn.children.append(goal_instance)
                                tree.add_node(nn, multiple_children=multiple_children)
                                tree.rewire(tree.neighbors(nn), nn, grid)


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
                        else:
                            tree.add_node(nn, multiple_children=multiple_children)
                            tree.rewire(tree.neighbors(nn), nn, grid)
                else:
                    # single child branch
                    for nn in new_nodes:
                        multiple_children = True if len(nn.parent.children) > 1 else False
                        if distance_to(nn, goal) <= DEFAULT_STEP_SIZE:
                                # ─── goal handling ─────────────────────────
                                goal_instance = Node(goal[0], goal[1], goal[2])
                                goal_instance.is_goal = True
                                goal_instance.parent = nn
                                nn.children.append(goal_instance)
                                tree.add_node(nn, multiple_children=multiple_children)
                                tree.rewire(tree.neighbors(nn), nn, grid)


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
                        # 2) Not near goal, so add the new node to the tree
                        else:
                            # Add the new node to the tree
                            # multiple_children = True if len(nn.parent.children) > 1 else False
                            tree.add_node(nn, multiple_children=multiple_children)
                            tree.rewire(tree.neighbors(nn), nn, grid)

                    
                        # redraw_tree_2d(tree, lc2d, edge_segments2d, highlighted_paths=None)

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
    
    
    MAX_ALLOWED_STEP = DEFAULT_STEP_SIZE + 1e-3  # Small epsilon

    print("\n--- Debug: Paths from start to goal ---")
    for idx, entry in enumerate(multiple_paths):
        path = entry["path"]
        nodes = path.nodes

        print(f"\nPath {idx+1} (Cost: {entry['cost']:.2f}, P_fail: {entry['p_fail']:.4f}):")
        for i in range(len(nodes) - 1):
            a, b = nodes[i], nodes[i+1]
            step_dist = distance_to(a, b)

            # DEBUG JUMP
            if step_dist > MAX_ALLOWED_STEP:
                print(f"    ILLEGAL JUMP DETECTED between nodes {i} and {i+1}:")
                print(f"    From: (x={a.x:.2f}, y={a.y:.2f})")
                print(f"    To:   (x={b.x:.2f}, y={b.y:.2f})")
                print(f"    Distance: {step_dist:.2f} > allowed {MAX_ALLOWED_STEP:.2f}")
                print("Illegal jump in path — investigate tree structure or rewire logic.")

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

    # list of Path objects you want to highlight
    highlight_paths = [entry["path"] for entry in filtered_paths]

    # draw 3D tree + green complete paths
    redraw_tree(tree, lc3d, edge_segments3d, highlighted_paths=highlight_paths)

    # draw 2D tree + green complete paths
    redraw_tree_2d(tree, lc2d, edge_segments2d, highlighted_paths=highlight_paths)

    # debug_filename = input("Tree debug filename [porrt_tree_debug.json]: ").strip() or "porrt_tree_debug.json"
    # save_tree_debug_json(debug_filename, tree)

    return filtered_paths, multiple_paths, tree

##################################
## CENTRAL PO_RRT_STAR FUNCTION ##
##################################


# ----------------------- #
#   Post-processing: clustering
# ----------------------- #
def cluster_paths(path_entries, cost_tol=1.0, p_fail_tol=0.05):
    """
    Cluster similar paths using only cost and p_fail tolerances.

    Inputs:
      - path_entries: list of dicts {"path": Path, "cost": float, "p_fail": float}
      - cost_tol: maximum absolute cost difference
      - p_fail_tol: maximum absolute p_fail difference

    Returns: list of clusters, each cluster is a dict:
      {"members": [entry,...], "representative": entry}

    Representative is chosen as the member with lowest cost (tie-breaker: lower p_fail).
    """
    remaining = list(path_entries)
    clusters = []

    def similar(a, b):
        cost_sim = abs(a['cost'] - b['cost']) <= cost_tol
        p_sim = abs(a['p_fail'] - b['p_fail']) <= p_fail_tol
        return cost_sim and p_sim

    while remaining:
        seed = remaining.pop(0)
        cluster = [seed]
        to_remove = []
        for other in remaining:
            if similar(seed, other):
                cluster.append(other)
                to_remove.append(other)
        # purge removed
        for r in to_remove:
            remaining.remove(r)

        # choose representative
        rep = min(cluster, key=lambda e: (e['cost'], e['p_fail']))
        clusters.append({'members': cluster, 'representative': rep})

    return clusters


def summarize_clusters(clusters):
    """Print a short summary for clusters."""
    print(f"Found {len(clusters)} clusters")
    for i, cl in enumerate(clusters, start=1):
        members = cl['members']
        rep = cl['representative']
        print(f"Cluster {i}: {len(members)} members | repr cost={rep['cost']:.4f}, p_fail={rep['p_fail']:.6f}")


def interactive_postprocess(filtered_paths, multiple_paths, obstacles=None):
    """
    Simple CLI interactive post-processing for clustering. Returns clusters.
    """
    if not filtered_paths:
        print("No filtered paths to post-process.")
        return []

    print("\nPost-process (cluster similar paths by cost & p_fail)")
    try:
        cost_tol = float(input("cost tolerance [1.0]: ").strip() or 1.0)
    except ValueError:
        cost_tol = 1.0
    try:
        p_fail_tol = float(input("p_fail tolerance [0.05]: ").strip() or 0.05)
    except ValueError:
        p_fail_tol = 0.05

    clusters = cluster_paths(filtered_paths, cost_tol=cost_tol, p_fail_tol=p_fail_tol)
    summarize_clusters(clusters)

    # Optionally plot cluster representatives
    do_plot = input("Plot cluster representatives? (y/N): ").strip().lower() == 'y'
    if do_plot and clusters:
        reps = [c['representative'] for c in clusters]
        try:
            plot_paths_summary(reps, obstacles=obstacles)
        except Exception as e:
            print(f"Plotting failed: {e}")

    return clusters

# ----------------------- #
#   Post-processing: clustering
# ----------------------- #



# Main code
def main():
    
    # Create main application window
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    obstacles = []

    
    start, goal = (3, 95, 0), (80, 50, 0)

    # Obstacle dictionary
    obstacles = [
        # {"type": "circular", "center": (50, 80), "radius": 10, "safe_dist": 5},
        # {"type": "rectangular", "x_range": (20, 80), "y_range": (20, 80), "probability": 0.05},
        # {"type": "rectangular", "x_range": (30, 90), "y_range": (20, 30), "probability": 0.05},
        # {"type": "rectangular", "x_range": (10, 60), "y_range": (40, 50), "probability": 0.07},
        {"type": "circular", "center": (50, 50), "radius": 25, "safe_dist": 7},
        {"type": "circular", "center": (0, 50), "radius": 10, "safe_dist": 4},
        {"type": "circular", "center": (100, 50), "radius": 10, "safe_dist": 4}
    ]


    # Ask for an integer number of samples. askinteger returns an int or None.
    sample_count = simpledialog.askinteger(
        "Input",
        "Enter how many samples you'd like to generate:",
        initialvalue=3000,
        minvalue=1,
    )
    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

    filtered_paths, multiple_paths, tree = PO_RRT_Star(start, goal, grid, sample_count)
    
    # Default plotting (ask user)

    # plot_final_tree_2d(
    #     tree=tree,
    #     filtered_paths=filtered_paths,
    #     grid=grid,
    #     obstacles=grid.obstacles,
    #     max_highlight_paths=10,
    #     title="PORRT* Final Tree with Pareto Paths"
    # )

    do_plot = input("Plot filtered paths and all paths? (filtered/all/none) [filtered]: ").strip().lower() or 'filtered'
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
    

    # Spectral clustering visualization
    do_spec = input("Open spectral clustering visualization? (y/N): ").strip().lower() == 'y'
    if do_spec:
        try:
            clusters, labels, dbg = spectral_cluster_paths(filtered_paths, n_clusters="auto")
            print(f"[spectral] chose k={dbg.get('k')} (sklearn={dbg.get('used_sklearn')})")
            # interactive exploration with sliders for k & weights
            interactive_spectral_cluster_plot(filtered_paths, spectral_cluster_paths, obstacles=obstacles)
        except Exception as e:
            print(f"Spectral visualization failed: {e}")

    # option to save everything for later replay (paths + whole tree)
    do_save = input("Export run to JSON for replay? (y/N): ").strip().lower() == 'y'
    if do_save:
        default_name = f"porrt_export_{int(time.time())}.json"
        fname = input(f"Output filename [{default_name}]: ").strip()
        outfile = fname or default_name
        save_run_json(outfile, start, goal, grid, filtered_paths, multiple_paths, edge_segments)

if __name__ == '__main__':
    main()
