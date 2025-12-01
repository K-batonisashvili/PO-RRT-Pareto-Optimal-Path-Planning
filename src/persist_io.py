# persist_io.py
import json, time, numpy as np
from dataclasses import dataclass

# --- Minimal "replay" objects so we don't need the original classes on load ---
@dataclass
class ReplayNode:
    x: float; y: float; theta: float
    cost: float; log_survival: float; p_fail: float

@dataclass
class ReplayPath:
    nodes: list  # list[ReplayNode]
    @property
    def cost(self): return self.nodes[-1].cost if self.nodes else 0.0
    @property
    def p_fail(self): return self.nodes[-1].p_fail if self.nodes else 0.0

def _pack(tree, grid, start, goal, extra_meta=None):
    """
    Convert your in-memory objects to a JSON-serializable dict + raw grid array.
    Assumes: tree.node_list, each node has .parent, .children, .x,.y,.theta,.cost,.log_survival,.p_fail
             tree.paths is a list of Path objects with .nodes list.
             grid has .width, .height, .grid (2D ndarray), .obstacles (list[dict])
    """
    node_ids = {id(n): i for i, n in enumerate(tree.node_list)}
    nodes = []
    for n in tree.node_list:
        nodes.append({
            "id": node_ids[id(n)],
            "x": float(n.x), "y": float(n.y), "theta": float(n.theta),
            "cost": float(n.cost), "log_survival": float(n.log_survival), "p_fail": float(n.p_fail),
            "parent": (node_ids[id(n.parent)] if getattr(n, "parent", None) else None),
        })

    edges = [[node_ids[id(n.parent)], node_ids[id(n)]] for n in tree.node_list if getattr(n, "parent", None)]
    paths = []
    for p in getattr(tree, "paths", []):
        paths.append([node_ids[id(n)] for n in p.nodes])

    data = {
        "meta": {
            "saved_at_unix": time.time(),
            "start": list(start), "goal": list(goal),
            "params": {
                # add any constants you want tracked for reference:
                # "PARETO_RADIUS": PARETO_RADIUS, "DEFAULT_STEP_SIZE": DEFAULT_STEP_SIZE, ...
            },
            **(extra_meta or {})
        },
        "grid": {
            "width": int(grid.width), "height": int(grid.height),
            "obstacles": grid.obstacles,  # serializable as-is (list of dicts)
            "shape": list(grid.grid.shape)
        },
        "nodes": nodes,
        "edges": edges,    # parent-child list; nice for debugging/graph draws
        "paths": paths     # list of node-id sequences
    }
    return data, grid.grid

def save_run_npz(path, tree, grid, start, goal, extra_meta=None):
    """Save a planning run to a single compressed file."""
    data, grid_array = _pack(tree, grid, start, goal, extra_meta=extra_meta)
    # Keep grid as ndarray (fast/compact) and dump the rest as JSON text:
    np.savez_compressed(path, meta_json=json.dumps(data), grid=grid_array)

def load_run_npz(path, reify=True):
    """
    Load a saved run. By default returns Replay* objects that your visualization can use.
    Returns: dict with keys: grid, nodes, paths, edges, meta
    """
    bundle = np.load(path, allow_pickle=True)
    meta = json.loads(bundle["meta_json"].item() if hasattr(bundle["meta_json"], "item") else bundle["meta_json"])
    grid_arr = bundle["grid"]

    # Reconstruct grid-like container (just what plotting needs)
    class ReplayGrid:
        pass
    g = ReplayGrid()
    g.width  = meta["grid"]["width"]
    g.height = meta["grid"]["height"]
    g.obstacles = meta["grid"]["obstacles"]
    g.grid = grid_arr  # if you want to re-draw risk carpets later

    # Reconstruct nodes
    raw_nodes = meta["nodes"]
    nodes = [ReplayNode(n["x"], n["y"], n["theta"], n["cost"], n["log_survival"], n["p_fail"]) for n in raw_nodes]

    # Reconstruct paths
    paths = []
    for p_idx_list in meta["paths"]:
        p_nodes = [nodes[i] for i in p_idx_list]
        paths.append(ReplayPath(p_nodes))

    return {
        "grid": g,
        "nodes": nodes,
        "paths": paths,
        "edges": meta["edges"],
        "meta": meta["meta"],
        "start": tuple(meta["meta"]["start"]),
        "goal":  tuple(meta["meta"]["goal"]),
    }
