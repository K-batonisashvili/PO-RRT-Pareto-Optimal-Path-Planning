import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from PO_RRT_Star_Theoretical import PO_RRT_Star, Grid, GRID_WIDTH, GRID_HEIGHT
from RRT_STAR import run_scalar_rrt

class MockPath:
    def __init__(self, nodes):
        self.nodes = nodes

def _draw_obstacles(ax, obstacles, width, height):
    """Helper to draw obstacles on a specific matplotlib axis."""
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    if not obstacles: return

    cmap = cm.get_cmap('YlOrRd')
    def occ_color(p):
        p_clipped = min(max(float(p), 0.01), 1.0)
        return cmap(0.40 + 0.60 * ((p_clipped - 0.01) / 0.99))

    for obs in obstacles:
        if obs["type"] == "circular":
            cx, cy = obs["center"]
            radius = obs["radius"]
            theta = np.linspace(0, 2 * np.pi, 100)
            x = cx + radius * np.cos(theta)
            y = cy + radius * np.sin(theta)
            ax.plot(x, y, '--', color=occ_color(obs.get("probability", 1.0)), alpha=0.7)
        elif obs["type"] == "rectangular":
            x0, x1 = obs["x_range"]
            y0, y1 = obs["y_range"]
            ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], '--', 
                    color=occ_color(obs.get("probability", 0.05)), alpha=0.7)

def plot_side_by_side(po_paths, baseline_results, obstacles, start, goal):
    """Generates a 1x3 subplot: Scatter, PO-RRT spatial, Baseline spatial."""
    # Increased figure height to accommodate detailed legends below
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
    plt.subplots_adjust(bottom=0.3) 
    
    # --- Panel 1: Cost vs P_fail Scatter ---
    ax1.set_title("Pareto Front & Baseline Outcomes")
    ax1.set_xlabel("Euclidean Distance")
    ax1.set_ylabel("Probability of Failure")
    ax1.grid(True)
    
    po_cmap = cm.get_cmap('tab10', max(1, len(po_paths)))
    for idx, p in enumerate(po_paths):
        ax1.scatter(p['cost'], p['p_fail'], color=po_cmap(idx), s=80, marker='o')
        
    base_cmap = cm.get_cmap('Set2', max(1, len(baseline_results)))
    for idx, b in enumerate(baseline_results):
        ax1.scatter(b['cost'], b['p_fail'], color=base_cmap(idx), s=150, marker='X', edgecolor='black')

    # --- Panel 2: PO-RRT* Spatial Paths ---
    ax2.set_title(f"Top {len(po_paths)} PO-RRT* Paths")
    _draw_obstacles(ax2, obstacles, GRID_WIDTH, GRID_HEIGHT)
    
    ax2.scatter(start[0], start[1], c='green', s=80, zorder=5, label='Start')
    ax2.scatter(goal[0], goal[1], c='red', marker='*', s=120, zorder=5, label='Goal')
    
    for idx, p in enumerate(po_paths):
        nodes = p["path"].nodes
        xs, ys = [n.x for n in nodes], [n.y for n in nodes]
        # Inject metrics directly into the label
        lbl = f"PO-RRT {idx+1} (d:{p['cost']:.1f}, pf:{p['p_fail']:.3f})"
        ax2.plot(xs, ys, color=po_cmap(idx), lw=2, alpha=0.8, label=lbl)

    ax2.legend(bbox_to_anchor=(0, -0.15), loc='upper left', ncol=1, fontsize=9)

    # --- Panel 3: Baseline RRT* Spatial Paths ---
    ax3.set_title("Scalar Baseline RRT* Paths")
    _draw_obstacles(ax3, obstacles, GRID_WIDTH, GRID_HEIGHT)
    
    ax3.scatter(start[0], start[1], c='green', s=80, zorder=5)
    ax3.scatter(goal[0], goal[1], c='red', marker='*', s=120, zorder=5)

    for idx, b in enumerate(baseline_results):
        nodes = b["path"].nodes
        xs, ys = [n.x if hasattr(n, 'x') else n[0] for n in nodes], [n.y if hasattr(n, 'y') else n[1] for n in nodes]
        # Inject metrics directly into the label
        lbl = f"{b['label']} (d:{b['cost']:.1f}, pf:{b['p_fail']:.3f})"
        ax3.plot(xs, ys, color=base_cmap(idx), lw=2.5, alpha=0.9, label=lbl)
        
    ax3.legend(bbox_to_anchor=(0, -0.15), loc='upper left', ncol=1, fontsize=9)

    plt.show(block=True)

def main():
    start = (3, 99)
    goal = (80, 1)
    obstacles = [
        {"type": "circular", "center": (50, 65), "radius": 15, "safe_dist": 5},
        {"type": "rectangular", "x_range": (30, 70), "y_range": (20, 40), "probability": 0.07},
    ]

    sample_count = 3500
    p_fail_threshold = 0.99
    grid = Grid(GRID_WIDTH, GRID_HEIGHT, obstacles)

    print(f"\n--- Running Benchmarks ({sample_count} iterations) ---\n")

    # 1. Run PO-RRT* (Array-Based)
    print("Running PO-RRT*...")
    t0 = time.perf_counter()
    filtered_po, _, tree_po, _ = PO_RRT_Star(start, goal, grid, max_iter=sample_count, p_fail_threshold=p_fail_threshold)
    time_po = time.perf_counter() - t0
    plt.close('all') 

    # Filter to top 10 evenly distributed paths
    top_po_paths = []
    if filtered_po:
        filtered_po.sort(key=lambda x: x["p_fail"])
        if len(filtered_po) > 10:
            indices = np.linspace(0, len(filtered_po) - 1, 10, dtype=int)
            top_po_paths = [filtered_po[i] for i in indices]
        else:
            top_po_paths = filtered_po

    # 2. Run Scalar Baselines
    baseline_results = []
    
    print("Running Baseline RRT* (Distance Focus)...")
    t1 = time.perf_counter()
    res_dist = run_scalar_rrt(start, goal, grid, max_iter=sample_count, mode="distance")
    if res_dist["success"]:
        baseline_results.append({
            "path": MockPath(res_dist["path"]), "cost": res_dist["cost"], 
            "p_fail": res_dist["p_fail"], "label": "Baseline (Distance)", "time": time.perf_counter() - t1
        })

    print("Running Baseline RRT* (Risk Focus)...")
    t2 = time.perf_counter()
    res_risk = run_scalar_rrt(start, goal, grid, max_iter=sample_count, mode="risk")
    if res_risk["success"]:
        baseline_results.append({
            "path": MockPath(res_risk["path"]), "cost": res_risk["cost"], 
            "p_fail": res_risk["p_fail"], "label": "Baseline (Risk)", "time": time.perf_counter() - t2
        })

    weights_w1 = [0.25, 0.1, 0.09, 0.08, 0.07, 0.06,0.05]
    for w1 in weights_w1:
        w2 = 1.0 - w1
        print(f"Running Baseline RRT* (Weighted: w1={w1}, w2={w2})...")
        tw = time.perf_counter()
        res_weight = run_scalar_rrt(start, goal, grid, max_iter=sample_count, mode="weighted", w1=w1, w2=w2)
        if res_weight["success"]:
            baseline_results.append({
                "path": MockPath(res_weight["path"]), "cost": res_weight["cost"], 
                "p_fail": res_weight["p_fail"], "label": f"Weighted (w1={w1})", "time": time.perf_counter() - tw
            })

    # --- Formatting Output ---
    print("\n=========================================================================")
    print(f"{'PLANNER / PATH':<25} | {'COST':<10} | {'P_FAIL':<10} | {'RUNTIME (s)':<10}")
    print("=========================================================================")
    
    for i, entry in enumerate(top_po_paths):
        print(f"PO-RRT* (Path {i+1}):{'':<8} | {entry['cost']:<10.2f} | {entry['p_fail']:<10.4f} | {time_po if i==0 else '':<10}")
        
    print("-" * 73)

    for b in baseline_results:
        print(f"{b['label']:<25} | {b['cost']:<10.2f} | {b['p_fail']:<10.4f} | {b['time']:<10.2f}")

    print("=========================================================================\n")

    plot_side_by_side(top_po_paths, baseline_results, obstacles, start, goal)

if __name__ == '__main__':
    main()