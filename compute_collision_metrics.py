"""
Comprehensive collision resolution metrics for DiffuScene.

Reads:
  - {room}/collision_params_merged.json    (BEFORE resolve)
  - {room}/collision_params_resolved.json  (AFTER resolve)

Produces:
  - collision_metrics_report.json          (machine-readable)
  - collision_metrics_report.md            (paper-ready markdown table)

Metrics implemented:
  1. Penetration Depth Metrics (before resolve)
  2. Convergence / Iteration Metrics
  3. Displacement Cost Metrics (before vs after)
  4. Resolution Success Rate
  5. Test set size reporting
  6. Stress-test scenario (separate function)

Reuses SAT logic from resolve_collision.py.
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

CATEGORY_TO_PROCESSED = {
    "bedroom":    "bedrooms_objfeats_32_64",
    "livingroom": "livingrooms_objfeats_32_64",
    "diningroom": "diningrooms_objfeats_32_64",
}
def _extract_base_id(scene_id):
    import re
    m = re.match(r'^(.+?-\d+)_\d+_\d+$', scene_id)
    if m: return m.group(1)
    return scene_id

def _build_room_bounds_cache(category, base_dir):
    processed_subdir = CATEGORY_TO_PROCESSED.get(category)
    if processed_subdir is None: return {}
    cat_dir = os.path.join(base_dir, "3d_front_processed", processed_subdir)
    cache = {}
    if not os.path.isdir(cat_dir): return cache
    import numpy as np
    for d in os.listdir(cat_dir):
        boxes_path = os.path.join(cat_dir, d, "boxes.npz")
        if not os.path.isfile(boxes_path): continue
        parts = d.split("_", 1)
        base_id = parts[1] if len(parts) == 2 else d
        if base_id in cache: continue
        try:
            data = np.load(boxes_path, allow_pickle=True)
            if "floor_plan_vertices" not in data: continue
            vertices = data["floor_plan_vertices"]
            xs = vertices[:, 0]
            zs = vertices[:, 2]
            cxL = 0.0
            czL = 0.0
            hxL = float((xs.max() - xs.min()) / 2.0)
            hzL = float((zs.max() - zs.min()) / 2.0)
            axL = np.array([1.0, 0.0])
            azL = np.array([0.0, 1.0])
            cache[base_id] = (cxL, czL, hxL, hzL, axL, azL)
        except Exception: pass
    return cache


# Import OBB collision primitives from existing codebase
from resolve_collision import _obb_corners_xz, _sat_overlap, _obb_sat_xz
from resolve_collision import resolve_bbox_collisions_obb


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# HELPER: compute per-pair penetration depth using SAT (reuses existing logic)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def compute_pairwise_penetrations(sizes, trans, angles_deg):
    """
    For a single scene, compute penetration depths for ALL colliding pairs.

    Returns list of dicts:
      [{"i": int, "j": int, "depth": float,
        "avg_obj_size": float, "depth_pct": float}, ...]
    """
    N = len(sizes)
    results = []
    for a in range(N):
        for b in range(a + 1, N):
            # Vertical check
            hy_a = float(sizes[a][1]) / 2.0
            cy_a = float(trans[a][1])
            hy_b = float(sizes[b][1]) / 2.0
            cy_b = float(trans[b][1])

            if (cy_a + hy_a) <= (cy_b - hy_b) or (cy_b + hy_b) <= (cy_a - hy_a):
                continue

            box_a = [sizes[a][0], sizes[a][1], sizes[a][2],
                     trans[a][0], trans[a][1], trans[a][2]]
            box_b = [sizes[b][0], sizes[b][1], sizes[b][2],
                     trans[b][0], trans[b][1], trans[b][2]]

            ang_a = float(np.array(angles_deg[a]).reshape(-1)[0])
            ang_b = float(np.array(angles_deg[b]).reshape(-1)[0])

            is_col, depth, _ = _obb_sat_xz(box_a, ang_a, box_b, ang_b)

            if is_col and depth > 1e-4:
                # Average object footprint size (geometric mean of l and w)
                size_a = math.sqrt(float(sizes[a][0]) * float(sizes[a][2]))
                size_b = math.sqrt(float(sizes[b][0]) * float(sizes[b][2]))
                avg_size = (size_a + size_b) / 2.0
                depth_pct = (depth / avg_size) if avg_size > 1e-6 else 0.0

                results.append({
                    "i": a, "j": b,
                    "depth": depth,
                    "avg_obj_size": avg_size,
                    "depth_pct": depth_pct,
                })
    return results


def has_any_collision(sizes, trans, angles_deg):
    """Quick check: does this scene have at least one collision?"""
    N = len(sizes)
    for a in range(N):
        for b in range(a + 1, N):
            hy_a = float(sizes[a][1]) / 2.0
            cy_a = float(trans[a][1])
            hy_b = float(sizes[b][1]) / 2.0
            cy_b = float(trans[b][1])
            if (cy_a + hy_a) <= (cy_b - hy_b) or (cy_b + hy_b) <= (cy_a - hy_a):
                continue
            box_a = [sizes[a][0], sizes[a][1], sizes[a][2],
                     trans[a][0], trans[a][1], trans[a][2]]
            box_b = [sizes[b][0], sizes[b][1], sizes[b][2],
                     trans[b][0], trans[b][1], trans[b][2]]
            ang_a = float(np.array(angles_deg[a]).reshape(-1)[0])
            ang_b = float(np.array(angles_deg[b]).reshape(-1)[0])
            is_col, depth, _ = _obb_sat_xz(box_a, ang_a, box_b, ang_b)
            if is_col and depth > 1e-4:
                return True
    return False


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MODIFIED SOLVER: returns iteration count + convergence flag
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def resolve_with_logging(sizes_list, trans_list, angles_rad_list, class_labels_list, room_bounds=None,
                         max_iter=500, push_eps=0.02):
    """
    Run the OBB collision solver on a single scene and return:
      - resolved translations
      - iterations_used (int)
      - converged (bool)
    """
    sizes = torch.tensor(sizes_list, dtype=torch.float32)
    trans = torch.tensor(trans_list, dtype=torch.float32)
    angles_rad = torch.tensor(angles_rad_list, dtype=torch.float32)
    angles_deg = angles_rad * (180.0 / np.pi)

    boxes = torch.cat([sizes, trans], dim=-1)

    cl = None
    if class_labels_list is not None:
        cl = np.array(class_labels_list, dtype=np.float32)

    # --- inline the solver loop to capture iteration count ---
    boxes_np = boxes.detach().cpu().numpy().astype(float)
    angles_np = angles_deg.detach().cpu().numpy().flatten().astype(float)
    N = boxes_np.shape[0]

    labels_idx = None
    if cl is not None:
        labels_idx = np.argmax(cl, axis=-1)

    layout_indices = []
    main_layout_idx = -1

    converged = False
    iterations_used = max_iter

    for iteration in range(max_iter):
        moved = False

        if room_bounds is not None:
            pass
        elif main_layout_idx != -1:
            L_i = main_layout_idx
            l_L, w_L = float(boxes_np[L_i, 0]), float(boxes_np[L_i, 2])
            cxL, czL = float(boxes_np[L_i, 3]), float(boxes_np[L_i, 5])
            ang_L = angles_np[L_i]
            theta_L = np.radians(ang_L)
            axL = np.array([np.cos(theta_L), np.sin(theta_L)])
            azL = np.array([-np.sin(theta_L), np.cos(theta_L)])
            hxL, hzL = l_L / 2.0, w_L / 2.0
            room_bounds = (cxL, czL, hxL, hzL, axL, azL)

        # Layout containment
        if room_bounds is not None:
            cxL, czL, hxL, hzL, axL, azL = room_bounds
            for i in range(N):
                if i in layout_indices:
                    continue
                corners_i = _obb_corners_xz(
                    float(boxes_np[i, 3]), float(boxes_np[i, 5]),
                    float(boxes_np[i, 0]), float(boxes_np[i, 2]), angles_np[i])
                c_rel = corners_i - np.array([cxL, czL])
                proj_x = c_rel @ axL
                proj_z = c_rel @ azL
                min_x, max_x = proj_x.min(), proj_x.max()
                min_z, max_z = proj_z.min(), proj_z.max()
                shift_x = 0.0
                if (max_x - min_x) > 2 * hxL:
                    shift_x = -(max_x + min_x) / 2.0
                else:
                    if max_x > hxL:
                        shift_x = hxL - max_x
                    elif min_x < -hxL:
                        shift_x = -hxL - min_x
                shift_z = 0.0
                if (max_z - min_z) > 2 * hzL:
                    shift_z = -(max_z + min_z) / 2.0
                else:
                    if max_z > hzL:
                        shift_z = hzL - max_z
                    elif min_z < -hzL:
                        shift_z = -hzL - min_z
                if abs(shift_x) > 1e-4 or abs(shift_z) > 1e-4:
                    boxes_np[i, 3] += shift_x * axL[0] + shift_z * azL[0]
                    boxes_np[i, 5] += shift_x * axL[1] + shift_z * azL[1]
                    moved = True

        # Object-object collisions
        for i in range(N):
            if i in layout_indices:
                continue
            for j in range(i + 1, N):
                if j in layout_indices:
                    continue
                hy_i = float(boxes_np[i, 1]) / 2.0
                cy_i = float(boxes_np[i, 4])
                hy_j = float(boxes_np[j, 1]) / 2.0
                cy_j = float(boxes_np[j, 4])
                if (cy_i + hy_i) <= (cy_j - hy_j) or (cy_j + hy_j) <= (cy_i - hy_i):
                    continue
                colliding, depth, push_axis = _obb_sat_xz(
                    boxes_np[i], angles_np[i],
                    boxes_np[j], angles_np[j],
                    room_bounds=room_bounds
                )
                if colliding and push_axis is not None:
                    moved = True
                    shift = (depth + push_eps) / 2.0
                    boxes_np[i, 3] -= shift * push_axis[0]
                    boxes_np[i, 5] -= shift * push_axis[1]
                    boxes_np[j, 3] += shift * push_axis[0]
                    boxes_np[j, 5] += shift * push_axis[1]

        if not moved:
            converged = True
            iterations_used = iteration + 1
            break

    # Extract resolved translations
    res_trans = boxes_np[:, 3:].tolist()
    return res_trans, iterations_used, converged


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# METRIC 1: Penetration Depth Metrics (before resolve)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def compute_penetration_metrics(data_before):
    """
    Compute per-scene and aggregate penetration depth metrics
    from the BEFORE-resolve data.
    """
    n_scenes = len(data_before["scene_ids"])

    per_scene_total_depth = []
    per_scene_max_depth = []
    per_scene_mean_depth = []
    per_scene_mean_pct = []
    all_depths = []
    all_pcts = []
    scenes_with_collisions = 0

    for idx in range(n_scenes):
        sizes = np.array(data_before["sizes"][idx], dtype=np.float32)
        trans = np.array(data_before["translations"][idx], dtype=np.float32)
        angles_rad = np.array(data_before["angles"][idx], dtype=np.float32)
        angles_deg = angles_rad * (180.0 / np.pi)

        pens = compute_pairwise_penetrations(sizes, trans, angles_deg)

        if len(pens) > 0:
            scenes_with_collisions += 1
            depths = [p["depth"] for p in pens]
            pcts = [p["depth_pct"] for p in pens]
            per_scene_total_depth.append(sum(depths))
            per_scene_max_depth.append(max(depths))
            per_scene_mean_depth.append(np.mean(depths))
            per_scene_mean_pct.append(np.mean(pcts))
            all_depths.extend(depths)
            all_pcts.extend(pcts)
        else:
            per_scene_total_depth.append(0.0)
            per_scene_max_depth.append(0.0)
            per_scene_mean_depth.append(0.0)
            per_scene_mean_pct.append(0.0)

    result = {
        "n_scenes": n_scenes,
        "n_scenes_with_collisions": scenes_with_collisions,
        "total_penetration_depth_mean": float(np.mean(per_scene_total_depth)),
        "total_penetration_depth_std": float(np.std(per_scene_total_depth)),
        "mean_penetration_depth_per_collision": float(np.mean(all_depths)) if all_depths else 0.0,
        "std_penetration_depth_per_collision": float(np.std(all_depths)) if all_depths else 0.0,
        "max_penetration_depth_mean": float(np.mean(per_scene_max_depth)),
        "max_penetration_depth_std": float(np.std(per_scene_max_depth)),
        "max_penetration_depth_global": float(max(per_scene_max_depth)) if per_scene_max_depth else 0.0,
        "penetration_depth_pct_mean": float(np.mean(all_pcts)) if all_pcts else 0.0,
        "penetration_depth_pct_std": float(np.std(all_pcts)) if all_pcts else 0.0,
        "total_colliding_pairs": len(all_depths),
    }
    return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# METRIC 2: Convergence / Iteration Metrics
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def compute_convergence_metrics(data_before, bounds_cache=None, max_iter=500, push_eps=0.02):
    """
    Re-run the solver on every scene, logging iterations and convergence.
    Returns per-scene logs and aggregate stats.
    """
    n_scenes = len(data_before["scene_ids"])
    iterations_list = []
    converged_list = []
    resolved_translations_all = []

    for idx in range(n_scenes):
        cl = data_before["class_labels"][idx] if "class_labels" in data_before else None
        rb = bounds_cache.get(_extract_base_id(data_before["scene_ids"][idx])) if bounds_cache else None

        res_trans, iters, conv = resolve_with_logging(
            data_before["sizes"][idx],
            data_before["translations"][idx],
            data_before["angles"][idx],
            cl,
            room_bounds=rb,
            max_iter=max_iter,
            push_eps=push_eps,
        )
        iterations_list.append(iters)
        converged_list.append(conv)
        resolved_translations_all.append(res_trans)

        if (idx + 1) % 50 == 0 or idx == n_scenes - 1:
            print(f"    Processed {idx + 1}/{n_scenes} scenes...")

    iters_arr = np.array(iterations_list)
    n_failed = sum(1 for c in converged_list if not c)

    result = {
        "n_scenes": n_scenes,
        "mean_iterations": float(np.mean(iters_arr)),
        "median_iterations": float(np.median(iters_arr)),
        "min_iterations": int(np.min(iters_arr)),
        "max_iterations": int(np.max(iters_arr)),
        "std_iterations": float(np.std(iters_arr)),
        "convergence_failure_count": n_failed,
        "convergence_failure_rate": float(n_failed / n_scenes) if n_scenes > 0 else 0.0,
        "iteration_histogram": {},
    }

    # Build histogram buckets: [1], [2-5], [6-10], [11-50], [51-100], [101-500], [500(max)]
    buckets = [(1, 1), (2, 5), (6, 10), (11, 50), (51, 100), (101, 499), (500, 500)]
    for lo, hi in buckets:
        label = f"{lo}" if lo == hi else f"{lo}-{hi}"
        count = int(np.sum((iters_arr >= lo) & (iters_arr <= hi)))
        result["iteration_histogram"][label] = count

    return result, iterations_list, converged_list, resolved_translations_all


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# METRIC 3: Displacement Cost Metrics
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def compute_displacement_metrics(data_before, resolved_translations_all):
    """
    Compare object centers BEFORE and AFTER resolve.
    Only counts objects that actually moved (displacement > 1e-5 m).
    """
    n_scenes = len(data_before["scene_ids"])

    per_scene_mean_disp = []
    per_scene_pct_moved = []
    per_scene_mean_rel_disp = []
    all_displacements = []
    all_rel_displacements = []
    total_objects = 0
    total_moved = 0

    for idx in range(n_scenes):
        trans_before = np.array(data_before["translations"][idx], dtype=np.float64)
        trans_after = np.array(resolved_translations_all[idx], dtype=np.float64)
        sizes = np.array(data_before["sizes"][idx], dtype=np.float64)

        N = len(trans_before)
        total_objects += N

        displacements = []
        rel_displacements = []
        n_moved = 0

        for k in range(N):
            # Euclidean displacement in xz plane
            dx = trans_after[k][0] - trans_before[k][0]
            dz = trans_after[k][2] - trans_before[k][2]
            disp = math.sqrt(dx * dx + dz * dz)

            if disp > 1e-5:
                n_moved += 1
                displacements.append(disp)
                all_displacements.append(disp)

                # Relative to max(l, w) of the object
                max_lw = max(float(sizes[k][0]), float(sizes[k][2]))
                if max_lw > 1e-6:
                    rel = disp / max_lw
                else:
                    rel = 0.0
                rel_displacements.append(rel)
                all_rel_displacements.append(rel)

        total_moved += n_moved
        pct_moved = (n_moved / N) if N > 0 else 0.0
        per_scene_pct_moved.append(pct_moved)

        if displacements:
            per_scene_mean_disp.append(np.mean(displacements))
            per_scene_mean_rel_disp.append(np.mean(rel_displacements))
        else:
            per_scene_mean_disp.append(0.0)
            per_scene_mean_rel_disp.append(0.0)

    result = {
        "n_scenes": n_scenes,
        "total_objects": total_objects,
        "total_objects_moved": total_moved,
        "pct_objects_moved": float(total_moved / total_objects) if total_objects > 0 else 0.0,
        "mean_displacement_m": float(np.mean(all_displacements)) if all_displacements else 0.0,
        "std_displacement_m": float(np.std(all_displacements)) if all_displacements else 0.0,
        "max_displacement_m": float(max(all_displacements)) if all_displacements else 0.0,
        "mean_displacement_relative": float(np.mean(all_rel_displacements)) if all_rel_displacements else 0.0,
        "std_displacement_relative": float(np.std(all_rel_displacements)) if all_rel_displacements else 0.0,
        "per_scene_pct_moved_mean": float(np.mean(per_scene_pct_moved)),
        "per_scene_mean_displacement_mean": float(np.mean(per_scene_mean_disp)),
    }
    return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# METRIC 4: Resolution Success Rate  +  ColObj / ColScene (standard metrics)
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def compute_collision_rates(data):
    """
    Compute standard ColObj and ColScene metrics using SAT.
    Returns dict with scene_rate, pair_rate, counts.
    """
    total_scenes = len(data["scene_ids"])
    collided_scenes = 0
    total_pairs = 0
    collided_pairs = 0

    for i in range(total_scenes):
        sizes = np.array(data["sizes"][i], dtype=np.float32)
        trans = np.array(data["translations"][i], dtype=np.float32)
        angles_rad = np.array(data["angles"][i], dtype=np.float32)
        angles_deg = angles_rad * (180.0 / np.pi)

        N = len(sizes)
        scene_has_col = False
        for a in range(N):
            for b in range(a + 1, N):
                hy_a = float(sizes[a, 1]) / 2.0
                cy_a = float(trans[a, 1])
                hy_b = float(sizes[b, 1]) / 2.0
                cy_b = float(trans[b, 1])
                if (cy_a + hy_a) <= (cy_b - hy_b) or (cy_b + hy_b) <= (cy_a - hy_a):
                    continue
                box_a = [sizes[a, 0], sizes[a, 1], sizes[a, 2],
                         trans[a, 0], trans[a, 1], trans[a, 2]]
                box_b = [sizes[b, 0], sizes[b, 1], sizes[b, 2],
                         trans[b, 0], trans[b, 1], trans[b, 2]]
                ang_a = float(angles_deg[a].reshape(-1)[0])
                ang_b = float(angles_deg[b].reshape(-1)[0])
                is_col, depth, _ = _obb_sat_xz(box_a, ang_a, box_b, ang_b)
                if is_col and depth > 1e-4:
                    scene_has_col = True
                    collided_pairs += 1
                total_pairs += 1
        if scene_has_col:
            collided_scenes += 1

    return {
        "n_scenes": total_scenes,
        "ColScene": float(collided_scenes / total_scenes) if total_scenes > 0 else 0.0,
        "ColObj": float(collided_pairs / total_pairs) if total_pairs > 0 else 0.0,
        "collided_scenes": collided_scenes,
        "total_pairs": total_pairs,
        "collided_pairs": collided_pairs,
    }


def compute_resolution_success(data_after, converged_list):
    """
    Resolution success rate = % scenes with 0 collisions after resolve.
    Cross-checked against solver convergence flag.
    """
    n = len(data_after["scene_ids"])
    n_zero_col = 0
    for i in range(n):
        sizes = np.array(data_after["sizes"][i], dtype=np.float32)
        trans = np.array(data_after["translations"][i], dtype=np.float32)
        angles_rad = np.array(data_after["angles"][i], dtype=np.float32)
        angles_deg = angles_rad * (180.0 / np.pi)
        if not has_any_collision(sizes, trans, angles_deg):
            n_zero_col += 1

    # Cross-check with solver convergence
    n_converged = sum(1 for c in converged_list if c) if converged_list else "N/A"

    return {
        "n_scenes": n,
        "resolution_success_count": n_zero_col,
        "resolution_success_rate": float(n_zero_col / n) if n > 0 else 0.0,
        "solver_converged_count": n_converged,
    }


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# METRIC 6: Stress-test scenario
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def run_stress_test(n_scenes=50, n_objs_per_scene=20, room_half_extent=2.5,
                    max_iter=500, push_eps=0.02, seed=42):
    """
    Generate synthetic layouts with forced high-density collisions.
    Objects are randomly placed within a tight bounding box to ensure
    many overlaps, then we run the solver and report convergence stats.
    """
    rng = np.random.RandomState(seed)
    print(f"\n  Stress test: {n_scenes} scenes, {n_objs_per_scene} objects each")

    iterations_list = []
    converged_list = []
    col_before_list = []
    col_after_list = []

    for s in range(n_scenes):
        # Random furniture sizes: l, h, w in [0.2, 1.5] meters
        sizes = rng.uniform(0.2, 1.5, size=(n_objs_per_scene, 3)).tolist()
        # Pack objects tightly in center region to force collisions
        trans = []
        for k in range(n_objs_per_scene):
            cx = rng.uniform(-room_half_extent * 0.6, room_half_extent * 0.6)
            cy = float(sizes[k][1]) / 2.0  # place on floor
            cz = rng.uniform(-room_half_extent * 0.6, room_half_extent * 0.6)
            trans.append([cx, cy, cz])
        # Random yaw angles in radians
        angles = [[float(rng.uniform(-math.pi, math.pi))] for _ in range(n_objs_per_scene)]

        # Count collisions before
        sizes_np = np.array(sizes, dtype=np.float32)
        trans_np = np.array(trans, dtype=np.float32)
        angles_np = np.array(angles, dtype=np.float32)
        angles_deg = angles_np * (180.0 / np.pi)
        pens_before = compute_pairwise_penetrations(sizes_np, trans_np, angles_deg)
        col_before_list.append(len(pens_before))

        # No class labels for synthetic scenes
        res_trans, iters, conv = resolve_with_logging(
            sizes, trans, angles, None,
            max_iter=max_iter, push_eps=push_eps,
        )

        iterations_list.append(iters)
        converged_list.append(conv)

        # Count collisions after
        trans_after_np = np.array(res_trans, dtype=np.float32)
        # angles are still in radians in our representation; convert for check
        pens_after = compute_pairwise_penetrations(sizes_np, trans_after_np, angles_deg)
        col_after_list.append(len(pens_after))

    iters_arr = np.array(iterations_list)
    n_failed = sum(1 for c in converged_list if not c)

    result = {
        "n_scenes": n_scenes,
        "n_objects_per_scene": n_objs_per_scene,
        "room_half_extent": room_half_extent,
        "mean_collisions_before": float(np.mean(col_before_list)),
        "mean_collisions_after": float(np.mean(col_after_list)),
        "mean_iterations": float(np.mean(iters_arr)),
        "median_iterations": float(np.median(iters_arr)),
        "max_iterations": int(np.max(iters_arr)),
        "std_iterations": float(np.std(iters_arr)),
        "convergence_failure_count": n_failed,
        "convergence_failure_rate": float(n_failed / n_scenes) if n_scenes > 0 else 0.0,
    }
    return result


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN PIPELINE
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

def process_room(room_name, room_dir, base_dir, max_iter=500, push_eps=0.02):
    """Process a single room category and return all metrics."""
    merged_path = os.path.join(room_dir, "collision_params_merged.json")
    resolved_path = os.path.join(room_dir, "collision_params_resolved.json")

    if not os.path.exists(merged_path):
        print(f"  [SKIP] {merged_path} not found")
        return None

    print(f"  Loading {merged_path}...")
    with open(merged_path, "r", encoding="utf-8") as f:
        data_before = json.load(f)

    n_scenes = len(data_before["scene_ids"])
    print(f"  {room_name}: {n_scenes} scenes")

    # â”€â”€ Metric 1: Penetration Depth (before resolve) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"  Computing penetration depth metrics (before resolve)...")
    pen_metrics = compute_penetration_metrics(data_before)
    print(f"    Total colliding pairs: {pen_metrics['total_colliding_pairs']}")
    print(f"    Mean depth per collision: {pen_metrics['mean_penetration_depth_per_collision']:.4f} m")

    # â”€â”€ Standard collision rates BEFORE â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"  Computing collision rates (before resolve)...")
    col_before = compute_collision_rates(data_before)
    print(f"    ColScene (before): {col_before['ColScene']:.4f}")
    print(f"    ColObj   (before): {col_before['ColObj']:.4f}")

    # â”€â”€ Metric 2: Convergence / Iterations (re-run solver) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"  Re-running solver with logging...")
    bounds_cache = _build_room_bounds_cache(room_name, base_dir)
    conv_metrics, iters_list, conv_list, resolved_trans_all = \
        compute_convergence_metrics(data_before, bounds_cache=bounds_cache, max_iter=max_iter, push_eps=push_eps)
    print(f"    Mean iterations: {conv_metrics['mean_iterations']:.1f}")
    print(f"    Convergence failures: {conv_metrics['convergence_failure_count']}/{n_scenes}")

    # ── Load actual resolved translations ────────────────────────
    if os.path.exists(resolved_path):
        with open(resolved_path, "r", encoding="utf-8") as f:
            data_after = json.load(f)
        resolved_trans_all = data_after["translations"]
    else:
        data_after = dict(data_before)
        data_after["translations"] = resolved_trans_all

    # â”€â”€ Metric 3: Displacement Cost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"  Computing displacement metrics...")
    disp_metrics = compute_displacement_metrics(data_before, resolved_trans_all)
    print(f"    Objects moved: {disp_metrics['total_objects_moved']}/{disp_metrics['total_objects']} "
          f"({disp_metrics['pct_objects_moved']*100:.1f}%)")
    print(f"    Mean displacement: {disp_metrics['mean_displacement_m']*100:.2f} cm")

    # â”€â”€ Standard collision rates AFTER â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    print(f"  Computing collision rates (after resolve)...")
    col_after = compute_collision_rates(data_after)
    print(f"    ColScene (after): {col_after['ColScene']:.4f}")
    print(f"    ColObj   (after): {col_after['ColObj']:.4f}")

    # â”€â”€ Metric 4: Resolution Success â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    res_success = compute_resolution_success(data_after, conv_list)
    print(f"    Resolution success rate: {res_success['resolution_success_rate']*100:.1f}%")

    # Assemble result
    return {
        "n_scenes": n_scenes,
        "penetration_depth": pen_metrics,
        "collision_rates_before": col_before,
        "collision_rates_after": col_after,
        "convergence": conv_metrics,
        "displacement": disp_metrics,
        "resolution_success": res_success,
    }


def generate_markdown_report(all_metrics, stress_metrics=None):
    """Generate a paper-ready markdown report."""
    lines = []
    lines.append("# Collision Resolution Metrics Report\n")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*\n")

    # â”€â”€ Table 1: Collision Rates (Before / After) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("## 1. Collision Rates (Before / After Resolution)\n")
    lines.append("| Room | n | ColObj (Before) | ColScene (Before) | ColObj (After) | ColScene (After) |")
    lines.append("|:-----|--:|:----:|:----:|:----:|:----:|")
    for room, m in all_metrics.items():
        lines.append(
            f"| {room} | {m['n_scenes']} | "
            f"{m['collision_rates_before']['ColObj']:.4f} | "
            f"{m['collision_rates_before']['ColScene']:.4f} | "
            f"{m['collision_rates_after']['ColObj']:.4f} | "
            f"{m['collision_rates_after']['ColScene']:.4f} |"
        )
    lines.append("")

    # â”€â”€ Table 2: Penetration Depth Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("## 2. Penetration Depth Metrics (Before Resolution)\n")
    lines.append("| Room | n | Colliding Pairs | Mean Depth (m) | Max Depth (m) | Depth/ObjSize (%) |")
    lines.append("|:-----|--:|---:|:----:|:----:|:----:|")
    for room, m in all_metrics.items():
        p = m["penetration_depth"]
        lines.append(
            f"| {room} | {m['n_scenes']} | "
            f"{p['total_colliding_pairs']} | "
            f"{p['mean_penetration_depth_per_collision']:.4f} Â± {p['std_penetration_depth_per_collision']:.4f} | "
            f"{p['max_penetration_depth_global']:.4f} | "
            f"{p['penetration_depth_pct_mean']*100:.1f} Â± {p['penetration_depth_pct_std']*100:.1f} |"
        )
    lines.append("")

    # â”€â”€ Table 3: Convergence / Iteration Metrics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("## 3. Convergence Metrics\n")
    lines.append("| Room | n | Mean Iter | Median Iter | Min | Max | Std | Failure Rate |")
    lines.append("|:-----|--:|:----:|:----:|:----:|:----:|:----:|:----:|")
    for room, m in all_metrics.items():
        c = m["convergence"]
        lines.append(
            f"| {room} | {m['n_scenes']} | "
            f"{c['mean_iterations']:.1f} | "
            f"{c['median_iterations']:.0f} | "
            f"{c['min_iterations']} | "
            f"{c['max_iterations']} | "
            f"{c['std_iterations']:.1f} | "
            f"{c['convergence_failure_rate']*100:.1f}% ({c['convergence_failure_count']}/{m['n_scenes']}) |"
        )
    lines.append("")

    # â”€â”€ Iteration Histogram â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("### Iteration Distribution\n")
    for room, m in all_metrics.items():
        hist = m["convergence"]["iteration_histogram"]
        lines.append(f"**{room}** (n={m['n_scenes']}):\n")
        lines.append("| Iterations | Count | Pct |")
        lines.append("|:-----------|------:|----:|")
        for bucket, count in hist.items():
            pct = count / m["n_scenes"] * 100 if m["n_scenes"] > 0 else 0
            lines.append(f"| {bucket} | {count} | {pct:.1f}% |")
        lines.append("")

    # â”€â”€ Table 4: Displacement Cost â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("## 4. Displacement Cost Metrics\n")
    lines.append("| Room | n | Objects Moved (%) | Mean Disp (cm) | Max Disp (cm) | Mean Disp/ObjSize |")
    lines.append("|:-----|--:|:----:|:----:|:----:|:----:|")
    for room, m in all_metrics.items():
        d = m["displacement"]
        lines.append(
            f"| {room} | {m['n_scenes']} | "
            f"{d['pct_objects_moved']*100:.1f}% ({d['total_objects_moved']}/{d['total_objects']}) | "
            f"{d['mean_displacement_m']*100:.2f} Â± {d['std_displacement_m']*100:.2f} | "
            f"{d['max_displacement_m']*100:.2f} | "
            f"{d['mean_displacement_relative']:.3f} Â± {d['std_displacement_relative']:.3f} |"
        )
    lines.append("")

    # â”€â”€ Table 5: Resolution Success â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    lines.append("## 5. Resolution Success Rate\n")
    lines.append("| Room | n | Success Rate | Solver Converged |")
    lines.append("|:-----|--:|:----:|:----:|")
    for room, m in all_metrics.items():
        r = m["resolution_success"]
        lines.append(
            f"| {room} | {m['n_scenes']} | "
            f"{r['resolution_success_rate']*100:.1f}% ({r['resolution_success_count']}/{r['n_scenes']}) | "
            f"{r['solver_converged_count']}/{r['n_scenes']} |"
        )
    lines.append("")

    # â”€â”€ Stress Test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if stress_metrics:
        lines.append("## 6. Stress Test Results\n")
        s = stress_metrics
        lines.append(f"- **Scenes**: {s['n_scenes']}, **Objects/scene**: {s['n_objects_per_scene']}, "
                      f"**Room half-extent**: {s['room_half_extent']} m")
        lines.append(f"- **Mean collisions before**: {s['mean_collisions_before']:.1f}")
        lines.append(f"- **Mean collisions after**: {s['mean_collisions_after']:.1f}")
        lines.append(f"- **Mean iterations**: {s['mean_iterations']:.1f} "
                      f"(median: {s['median_iterations']:.0f}, max: {s['max_iterations']})")
        lines.append(f"- **Convergence failure rate**: {s['convergence_failure_rate']*100:.1f}% "
                      f"({s['convergence_failure_count']}/{s['n_scenes']})")
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive collision resolution metrics for DiffuScene"
    )
    parser.add_argument(
        "--base_dir", default=".",
        help="Base directory containing bedroom/, diningroom/, livingroom/ folders"
    )
    parser.add_argument(
        "--rooms", default="bedroom,diningroom,livingroom",
        help="Comma-separated room categories to process"
    )
    parser.add_argument(
        "--max_iter", type=int, default=500,
        help="Max solver iterations (Kmax)"
    )
    parser.add_argument(
        "--push_eps", type=float, default=0.02,
        help="Solver push epsilon (m)"
    )
    parser.add_argument(
        "--output_json", default="collision_metrics_report.json",
        help="Output JSON file"
    )
    parser.add_argument(
        "--output_md", default="collision_metrics_report.md",
        help="Output Markdown file"
    )
    parser.add_argument(
        "--skip_stress", action="store_true",
        help="Skip stress test"
    )
    parser.add_argument(
        "--stress_scenes", type=int, default=50,
        help="Number of stress test scenes"
    )
    parser.add_argument(
        "--stress_objs", type=int, default=20,
        help="Objects per stress test scene"
    )

    args = parser.parse_args()

    rooms = [r.strip() for r in args.rooms.split(",")]
    all_metrics = {}

    print("=" * 70)
    print("COLLISION RESOLUTION METRICS â€” COMPREHENSIVE REPORT")
    print("=" * 70)

    for room in rooms:
        room_dir = os.path.join(args.base_dir, room)
        print(f"\n{'â”€' * 70}")
        print(f"[{room.upper()}]")
        print(f"{'â”€' * 70}")

        result = process_room(room, room_dir, args.base_dir, max_iter=args.max_iter, push_eps=args.push_eps)
        if result is not None:
            all_metrics[room] = result

    # â”€â”€ Stress Test â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    stress_result = None
    if not args.skip_stress:
        print(f"\n{'â”€' * 70}")
        print("[STRESS TEST]")
        print(f"{'â”€' * 70}")
        stress_result = run_stress_test(
            n_scenes=args.stress_scenes,
            n_objs_per_scene=args.stress_objs,
            max_iter=args.max_iter,
            push_eps=args.push_eps,
        )
        print(f"  Convergence failures: {stress_result['convergence_failure_count']}/{stress_result['n_scenes']}")
        print(f"  Mean iterations: {stress_result['mean_iterations']:.1f}")

    # â”€â”€ Save JSON â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    output = {
        "rooms": all_metrics,
        "stress_test": stress_result,
        "params": {
            "max_iter": args.max_iter,
            "push_eps": args.push_eps,
        }
    }
    json_path = os.path.join(args.base_dir, args.output_json)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nJSON report saved to: {json_path}")

    # â”€â”€ Save Markdown â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    md_content = generate_markdown_report(all_metrics, stress_result)
    md_path = os.path.join(args.base_dir, args.output_md)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Markdown report saved to: {md_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()

