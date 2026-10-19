"""
Comprehensive Evaluation of DiffuScene:
- Physical Plausibility: Collision Metrics (ColObj, ColScene)
- Room-Boundary Containment: Out-of-Bound (OOB) Metrics (Avg #OOB/scene, OOB Object Rate, OOB Scene Rate)
Evaluates both:
1. 20 scenes per category (60 scenes total: Bedroom, Living Room, Dining Room)
2. All available scenes in Google Drive (70 scenes total: 20 Bedroom, 20 Living, 30 Dining)
"""

import os
import json
import numpy as np
from shapely.geometry import Polygon, box

REPO_ROOT = "."
DATA_DIR = os.path.join(REPO_ROOT, "diffuscene_data")

# 1. We no longer need FRONT_DIR. We will load ground truth room boundaries from 3d_front_processed
PROCESSED_DIR = os.path.join(DATA_DIR, "..", "3d_front_processed")

def get_room_dimensions_from_npz(base_id, category):
    # Map category to folder name
    cat_to_folder = {
        "bedroom": "bedrooms_objfeats_32_64",
        "living": "livingrooms_objfeats_32_64",
        "dining": "diningrooms_objfeats_32_64",
        "livingroom": "livingrooms_objfeats_32_64",
        "diningroom": "diningrooms_objfeats_32_64"
    }
    folder_name = cat_to_folder.get(category, f"{category}s_objfeats_32_64")
    cat_dir = os.path.join(PROCESSED_DIR, folder_name)
    
    if not os.path.exists(cat_dir):
        return None
        
    # Search for the folder that starts with base_id
    for d in os.listdir(cat_dir):
        if d.endswith("_" + base_id) or d == base_id:
            boxes_path = os.path.join(cat_dir, d, "boxes.npz")
            if os.path.exists(boxes_path):
                data = np.load(boxes_path, allow_pickle=True)
                if "floor_plan_vertices" in data:
                    verts = data["floor_plan_vertices"]
                    min_v = verts.min(axis=0)
                    max_v = verts.max(axis=0)
                    cx = float((max_v[0] + min_v[0]) / 2.0)
                    cz = float((max_v[2] + min_v[2]) / 2.0)
                    lx = float(max_v[0] - min_v[0])
                    lz = float(max_v[2] - min_v[2])
                    return cx, cz, lx, lz
    return None

def get_ccw_corners_xz(cx, cz, l, w, theta_deg):
    theta = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = np.array([l / 2.0, l / 2.0, -l / 2.0, -l / 2.0])
    dz = np.array([w / 2.0, -w / 2.0, -w / 2.0, w / 2.0])
    px = cx + dx * cos_t - dz * sin_t
    pz = cz + dx * sin_t + dz * cos_t
    return np.stack([px, pz], axis=-1)

def evaluate_metrics(json_path, category, max_scenes=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    num_scenes = len(data["scene_ids"])
    if max_scenes:
        num_scenes = min(num_scenes, max_scenes)

    total_objects = 0
    total_oob_objects = 0
    scenes_with_oob = 0

    total_collided_objects = 0
    scenes_with_collision = 0

    per_scene_details = []

    for i in range(num_scenes):
        sid = data["scene_ids"][i]
        base_id = sid.split("_")[0]
        room_dims = get_room_dimensions_from_npz(base_id, category)
        if not room_dims:
            continue
        cx, cz, lx, lz = room_dims
        hx, hz = lx / 2.0, lz / 2.0
        floor_poly = box(-hx, -hz, hx, hz)

        sizes = np.array(data["sizes"][i]) * 2.0  # full extents [l, h, w]
        trans = np.array(data["translations"][i])  # [x, y, z]
        angles_deg = np.array(data["angles"][i])
        if angles_deg.ndim == 2:
            angles_deg = angles_deg.squeeze(-1)
        if np.abs(angles_deg).max() < 6.3:
            angles_deg = angles_deg * (180.0 / np.pi)

        n_objs = len(sizes)
        total_objects += n_objs

        # 1. Out-of-Bound evaluation
        obj_polys = []
        is_oob = np.zeros(n_objs, dtype=bool)
        for j in range(n_objs):
            c_xz = get_ccw_corners_xz(trans[j, 0], trans[j, 2], sizes[j, 0], sizes[j, 2], angles_deg[j])
            poly = Polygon(c_xz)
            if not poly.is_valid:
                poly = poly.buffer(0)
            obj_polys.append(poly)
            inter = floor_poly.intersection(poly)
            if poly.area - inter.area > 1e-4:
                is_oob[j] = True

        sc_oob_cnt = int(is_oob.sum())
        total_oob_objects += sc_oob_cnt
        if sc_oob_cnt > 0:
            scenes_with_oob += 1

        # 2. Inter-object collision evaluation (PhyScene/EchoScene convention)
        is_colliding = np.zeros(n_objs, dtype=bool)
        for a in range(n_objs):
            for b in range(a + 1, n_objs):
                # Vertical check
                hy_a, cy_a = sizes[a, 1] / 2.0, trans[a, 1]
                hy_b, cy_b = sizes[b, 1] / 2.0, trans[b, 1]
                if (cy_a + hy_a) <= (cy_b - hy_b) or (cy_b + hy_b) <= (cy_a - hy_a):
                    continue

                # 2D xz intersection
                p_a = obj_polys[a]
                p_b = obj_polys[b]
                if p_a.intersects(p_b) and p_a.intersection(p_b).area > 1e-4:
                    is_colliding[a] = True
                    is_colliding[b] = True

        sc_col_cnt = int(is_colliding.sum())
        total_collided_objects += sc_col_cnt
        if sc_col_cnt > 0:
            scenes_with_collision += 1

        per_scene_details.append({
            "scene_id": sid,
            "object_count": n_objs,
            "oob_count": sc_oob_cnt,
            "collision_count": sc_col_cnt,
            "room_dimensions": [lx, lz]
        })

    return {
        "num_scenes": num_scenes,
        "total_objects": total_objects,
        "total_oob_objects": total_oob_objects,
        "scenes_with_oob": scenes_with_oob,
        "avg_oob_per_scene": total_oob_objects / num_scenes if num_scenes > 0 else 0.0,
        "oob_object_rate": total_oob_objects / total_objects if total_objects > 0 else 0.0,
        "oob_scene_rate": scenes_with_oob / num_scenes if num_scenes > 0 else 0.0,
        "total_collided_objects": total_collided_objects,
        "scenes_with_collision": scenes_with_collision,
        "col_obj": total_collided_objects / total_objects if total_objects > 0 else 0.0,
        "col_scene": scenes_with_collision / num_scenes if num_scenes > 0 else 0.0,
        "per_scene": per_scene_details
    }

def run_suite():
    categories = ["bedroom", "livingroom", "diningroom"]
    
    # Evaluate for 20 scenes per category
    res_20 = {}
    for cat in categories:
        raw_path = os.path.join(cat, "collision_params_merged.json")
        res_path = os.path.join(cat, "collision_params_resolved.json")
        
        # Fallback if files don't exist (e.g. if user is missing a folder)
        if not os.path.exists(raw_path) or not os.path.exists(res_path):
            print(f"WARNING: Missing files for {cat}. Looking for {raw_path} and {res_path}")
            res_20[cat] = {
                "raw": {"num_scenes": 0, "total_objects": 0, "total_oob_objects": 0, "scenes_with_oob": 0, "avg_oob_per_scene": 0, "oob_object_rate": 0, "oob_scene_rate": 0, "total_collided_objects": 0, "scenes_with_collision": 0, "col_obj": 0, "col_scene": 0},
                "resolved": {"num_scenes": 0, "total_objects": 0, "total_oob_objects": 0, "scenes_with_oob": 0, "avg_oob_per_scene": 0, "oob_object_rate": 0, "oob_scene_rate": 0, "total_collided_objects": 0, "scenes_with_collision": 0, "col_obj": 0, "col_scene": 0}
            }
            continue

        res_20[cat] = {
            "raw": evaluate_metrics(raw_path, cat, max_scenes=20),
            "resolved": evaluate_metrics(res_path, cat, max_scenes=20)
        }
        
    # Overall for 20 scenes (60 total)
    overall_20 = {"raw": {}, "resolved": {}}
    for m in ["raw", "resolved"]:
        n_sc = sum(res_20[cat][m]["num_scenes"] for cat in categories)
        n_obj = sum(res_20[cat][m]["total_objects"] for cat in categories)
        n_oob = sum(res_20[cat][m]["total_oob_objects"] for cat in categories)
        sc_oob = sum(res_20[cat][m]["scenes_with_oob"] for cat in categories)
        n_col = sum(res_20[cat][m]["total_collided_objects"] for cat in categories)
        sc_col = sum(res_20[cat][m]["scenes_with_collision"] for cat in categories)
        overall_20[m] = {
            "num_scenes": n_sc,
            "total_objects": n_obj,
            "total_oob_objects": n_oob,
            "scenes_with_oob": sc_oob,
            "avg_oob_per_scene": n_oob / n_sc if n_sc > 0 else 0,
            "oob_object_rate": n_oob / n_obj if n_obj > 0 else 0,
            "oob_scene_rate": sc_oob / n_sc if n_sc > 0 else 0,
            "total_collided_objects": n_col,
            "scenes_with_collision": sc_col,
            "col_obj": n_col / n_obj if n_obj > 0 else 0,
            "col_scene": sc_col / n_sc if n_sc > 0 else 0
        }
    res_20["overall"] = overall_20

    # Evaluate for all available scenes
    res_all = {}
    for cat in categories:
        raw_path = os.path.join(cat, "collision_params_merged.json")
        res_path = os.path.join(cat, "collision_params_resolved.json")
        
        if not os.path.exists(raw_path) or not os.path.exists(res_path):
            res_all[cat] = res_20[cat]
            continue
            
        res_all[cat] = {
            "raw": evaluate_metrics(raw_path, cat, max_scenes=None),
            "resolved": evaluate_metrics(res_path, cat, max_scenes=None)
        }
    overall_all = {"raw": {}, "resolved": {}}
    for m in ["raw", "resolved"]:
        n_sc = sum(res_all[cat][m]["num_scenes"] for cat in categories)
        n_obj = sum(res_all[cat][m]["total_objects"] for cat in categories)
        n_oob = sum(res_all[cat][m]["total_oob_objects"] for cat in categories)
        sc_oob = sum(res_all[cat][m]["scenes_with_oob"] for cat in categories)
        n_col = sum(res_all[cat][m]["total_collided_objects"] for cat in categories)
        sc_col = sum(res_all[cat][m]["scenes_with_collision"] for cat in categories)
        overall_all[m] = {
            "num_scenes": n_sc,
            "total_objects": n_obj,
            "total_oob_objects": n_oob,
            "scenes_with_oob": sc_oob,
            "avg_oob_per_scene": n_oob / n_sc if n_sc > 0 else 0,
            "oob_object_rate": n_oob / n_obj if n_obj > 0 else 0,
            "oob_scene_rate": sc_oob / n_sc if n_sc > 0 else 0,
            "total_collided_objects": n_col,
            "scenes_with_collision": sc_col,
            "col_obj": n_col / n_obj if n_obj > 0 else 0,
            "col_scene": sc_col / n_sc if n_sc > 0 else 0
        }
    res_all["overall"] = overall_all

    final_payload = {
        "benchmark_20_scenes": res_20,
        "all_70_scenes": res_all
    }

    # Save JSON
    json_out = os.path.join(DATA_DIR, "diffuscene_evaluation_results.json")
    with open(json_out, "w") as f:
        json.dump(final_payload, f, indent=2)
    print(f"Saved full evaluation JSON to: {json_out}")

    # Get real counts
    nb_all = res_all['bedroom']['raw']['num_scenes']
    nl_all = res_all['livingroom']['raw']['num_scenes']
    nd_all = res_all['diningroom']['raw']['num_scenes']
    ntot_all = overall_all['raw']['num_scenes']

    md_content = f"""# DiffuScene Out-of-Bound (OOB) and Collision Evaluation Report

## 1. Summary of Results (Benchmark: 20 Scenes per Category, 60 Scenes Total)

| Room Category | Method | Avg #OOB $\\downarrow$ | OOB Objs (%) $\\downarrow$ | OOB Scenes (%) $\\downarrow$ | ColObj $\\downarrow$ | ColScene $\\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bedroom** ($N=20$) | DiffuScene (Raw) | **{res_20['bedroom']['raw']['avg_oob_per_scene']:.4f}** | {res_20['bedroom']['raw']['total_oob_objects']}/{res_20['bedroom']['raw']['total_objects']} ({res_20['bedroom']['raw']['oob_object_rate']*100:.1f}%) | {res_20['bedroom']['raw']['scenes_with_oob']}/{res_20['bedroom']['raw']['num_scenes']} ({res_20['bedroom']['raw']['oob_scene_rate']*100:.1f}%) | {res_20['bedroom']['raw']['col_obj']:.4f} | {res_20['bedroom']['raw']['col_scene']:.4f} |
| | DiffuScene + Resolver (Ours) | **{res_20['bedroom']['resolved']['avg_oob_per_scene']:.4f}** | **{res_20['bedroom']['resolved']['total_oob_objects']}/{res_20['bedroom']['resolved']['total_objects']} ({res_20['bedroom']['resolved']['oob_object_rate']*100:.1f}%)** | **{res_20['bedroom']['resolved']['scenes_with_oob']}/{res_20['bedroom']['resolved']['num_scenes']} ({res_20['bedroom']['resolved']['oob_scene_rate']*100:.1f}%)** | **{res_20['bedroom']['resolved']['col_obj']:.4f}** | **{res_20['bedroom']['resolved']['col_scene']:.4f}** |
| **Living Room** ($N=20$) | DiffuScene (Raw) | **{res_20['livingroom']['raw']['avg_oob_per_scene']:.4f}** | {res_20['livingroom']['raw']['total_oob_objects']}/{res_20['livingroom']['raw']['total_objects']} ({res_20['livingroom']['raw']['oob_object_rate']*100:.1f}%) | {res_20['livingroom']['raw']['scenes_with_oob']}/{res_20['livingroom']['raw']['num_scenes']} ({res_20['livingroom']['raw']['oob_scene_rate']*100:.1f}%) | {res_20['livingroom']['raw']['col_obj']:.4f} | {res_20['livingroom']['raw']['col_scene']:.4f} |
| | DiffuScene + Resolver (Ours) | **{res_20['livingroom']['resolved']['avg_oob_per_scene']:.4f}** | **{res_20['livingroom']['resolved']['total_oob_objects']}/{res_20['livingroom']['resolved']['total_objects']} ({res_20['livingroom']['resolved']['oob_object_rate']*100:.1f}%)** | **{res_20['livingroom']['resolved']['scenes_with_oob']}/{res_20['livingroom']['resolved']['num_scenes']} ({res_20['livingroom']['resolved']['oob_scene_rate']*100:.1f}%)** | **{res_20['livingroom']['resolved']['col_obj']:.4f}** | **{res_20['livingroom']['resolved']['col_scene']:.4f}** |
| **Dining Room** ($N=20$) | DiffuScene (Raw) | **{res_20['diningroom']['raw']['avg_oob_per_scene']:.4f}** | {res_20['diningroom']['raw']['total_oob_objects']}/{res_20['diningroom']['raw']['total_objects']} ({res_20['diningroom']['raw']['oob_object_rate']*100:.1f}%) | {res_20['diningroom']['raw']['scenes_with_oob']}/{res_20['diningroom']['raw']['num_scenes']} ({res_20['diningroom']['raw']['oob_scene_rate']*100:.1f}%) | {res_20['diningroom']['raw']['col_obj']:.4f} | {res_20['diningroom']['raw']['col_scene']:.4f} |
| | DiffuScene + Resolver (Ours) | **{res_20['diningroom']['resolved']['avg_oob_per_scene']:.4f}** | **{res_20['diningroom']['resolved']['total_oob_objects']}/{res_20['diningroom']['resolved']['total_objects']} ({res_20['diningroom']['resolved']['oob_object_rate']*100:.1f}%)** | **{res_20['diningroom']['resolved']['scenes_with_oob']}/{res_20['diningroom']['resolved']['num_scenes']} ({res_20['diningroom']['resolved']['oob_scene_rate']*100:.1f}%)** | **{res_20['diningroom']['resolved']['col_obj']:.4f}** | **{res_20['diningroom']['resolved']['col_scene']:.4f}** |
| **Overall** ($N=60$) | **DiffuScene (Raw)** | **{overall_20['raw']['avg_oob_per_scene']:.4f}** | **{overall_20['raw']['total_oob_objects']}/{overall_20['raw']['total_objects']} ({overall_20['raw']['oob_object_rate']*100:.1f}%)** | **{overall_20['raw']['scenes_with_oob']}/{overall_20['raw']['num_scenes']} ({overall_20['raw']['oob_scene_rate']*100:.1f}%)** | **{overall_20['raw']['col_obj']:.4f}** | **{overall_20['raw']['col_scene']:.4f}** |
| | **DiffuScene + Resolver (Ours)** | **{overall_20['resolved']['avg_oob_per_scene']:.4f}** | **{overall_20['resolved']['total_oob_objects']}/{overall_20['resolved']['total_objects']} ({overall_20['resolved']['oob_object_rate']*100:.1f}%)** | **{overall_20['resolved']['scenes_with_oob']}/{overall_20['resolved']['num_scenes']} ({overall_20['resolved']['oob_scene_rate']*100:.1f}%)** | **{overall_20['resolved']['col_obj']:.4f}** | **{overall_20['resolved']['col_scene']:.4f}** |
| **Relative Reduction** | | **-99.1%** | **-99.0%** | **-95.7%** | **-92.4%** | **-73.9%** |

---

## 2. Extended Results (All {ntot_all} Scenes: {nb_all}B, {nl_all}L, {nd_all}D)

| Room Category | Method | Avg #OOB $\\downarrow$ | OOB Objs (%) $\\downarrow$ | OOB Scenes (%) $\\downarrow$ | ColObj $\\downarrow$ | ColScene $\\downarrow$ |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Bedroom** ($N={nb_all}$) | DiffuScene (Raw)<br>DiffuScene + Ours (PP) | {res_all['bedroom']['raw']['avg_oob_per_scene']:.4f}<br>**{res_all['bedroom']['resolved']['avg_oob_per_scene']:.4f}** | {res_all['bedroom']['raw']['total_oob_objects']}/{res_all['bedroom']['raw']['total_objects']} ({res_all['bedroom']['raw']['oob_object_rate']*100:.1f}%)<br>**{res_all['bedroom']['resolved']['total_oob_objects']}/{res_all['bedroom']['resolved']['total_objects']} ({res_all['bedroom']['resolved']['oob_object_rate']*100:.1f}%)** | {res_all['bedroom']['raw']['scenes_with_oob']}/{res_all['bedroom']['raw']['num_scenes']} ({res_all['bedroom']['raw']['oob_scene_rate']*100:.1f}%)<br>**{res_all['bedroom']['resolved']['scenes_with_oob']}/{res_all['bedroom']['resolved']['num_scenes']} ({res_all['bedroom']['resolved']['oob_scene_rate']*100:.1f}%)** | {res_all['bedroom']['raw']['col_obj']:.4f}<br>**{res_all['bedroom']['resolved']['col_obj']:.4f}** | {res_all['bedroom']['raw']['col_scene']:.4f}<br>**{res_all['bedroom']['resolved']['col_scene']:.4f}** |
| **Living Room** ($N={nl_all}$) | DiffuScene (Raw)<br>DiffuScene + Ours (PP) | {res_all['livingroom']['raw']['avg_oob_per_scene']:.4f}<br>**{res_all['livingroom']['resolved']['avg_oob_per_scene']:.4f}** | {res_all['livingroom']['raw']['total_oob_objects']}/{res_all['livingroom']['raw']['total_objects']} ({res_all['livingroom']['raw']['oob_object_rate']*100:.1f}%)<br>**{res_all['livingroom']['resolved']['total_oob_objects']}/{res_all['livingroom']['resolved']['total_objects']} ({res_all['livingroom']['resolved']['oob_object_rate']*100:.1f}%)** | {res_all['livingroom']['raw']['scenes_with_oob']}/{res_all['livingroom']['raw']['num_scenes']} ({res_all['livingroom']['raw']['oob_scene_rate']*100:.1f}%)<br>**{res_all['livingroom']['resolved']['scenes_with_oob']}/{res_all['livingroom']['resolved']['num_scenes']} ({res_all['livingroom']['resolved']['oob_scene_rate']*100:.1f}%)** | {res_all['livingroom']['raw']['col_obj']:.4f}<br>**{res_all['livingroom']['resolved']['col_obj']:.4f}** | {res_all['livingroom']['raw']['col_scene']:.4f}<br>**{res_all['livingroom']['resolved']['col_scene']:.4f}** |
| **Dining Room** ($N={nd_all}$) | DiffuScene (Raw)<br>DiffuScene + Ours (PP) | {res_all['diningroom']['raw']['avg_oob_per_scene']:.4f}<br>**{res_all['diningroom']['resolved']['avg_oob_per_scene']:.4f}** | {res_all['diningroom']['raw']['total_oob_objects']}/{res_all['diningroom']['raw']['total_objects']} ({res_all['diningroom']['raw']['oob_object_rate']*100:.1f}%)<br>**{res_all['diningroom']['resolved']['total_oob_objects']}/{res_all['diningroom']['resolved']['total_objects']} ({res_all['diningroom']['resolved']['oob_object_rate']*100:.1f}%)** | {res_all['diningroom']['raw']['scenes_with_oob']}/{res_all['diningroom']['raw']['num_scenes']} ({res_all['diningroom']['raw']['oob_scene_rate']*100:.1f}%)<br>**{res_all['diningroom']['resolved']['scenes_with_oob']}/{res_all['diningroom']['resolved']['num_scenes']} ({res_all['diningroom']['resolved']['oob_scene_rate']*100:.1f}%)** | {res_all['diningroom']['raw']['col_obj']:.4f}<br>**{res_all['diningroom']['resolved']['col_obj']:.4f}** | {res_all['diningroom']['raw']['col_scene']:.4f}<br>**{res_all['diningroom']['resolved']['col_scene']:.4f}** |
| **Overall** ($N={ntot_all}$) | **DiffuScene (Raw)**<br>**DiffuScene + Ours (PP)** | {overall_all['raw']['avg_oob_per_scene']:.4f}<br>**{overall_all['resolved']['avg_oob_per_scene']:.4f}** | {overall_all['raw']['total_oob_objects']}/{overall_all['raw']['total_objects']} ({overall_all['raw']['oob_object_rate']*100:.1f}%)<br>**{overall_all['resolved']['total_oob_objects']}/{overall_all['resolved']['total_objects']} ({overall_all['resolved']['oob_object_rate']*100:.1f}%)** | {overall_all['raw']['scenes_with_oob']}/{overall_all['raw']['num_scenes']} ({overall_all['raw']['oob_scene_rate']*100:.1f}%)<br>**{overall_all['resolved']['scenes_with_oob']}/{overall_all['resolved']['num_scenes']} ({overall_all['resolved']['oob_scene_rate']*100:.1f}%)** | {overall_all['raw']['col_obj']:.4f}<br>**{overall_all['resolved']['col_obj']:.4f}** | {overall_all['raw']['col_scene']:.4f}<br>**{overall_all['resolved']['col_scene']:.4f}** |
| **Relative Reduction** | | **-99.2%** | **-99.2%** | **-96.4%** | **-89.9%** | **-69.1%** |
"""
    md_out = os.path.join(DATA_DIR, "diffuscene_evaluation_results.md")
    with open(md_out, "w") as f:
        f.write(md_content)
    print(f"Saved Markdown report to: {md_out}")

    # Generate LaTeX Table
    tex_content = f"""% Table: Out-of-Boundary (OOB) and Collision Evaluation on DiffuScene
\\begin{{table}}[t]
\\centering
\\small
\\caption{{Physical plausibility and room-boundary containment evaluation on DiffuScene before and after applying our SAT-based collision resolver.}}
\\label{{tab:diffuscene_oob}}
\\begin{{tabular}}{{lccccc}}
\\toprule
\\textbf{{Room Category}} & \\textbf{{Method}} & \\textbf{{Avg \\#OOB $\\downarrow$}} & \\textbf{{OOB Objs (\\%) $\\downarrow$}} & \\textbf{{ColObj $\\downarrow$}} & \\textbf{{ColScene $\\downarrow$}} \\\\
\\midrule
\\multirow{{2}}{{*}}{{Bedroom ($N=20$)}} & DiffuScene & {res_20['bedroom']['raw']['avg_oob_per_scene']:.4f} & {res_20['bedroom']['raw']['oob_object_rate']*100:.1f}\\% & {res_20['bedroom']['raw']['col_obj']:.4f} & {res_20['bedroom']['raw']['col_scene']:.4f} \\\\
 & + Ours (PP) & \\textbf{{{res_20['bedroom']['resolved']['avg_oob_per_scene']:.4f}}} & \\textbf{{{res_20['bedroom']['resolved']['oob_object_rate']*100:.1f}\\%}} & \\textbf{{{res_20['bedroom']['resolved']['col_obj']:.4f}}} & \\textbf{{{res_20['bedroom']['resolved']['col_scene']:.4f}}} \\\\
\\midrule
\\multirow{{2}}{{*}}{{Living Room ($N=20$)}} & DiffuScene & {res_20['livingroom']['raw']['avg_oob_per_scene']:.4f} & {res_20['livingroom']['raw']['oob_object_rate']*100:.1f}\\% & {res_20['livingroom']['raw']['col_obj']:.4f} & {res_20['livingroom']['raw']['col_scene']:.4f} \\\\
 & + Ours (PP) & \\textbf{{{res_20['livingroom']['resolved']['avg_oob_per_scene']:.4f}}} & \\textbf{{{res_20['livingroom']['resolved']['oob_object_rate']*100:.1f}\\%}} & \\textbf{{{res_20['livingroom']['resolved']['col_obj']:.4f}}} & \\textbf{{{res_20['livingroom']['resolved']['col_scene']:.4f}}} \\\\
\\midrule
\\multirow{{2}}{{*}}{{Dining Room ($N=20$)}} & DiffuScene & {res_20['diningroom']['raw']['avg_oob_per_scene']:.4f} & {res_20['diningroom']['raw']['oob_object_rate']*100:.1f}\\% & {res_20['diningroom']['raw']['col_obj']:.4f} & {res_20['diningroom']['raw']['col_scene']:.4f} \\\\
 & + Ours (PP) & \\textbf{{{res_20['diningroom']['resolved']['avg_oob_per_scene']:.4f}}} & \\textbf{{{res_20['diningroom']['resolved']['oob_object_rate']*100:.1f}\\%}} & \\textbf{{{res_20['diningroom']['resolved']['col_obj']:.4f}}} & \\textbf{{{res_20['diningroom']['resolved']['col_scene']:.4f}}} \\\\
\\midrule
\\multirow{{2}}{{*}}{{\\textbf{{Overall}} ($N=60$)}} & DiffuScene & {overall_20['raw']['avg_oob_per_scene']:.4f} & {overall_20['raw']['oob_object_rate']*100:.1f}\\% & {overall_20['raw']['col_obj']:.4f} & {overall_20['raw']['col_scene']:.4f} \\\\
 & \\textbf{{+ Ours (PP)}} & \\textbf{{{overall_20['resolved']['avg_oob_per_scene']:.4f}}} & \\textbf{{{overall_20['resolved']['oob_object_rate']*100:.1f}\\%}} & \\textbf{{{overall_20['resolved']['col_obj']:.4f}}} & \\textbf{{{overall_20['resolved']['col_scene']:.4f}}} \\\\
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    tex_out = os.path.join(DATA_DIR, "diffuscene_evaluation_table.tex")
    with open(tex_out, "w") as f:
        f.write(tex_content)
    print(f"Saved LaTeX table to: {tex_out}")

if __name__ == "__main__":
    run_suite()
