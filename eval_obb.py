import os
import json
import numpy as np
from shapely.geometry import Polygon, box

REPO_ROOT = "."
DATA_DIR = os.path.join(REPO_ROOT, "diffuscene_data")
PROCESSED_DIR = os.path.join(DATA_DIR, "..", "3d_front_processed")

def get_room_dimensions_from_npz(base_id, category):
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
        
    for d in os.listdir(cat_dir):
        if d.endswith("_" + base_id) or d == base_id:
            boxes_path = os.path.join(cat_dir, d, "boxes.npz")
            if os.path.exists(boxes_path):
                data = np.load(boxes_path, allow_pickle=True)
                if "floor_plan_vertices" in data:
                    verts = data["floor_plan_vertices"]
                    min_v = verts.min(axis=0)
                    max_v = verts.max(axis=0)
                    lx = float(max_v[0] - min_v[0])
                    lz = float(max_v[2] - min_v[2])
                    return lx, lz
    return None

def get_ccw_corners_xz(cx, cz, l, w, theta_deg):
    theta = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    dx = np.array([l / 2.0, l / 2.0, -l / 2.0, -l / 2.0])
    dz = np.array([w / 2.0, -w / 2.0, -w / 2.0, w / 2.0])
    px = cx + dx * cos_t - dz * sin_t
    pz = cz + dx * sin_t + dz * cos_t
    return np.stack([px, pz], axis=-1)

def evaluate_oob(json_path, category):
    if not os.path.exists(json_path):
        return None
    with open(json_path, "r") as f:
        data = json.load(f)

    num_scenes = len(data["scene_ids"])
    total_objects = 0
    total_oob_objects = 0
    scenes_with_oob = 0

    for i in range(num_scenes):
        sid = data["scene_ids"][i]
        base_id = sid.split("_")[0]
        room_dims = get_room_dimensions_from_npz(base_id, category)
        if not room_dims:
            continue
        lx, lz = room_dims
        hx, hz = lx / 2.0, lz / 2.0
        # OOB Box is centered around origin
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

        sc_oob_cnt = 0
        for j in range(n_objs):
            c_xz = get_ccw_corners_xz(trans[j, 0], trans[j, 2], sizes[j, 0], sizes[j, 2], angles_deg[j])
            poly = Polygon(c_xz)
            if not poly.is_valid:
                poly = poly.buffer(0)
            inter = floor_poly.intersection(poly)
            if poly.area - inter.area > 1e-4:
                sc_oob_cnt += 1

        total_oob_objects += sc_oob_cnt
        if sc_oob_cnt > 0:
            scenes_with_oob += 1

    if num_scenes == 0 or total_objects == 0:
        return None

    return {
        "num_scenes": num_scenes,
        "total_objects": total_objects,
        "total_oob_objects": total_oob_objects,
        "scenes_with_oob": scenes_with_oob,
        "avg_oob_per_scene": total_oob_objects / num_scenes,
        "oob_object_rate": total_oob_objects / total_objects,
        "oob_scene_rate": scenes_with_oob / num_scenes,
    }

def main():
    categories = ["bedroom", "livingroom", "diningroom"]
    for cat in categories:
        print(f"\n=== {cat.upper()} ===")
        raw_path = os.path.join(cat, "collision_params_merged.json")
        res_path = os.path.join(cat, "collision_params_resolved.json")
        
        raw_res = evaluate_oob(raw_path, cat)
        if raw_res:
            print("RAW:")
            print(f"  OOB Object Rate: {raw_res['oob_object_rate']*100:.3f}% ({raw_res['total_oob_objects']}/{raw_res['total_objects']})")
            print(f"  OOB Scene Rate : {raw_res['oob_scene_rate']*100:.3f}% ({raw_res['scenes_with_oob']}/{raw_res['num_scenes']})")
            print(f"  Avg OOB/Scene  : {raw_res['avg_oob_per_scene']:.3f}")
        
        res_res = evaluate_oob(res_path, cat)
        if res_res:
            print("RESOLVED:")
            print(f"  OOB Object Rate: {res_res['oob_object_rate']*100:.3f}% ({res_res['total_oob_objects']}/{res_res['total_objects']})")
            print(f"  OOB Scene Rate : {res_res['oob_scene_rate']*100:.3f}% ({res_res['scenes_with_oob']}/{res_res['num_scenes']})")
            print(f"  Avg OOB/Scene  : {res_res['avg_oob_per_scene']:.3f}")

if __name__ == '__main__':
    main()
