import json
import numpy as np
import os

def _obb_corners_xz(cx, cz, l, w, angle_deg):
    angle_rad = np.radians(angle_deg)
    hx, hz = l / 2.0, w / 2.0
    corners = np.array([
        [ hx,  hz],
        [ hx, -hz],
        [-hx, -hz],
        [-hx,  hz]
    ])
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return corners.dot(R.T) + np.array([cx, cz])

def _obb_sat_xz(box_a, angle_a_deg, box_b, angle_b_deg):
    ca_x, ca_z, la, wa = box_a[3], box_a[5], box_a[0], box_a[2]
    cb_x, cb_z, lb, wb = box_b[3], box_b[5], box_b[0], box_b[2]
    corners_a = _obb_corners_xz(ca_x, ca_z, la, wa, angle_a_deg)
    corners_b = _obb_corners_xz(cb_x, cb_z, lb, wb, angle_b_deg)
    
    a_rad = np.radians(angle_a_deg)
    b_rad = np.radians(angle_b_deg)
    axes = [
        np.array([np.cos(a_rad), np.sin(a_rad)]),
        np.array([-np.sin(a_rad), np.cos(a_rad)]),
        np.array([np.cos(b_rad), np.sin(b_rad)]),
        np.array([-np.sin(b_rad), np.cos(b_rad)])
    ]
    
    min_depth = float('inf')
    for axis in axes:
        proj_a = [np.dot(c, axis) for c in corners_a]
        proj_b = [np.dot(c, axis) for c in corners_b]
        min_a, max_a = min(proj_a), max(proj_a)
        min_b, max_b = min(proj_b), max(proj_b)
        
        if max_a <= min_b or max_b <= min_a:
            return False, 0.0
        
        overlap = min(max_a, max_b) - max(min_a, min_b)
        if overlap < min_depth:
            min_depth = overlap
            
    return True, min_depth

def evaluate_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    col_objs = []
    col_scenes = []
    
    for i in range(len(data['scene_ids'])):
        sizes = np.array(data['sizes'][i]) * 2.0
        translations = np.array(data['translations'][i])
        angles_rad = np.array(data['angles'][i])
        angles_deg = angles_rad * (180.0 / np.pi)
        
        labels_idx = np.argmax(np.array(data['class_labels'][i]), axis=-1)
        
        # Skip layout and lamps
        valid_indices = []
        for j in range(len(sizes)):
            if labels_idx[j] not in [0, 4, 12, 15]:
                valid_indices.append(j)
                
        scene_col = 0
        total_pairs = 0
        
        for j in range(len(valid_indices)):
            for k in range(j + 1, len(valid_indices)):
                idx_j = valid_indices[j]
                idx_k = valid_indices[k]
                
                box_j = [sizes[idx_j][0], sizes[idx_j][1], sizes[idx_j][2], 
                         translations[idx_j][0], translations[idx_j][1], translations[idx_j][2]]
                box_k = [sizes[idx_k][0], sizes[idx_k][1], sizes[idx_k][2], 
                         translations[idx_k][0], translations[idx_k][1], translations[idx_k][2]]
                
                col, depth = _obb_sat_xz(box_j, angles_deg[idx_j][0], box_k, angles_deg[idx_k][0])
                
                hy_j = box_j[1] / 2.0
                cy_j = box_j[4]
                hy_k = box_k[1] / 2.0
                cy_k = box_k[4]
                y_overlap = min(cy_j + hy_j, cy_k + hy_k) - max(cy_j - hy_j, cy_k - hy_k)
                
                # Use a small epsilon to avoid floating point precision issues on resolved objects
                if col and y_overlap > 1e-4 and depth > 1e-4:
                    scene_col += 1
                total_pairs += 1
                    
        if total_pairs > 0:
            col_objs.append(scene_col / total_pairs)
        else:
            col_objs.append(0.0)
            
        if scene_col > 0:
            col_scenes.append(1.0)
        else:
            col_scenes.append(0.0)
            
    avg_col_obj = np.mean(col_objs)
    avg_col_scene = np.mean(col_scenes)
    return avg_col_obj, avg_col_scene

def process_folder(folder_path):
    orig_json = os.path.join(folder_path, "collision_params.json")
    resolved_json = os.path.join(folder_path, "collision_params_resolved.json")
    
    log_path = os.path.join(folder_path, "col.log")
    with open(log_path, 'w') as f:
        if os.path.exists(orig_json):
            col_obj, col_scene = evaluate_json(orig_json)
            print(f"{folder_path}/collision_params.json -> Col Obj: {col_obj:.4f}, Col Scene: {col_scene:.4f}")
            f.write(f"collision_params.json\nCol Obj: {col_obj:.4f}\nCol Scene: {col_scene:.4f}\n\n")
            
        if os.path.exists(resolved_json):
            col_obj, col_scene = evaluate_json(resolved_json)
            print(f"{folder_path}/collision_params_resolved.json -> Col Obj: {col_obj:.4f}, Col Scene: {col_scene:.4f}")
            f.write(f"collision_params_resolved.json\nCol Obj: {col_obj:.4f}\nCol Scene: {col_scene:.4f}\n\n")

def main():
    folders = [
        "/Users/lehoangan/Downloads/merge_b/drive-download-20260731T041851Z-1-001/merged",
        "/Users/lehoangan/Downloads/merge_L/merged",
        "/Users/lehoangan/Downloads/merge_d/merged"
    ]
    for folder in folders:
        if os.path.exists(folder):
            process_folder(folder)

if __name__ == '__main__':
    main()
