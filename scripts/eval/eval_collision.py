import argparse
import datetime
import json
import numpy as np
import os

def _obb_corners_xz(cx, cz, l, w, angle_rad):
    cos_t, sin_t = np.cos(angle_rad), np.sin(angle_rad)
    ax = np.array([cos_t, sin_t])
    az = np.array([-sin_t, cos_t])
    c = np.array([cx, cz])
    hx, hz = l / 2.0, w / 2.0
    return np.stack([
        c + hx*ax + hz*az,
        c + hx*ax - hz*az,
        c - hx*ax + hz*az,
        c - hx*ax - hz*az,
    ])

def _sat_overlap(corners_a, corners_b, axis):
    pa = corners_a @ axis
    pb = corners_b @ axis
    return min(pa.max(), pb.max()) - max(pa.min(), pb.min())

def check_collision(box_i, ang_i, box_j, ang_j):
    # box is [l, h, w, cx, cy, cz]
    li, hi, wi, cxi, cyi, czi = box_i
    lj, hj, wj, cxj, cyj, czj = box_j

    # Vertical check
    hy_i = hi / 2.0
    hy_j = hj / 2.0
    if (cyi + hy_i) <= (cyj - hy_j) or (cyj + hy_j) <= (cyi - hy_i):
        return False

    corners_i = _obb_corners_xz(cxi, czi, li, wi, ang_i)
    corners_j = _obb_corners_xz(cxj, czj, lj, wj, ang_j)

    axes = [
        np.array([ np.cos(ang_i),  np.sin(ang_i)]),
        np.array([-np.sin(ang_i),  np.cos(ang_i)]),
        np.array([ np.cos(ang_j),  np.sin(ang_j)]),
        np.array([-np.sin(ang_j),  np.cos(ang_j)]),
    ]

    for axis in axes:
        n = np.linalg.norm(axis)
        if n < 1e-8: continue
        axis = axis / n
        overlap = _sat_overlap(corners_i, corners_j, axis)
        if overlap <= 0:
            return False  # Separating axis found

    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path', type=str, default='../../DiffuScene_Results/collision_params.json')
    args = parser.parse_args()

    with open(args.json_path, 'r') as f:
        data = json.load(f)

    num_scenes = len(data['scene_ids'])
    
    total_objects = 0
    collided_objects = 0
    collided_scenes = 0

    has_objectness = 'objectness' in data

    for i in range(num_scenes):
        sizes = data['sizes'][i]
        translations = data['translations'][i]
        angles = data['angles'][i]
        class_labels = data['class_labels'][i]
        
        objectness = data['objectness'][i] if has_objectness else None
        
        N = len(sizes)
        labels_idx = np.argmax(class_labels, axis=-1)

        valid_objs = []
        for j in range(N):
            if objectness is not None and objectness[j][0] <= 0:
                continue
            
            # Skip layout(0), floor(6), ceiling(14), lamps(7)
            if labels_idx[j] in [0, 6, 14, 7]:
                continue
                
            box = [sizes[j][0], sizes[j][1], sizes[j][2], 
                   translations[j][0], translations[j][1], translations[j][2]]
            ang = angles[j][0]
            valid_objs.append((box, ang))

        scene_total_objects = len(valid_objs)
        scene_collided_objects = 0
        scene_has_collision = False

        collided_flags = [False] * scene_total_objects

        for idx_a in range(scene_total_objects):
            for idx_b in range(idx_a + 1, scene_total_objects):
                if check_collision(valid_objs[idx_a][0], valid_objs[idx_a][1],
                                   valid_objs[idx_b][0], valid_objs[idx_b][1]):
                    collided_flags[idx_a] = True
                    collided_flags[idx_b] = True

        scene_collided_objects = sum(collided_flags)
        if scene_collided_objects > 0:
            scene_has_collision = True

        total_objects += scene_total_objects
        collided_objects += scene_collided_objects
        if scene_has_collision:
            collided_scenes += 1

    col_obj = collided_objects / total_objects if total_objects > 0 else 0
    col_scene = collided_scenes / num_scenes if num_scenes > 0 else 0

    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"eval_collision_{timestamp}.log"
    log_content = f"File: {args.json_path}\nColObj: {col_obj:.6f} ({collided_objects}/{total_objects})\nColScene: {col_scene:.6f} ({collided_scenes}/{num_scenes})\n"
    
    print(log_content)
    with open(log_filename, 'a') as f:
        f.write(log_content)
    print(f"Results appended to {log_filename}")

if __name__ == '__main__':
    main()

