import json
import os
import sys
import numpy as np
import torch
sys.path.append('/Users/lehoangan/Documents/GitHub/ROOM/echoscene/helpers')
from resolve_collision import resolve_bbox_collisions_obb

def fix_and_resolve(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    resolved_data = {
        "scene_ids": data["scene_ids"],
        "class_labels": data["class_labels"],
        "translations": []
    }
    
    for i, scene_id in enumerate(data["scene_ids"]):
        sizes = np.array(data["sizes"][i]) * 2.0  # FIX: multiply half-extents by 2 to get full sizes
        trans = np.array(data["translations"][i])
        angles = np.array(data["angles"][i])
        class_labels = np.array(data["class_labels"][i])
        
        boxes = np.hstack([sizes, trans])
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        angles_tensor = torch.tensor(angles, dtype=torch.float32)
        
        # We don't have objectness_mask in the JSON, so pass None
        # We also pass class_labels for layout ignoring (floor = 0)
        resolved_boxes = resolve_bbox_collisions_obb(
            boxes_tensor, angles_tensor, class_labels=class_labels, verbose=False
        )
        
        res_trans = resolved_boxes[:, 3:6].cpu().numpy().tolist()
        resolved_data["translations"].append(res_trans)
        print(f"Resolved scene {scene_id}")

    out_path = os.path.join(os.path.dirname(json_path), "collision_params_resolved.json")
    with open(out_path, 'w') as f:
        json.dump(resolved_data, f)
    print(f"Saved {out_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fix_and_resolve(sys.argv[1])
    else:
        print("Usage: python fix_and_resolve_collisions.py <path_to_collision_params.json>")
