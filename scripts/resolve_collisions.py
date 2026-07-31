import json
import torch
import numpy as np
import sys
import os

# Ensure helpers module can be imported
sys.path.append('/Users/lehoangan/Documents/GitHub/ROOM/echoscene')
from helpers.resolve_collision import resolve_bbox_collisions_obb

def resolve_json(input_path, output_path):
    print(f"\nLoading {input_path}...")
    with open(input_path, 'r') as f:
        data = json.load(f)
        
    num_scenes = len(data['scene_ids'])
    print(f"Processing {num_scenes} scenes...")
    
    for i in range(num_scenes):
        sizes = torch.tensor(data['sizes'][i], dtype=torch.float32)
        translations = torch.tensor(data['translations'][i], dtype=torch.float32)
        angles_rad = torch.tensor(data['angles'][i], dtype=torch.float32)
        
        # Angles in JSON are radians, solver expects degrees
        angles_deg = angles_rad * (180.0 / np.pi)
        
        objectness = None
        if 'objectness' in data:
            objectness = torch.tensor(data['objectness'][i], dtype=torch.float32)
            
        class_labels = np.array(data['class_labels'][i])
        
        boxes = torch.cat([sizes, translations], dim=-1)
        
        resolved_boxes = resolve_bbox_collisions_obb(
            boxes=boxes,
            angles_pred=angles_deg,
            objectness_mask=objectness,
            class_labels=class_labels,
            max_iter=500,
            push_eps=0.02,
            verbose=False
        )
        
        # Extract updated translations [x, y, z] from [l, h, w, x, y, z]
        data['translations'][i] = resolved_boxes[:, 3:].tolist()
        
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
        
    print(f"Saved resolved bboxes to {output_path}")

def main():
    files = [
        "/Users/lehoangan/Downloads/merge_b/drive-download-20260731T041851Z-1-001/merged/collision_params.json",
        "/Users/lehoangan/Downloads/merge_L/merged/collision_params.json",
        "/Users/lehoangan/Downloads/merge_d/merged/collision_params.json"
    ]
    
    for in_file in files:
        if not os.path.exists(in_file):
            print(f"Skipping {in_file}, does not exist")
            continue
        
        base, ext = os.path.splitext(in_file)
        out_file = base + "_resolved" + ext
        resolve_json(in_file, out_file)

if __name__ == '__main__':
    main()
