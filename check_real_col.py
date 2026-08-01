import json
import numpy as np
import sys
sys.path.append('/Users/lehoangan/Documents/GitHub/ROOM/echoscene')
from helpers.resolve_collision import _obb_sat_xz

d = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
idx = d['scene_ids'].index('LivingDiningRoom-9790_13_013')
sizes = np.array(d['sizes'][idx]) * 2.0
trans = d['translations'][idx]
angles = np.array(d['angles'][idx]) * 180 / np.pi
labels = np.argmax(d['class_labels'][idx], axis=1)

classes = ["layout", "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "floor_lamp", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe"]


for i in range(len(sizes)):
    if labels[i] in [0, 6, 14, 7]: continue
    box_i = [sizes[i][0], sizes[i][1], sizes[i][2], trans[i][0], trans[i][1], trans[i][2]]
    for j in range(i+1, len(sizes)):
        if labels[j] in [0, 6, 14, 7]: continue
        box_j = [sizes[j][0], sizes[j][1], sizes[j][2], trans[j][0], trans[j][1], trans[j][2]]
        col, depth, _ = _obb_sat_xz(box_i, angles[i][0], box_j, angles[j][0])
        
        # Check vertical
        cy_i = trans[i][1]
        hy_i = sizes[i][1] / 2.0
        cy_j = trans[j][1]
        hy_j = sizes[j][1] / 2.0
        vert_col = not ((cy_i + hy_i) <= (cy_j - hy_j) or (cy_j + hy_j) <= (cy_i - hy_i))
        
        if col and vert_col and depth > 0:
            name_i = classes[labels[i]] if labels[i] < len(classes) else str(labels[i])
            name_j = classes[labels[j]] if labels[j] < len(classes) else str(labels[j])
            print(f"Collision in 3D: {name_i}({i}) vs {name_j}({j}), depth={depth}")
