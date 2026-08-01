import json
import numpy as np

d = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'))
idx = d['scene_ids'].index('LivingDiningRoom-9790_13_013')
sizes = np.array(d['sizes'][idx]) * 2.0
trans = d['translations'][idx]
labels = np.argmax(d['class_labels'][idx], axis=1)
classes = ["layout", "armchair", "bookshelf", "cabinet", "ceiling_lamp", "chair", "children_cabinet", "coffee_table", "desk", "double_bed", "dressing_chair", "dressing_table", "floor_lamp", "kids_bed", "nightstand", "pendant_lamp", "shelf", "single_bed", "sofa", "stool", "table", "tv_stand", "wardrobe"]

pairs = [(2, 4), (2, 6), (2, 7)]
for i, j in pairs:
    ci = labels[i]
    cj = labels[j]
    name_i = classes[ci] if ci < len(classes) else str(ci)
    name_j = classes[cj] if cj < len(classes) else str(cj)
    
    hy_i = sizes[i][1] / 2.0
    cy_i = trans[i][1]
    hy_j = sizes[j][1] / 2.0
    cy_j = trans[j][1]
    
    col_y = not ((cy_i + hy_i) <= (cy_j - hy_j) or (cy_j + hy_j) <= (cy_i - hy_i))
    print(f"{name_i} (id {i}) vs {name_j} (id {j})")
    print(f"  {name_i}: y in [{cy_i - hy_i:.3f}, {cy_i + hy_i:.3f}]")
    print(f"  {name_j}: y in [{cy_j - hy_j:.3f}, {cy_j + hy_j:.3f}]")
    print(f"  Overlap in Y? {col_y}")

