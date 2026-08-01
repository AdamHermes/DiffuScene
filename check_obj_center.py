import json
import numpy as np

with open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json', 'r') as f:
    orig = json.load(f)

idx = orig['scene_ids'].index('LivingDiningRoom-9790_13_013')
trans = np.array(orig['translations'][idx])

def get_obj_center(path):
    verts = []
    with open(path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                verts.append([float(x) for x in line.strip().split()[1:]])
    if not verts: return None
    verts = np.array(verts)
    return (verts.min(axis=0) + verts.max(axis=0)) / 2.0

for i in range(len(trans)):
    path = f'/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_{i:03d}.obj'
    center = get_obj_center(path)
    print(f"obj {i}: JSON trans = {trans[i]}, OBJ center = {center}")
