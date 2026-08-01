import trimesh
import numpy as np
import json
import glob

d1 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
d2 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'))
idx = d1['scene_ids'].index('LivingDiningRoom-9790_13_013')
labels = np.argmax(d1['class_labels'][idx], axis=1)

obj_files = sorted(glob.glob('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_*.obj'))

meshes = []
for i, f in enumerate(obj_files):
    m = trimesh.load(f, force='mesh')
    if i < len(d1['translations'][idx]):
        orig_t = np.array(d1['translations'][idx][i])
        res_t = np.array(d2['translations'][idx][i])
        m.apply_translation(res_t - orig_t)
    meshes.append((i, m))

def bounds_intersect(b1, b2):
    return np.all(b1[0] < b2[1]) and np.all(b1[1] > b2[0])

for a in range(len(meshes)):
    if labels[a] in [0, 6, 14, 7]: continue
    for b in range(a+1, len(meshes)):
        if labels[b] in [0, 6, 14, 7]: continue
        b1 = meshes[a][1].bounds
        b2 = meshes[b][1].bounds
        if bounds_intersect(b1, b2):
            print(f"Bounds intersect: obj {a} (label {labels[a]}) and obj {b} (label {labels[b]})")
