import json
import trimesh
import numpy as np
import glob

d = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
idx = d['scene_ids'].index('LivingDiningRoom-9790_13_013')

obj_files = sorted(glob.glob('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_*.obj'))

for i, f in enumerate(obj_files):
    m = trimesh.load(f, force='mesh')
    v = m.vertices
    mesh_size = v.max(axis=0) - v.min(axis=0)
    json_size = np.array(d['sizes'][idx][i]) * 2.0
    print(f"obj {i}: JSON size = {json_size}, MESH size = {mesh_size}")
