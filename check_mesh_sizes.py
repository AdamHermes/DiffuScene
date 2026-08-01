import json
import numpy as np
import trimesh

d = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
idx = d['scene_ids'].index('LivingDiningRoom-9790_13_013')

for i in [4, 6]:
    mesh = trimesh.load(f'/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_00{i}.obj')
    print(f"Object {i} mesh bounds: {mesh.bounds}")
    extents = mesh.extents
    print(f"Object {i} mesh extents (L, H, W): {extents}")
    
    json_size = np.array(d['sizes'][idx][i]) * 2.0
    print(f"Object {i} json size * 2.0: {json_size}")
    
    json_trans = np.array(d['translations'][idx][i])
    print(f"Object {i} json trans: {json_trans}")
    print(f"Object {i} mesh centroid: {mesh.centroid}")
