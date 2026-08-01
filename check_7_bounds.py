import trimesh
import numpy as np
import json
import glob

d1 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
idx = d1['scene_ids'].index('LivingDiningRoom-9790_13_013')
orig_t = np.array(d1['translations'][idx])
sizes = np.array(d1['sizes'][idx]) * 2.0

obj_files = sorted(glob.glob('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_*.obj'))

for i in [0, 3]:
    m = trimesh.load(obj_files[i], force='mesh')
    b = m.bounds
    mesh_center = (b[0] + b[1]) / 2.0
    mesh_size = b[1] - b[0]
    
    print(f"Object {i}:")
    print(f"  Mesh center: {mesh_center}")
    print(f"  JSON transl: {orig_t[i]}")
    print(f"  Mesh size  : {mesh_size}")
    print(f"  JSON size  : {sizes[i]}")
