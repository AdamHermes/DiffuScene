import trimesh
import numpy as np

# Load individual resolved meshes, check if they intersect
files = [
    '/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_004.obj',
    '/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_006.obj',
    '/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_007.obj'
]

import json
d1 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
d2 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'))
idx = d1['scene_ids'].index('LivingDiningRoom-9790_13_013')

meshes = []
for i in [4, 6, 7]:
    mesh = trimesh.load(f'/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_00{i}.obj', force='mesh')
    orig_t = np.array(d1['translations'][idx][i])
    res_t = np.array(d2['translations'][idx][i])
    mesh.apply_translation(res_t - orig_t)
    meshes.append((i, mesh))

# Just check bounds intersection
def bounds_intersect(b1, b2):
    return np.all(b1[0] <= b2[1]) and np.all(b1[1] >= b2[0])

for a in range(len(meshes)):
    for b in range(a+1, len(meshes)):
        b1 = meshes[a][1].bounds
        b2 = meshes[b][1].bounds
        if bounds_intersect(b1, b2):
            print(f"Mesh {meshes[a][0]} and Mesh {meshes[b][0]} BOUNDS INTERSECT!")
            
            # Check actual mesh overlap using trimesh collision manager
            manager = trimesh.collision.CollisionManager()
            manager.add_object('a', meshes[a][1])
            manager.add_object('b', meshes[b][1])
            col = manager.in_collision_internal()
            print(f"Mesh collision: {col}")
        else:
            print(f"Mesh {meshes[a][0]} and Mesh {meshes[b][0]} bounds DO NOT intersect.")

