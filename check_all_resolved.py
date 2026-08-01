import trimesh
import numpy as np
import json
import glob

# Load original translations
d1 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params.json'))
d2 = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'))
idx = d1['scene_ids'].index('LivingDiningRoom-9790_13_013')

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
    return np.all(b1[0] <= b2[1]) and np.all(b1[1] >= b2[0])

manager = trimesh.collision.CollisionManager()
for i, m in meshes:
    manager.add_object(str(i), m)

is_col, contacts = manager.in_collision_internal(return_names=True)
print(f"Is there a collision in the resolved meshes? {is_col}")
for c in contacts:
    print(f"Collision between {c[0]} and {c[1]}")
