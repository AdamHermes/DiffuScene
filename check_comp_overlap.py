import trimesh
import numpy as np

m = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved/LivingDiningRoom-9790_13_013.ply', force='mesh')
components = m.split(only_watertight=False)
print(f"Loaded {len(components)} components from resolved PLY.")

def bounds_intersect(b1, b2):
    return np.all(b1[0] < b2[1]) and np.all(b1[1] > b2[0])

overlaps = 0
for a in range(len(components)):
    for b in range(a+1, len(components)):
        if bounds_intersect(components[a].bounds, components[b].bounds):
            overlaps += 1
print(f"Total overlaps between components: {overlaps}")
