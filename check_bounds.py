import trimesh
import glob
import numpy as np
files = glob.glob('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_*.obj')
all_verts = []
for f in files:
    m = trimesh.load(f, force='mesh')
    all_verts.append(m.vertices)
v = np.vstack(all_verts)
print("Min:", v.min(axis=0))
print("Max:", v.max(axis=0))
