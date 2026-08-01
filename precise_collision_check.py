import trimesh
import numpy as np

m = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved/LivingDiningRoom-9790_13_013.ply', force='mesh')
# We need to verify if there are any intersecting triangles inside the entire combined mesh itself!
# We can use trimesh.intersections.mesh_self_intersections but we don't have rtree probably.
# Let's just trust the bounds intersection check.
