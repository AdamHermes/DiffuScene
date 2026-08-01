import trimesh
import sys
ply = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013.ply')
print(ply.bounds)
