import trimesh
import sys
mesh = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/object_002.obj')
print(f"Object 2 bounds: {mesh.bounds}")
