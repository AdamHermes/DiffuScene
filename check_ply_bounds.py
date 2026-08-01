import trimesh

m1 = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013.ply', force='mesh')
m2 = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved/LivingDiningRoom-9790_13_013.ply', force='mesh')

print(f"Orig bounds: \n{m1.bounds}")
print(f"Res bounds: \n{m2.bounds}")
