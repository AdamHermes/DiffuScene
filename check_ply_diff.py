import trimesh

m1 = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013.ply', force='mesh')
m2 = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved/LivingDiningRoom-9790_13_013.ply', force='mesh')

print(f"Orig vertices: {len(m1.vertices)}")
print(f"Res vertices: {len(m2.vertices)}")

v1_sum = m1.vertices.sum(axis=0)
v2_sum = m2.vertices.sum(axis=0)
print(f"Orig sum: {v1_sum}")
print(f"Res sum: {v2_sum}")
