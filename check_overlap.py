import json
import trimesh
import numpy as np

d = json.load(open('/Users/lehoangan/Downloads/merge_d/merged/collision_params_resolved.json'))
idx = d['scene_ids'].index('LivingDiningRoom-9790_13_013')

m_ply = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved/LivingDiningRoom-9790_13_013.ply')
# We need to find if coffee table and wardrobe overlap in the resolved .ply!
# Wait, trimesh loads them as a single concatenated mesh.
# Let's load the individual resolved meshes by applying res_t - orig_t!
