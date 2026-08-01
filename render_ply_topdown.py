import trimesh
import matplotlib.pyplot as plt
import numpy as np

m = trimesh.load('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved/LivingDiningRoom-9790_13_013.ply', force='mesh')

# Get all connected components to treat them as individual objects roughly
components = m.split(only_watertight=False)
print(f"Found {len(components)} components")

fig, ax = plt.subplots(figsize=(10, 10))

for c in components:
    # Get 2D bounds in XZ plane
    b = c.bounds
    min_x, min_z = b[0][0], b[0][2]
    max_x, max_z = b[1][0], b[1][2]
    width = max_x - min_x
    height = max_z - min_z
    
    rect = plt.Rectangle((min_x, min_z), width, height, fill=False, edgecolor=np.random.rand(3,), linewidth=2)
    ax.add_patch(rect)

ax.autoscale()
plt.axis('equal')
plt.savefig('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh_resolved/LivingDiningRoom-9790_13_013_topdown.png')
