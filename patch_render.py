with open('/Volumes/Baracuda/SavedGithubRepository/DiffuScene/scripts/render_resolved.py', 'r') as f:
    content = f.read()

import re

# We need to add `Lines` import and room boundary drawing
content = content.replace("from simple_3dviz import Mesh, Scene", "from simple_3dviz import Mesh, Scene, Lines")

# Add bounding box calculation and drawing before rendering
injection = """
        if not renderables:
            print(f"  WARNING: Could not load any meshes for {scene_id}")
            skipped_count += 1
            continue
            
        # Draw red room boundary using max/min X and Z
        all_verts = []
        for r in renderables:
            all_verts.append(r.vertices)
        
        if all_verts:
            all_verts = np.vstack(all_verts)
            min_x, max_x = all_verts[:, 0].min(), all_verts[:, 0].max()
            min_z, max_z = all_verts[:, 2].min(), all_verts[:, 2].max()
            
            # Create lines for the rectangle
            y = 0.0
            corners = [
                [min_x, y, min_z],
                [max_x, y, min_z],
                [max_x, y, max_z],
                [min_x, y, max_z],
                [min_x, y, min_z]
            ]
            
            room_bounds = Lines(corners, colors=[1.0, 0.0, 0.0], width=0.03)
            renderables.append(room_bounds)
"""

content = re.sub(
    r"        if not renderables:.*?continue", 
    injection.strip(), 
    content, 
    flags=re.DOTALL
)

with open('/Volumes/Baracuda/SavedGithubRepository/DiffuScene/scripts/render_resolved.py', 'w') as f:
    f.write(content)
