with open('/Volumes/Baracuda/SavedGithubRepository/DiffuScene/scripts/render_resolved.py', 'r') as f:
    lines = f.readlines()
with open('/Volumes/Baracuda/SavedGithubRepository/DiffuScene/scripts/render_resolved.py', 'w') as f:
    for line in lines:
        if line.startswith("        if not renderables:") or line.startswith("            print(f\"  WARNING") or line.startswith("            skipped_count") or line.startswith("            continue") or line.startswith("        # Draw red room") or line.startswith("        all_verts =") or line.startswith("        for r in renderables") or line.startswith("            all_verts.append") or line.startswith("        if all_verts:") or line.startswith("            all_verts =") or line.startswith("            min_x, max_x") or line.startswith("            min_z, max_z") or line.startswith("            # Create lines") or line.startswith("            y = 0") or line.startswith("            corners =") or line.startswith("                [min_x") or line.startswith("                [max_x") or line.startswith("            ]") or line.startswith("            room_bounds") or line.startswith("            renderables.append"):
            f.write(line.replace("        ", "        ", 1))
        else:
            f.write(line)
