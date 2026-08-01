with open('/Volumes/Baracuda/SavedGithubRepository/DiffuScene/scripts/render_resolved.py', 'r') as f:
    lines = f.readlines()
with open('/Volumes/Baracuda/SavedGithubRepository/DiffuScene/scripts/render_resolved.py', 'w') as f:
    skip = False
    for line in lines:
        if line.startswith("        # Draw red room boundary using max/min X and Z"):
            skip = True
        if skip and line.strip() == "renderables.append(room_bounds)":
            skip = False
            continue
        if not skip:
            f.write(line)
