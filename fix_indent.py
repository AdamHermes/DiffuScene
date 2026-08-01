with open('/Volumes/Baracuda/SavedGithubRepository/DiffuScene/scripts/render_resolved.py', 'r') as f:
    lines = f.readlines()

for i, line in enumerate(lines):
    if line.startswith("        all_verts = []"):
        print(f"Found at {i}")
        # Need to fix the indentation of this block
        pass

