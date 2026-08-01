import trimesh
import glob

files = glob.glob('/Users/lehoangan/Downloads/merge_d/merged/scene_mesh/LivingDiningRoom-9790_13_013_individual_objs/*_2.obj')
if not files:
    print("No file found for object 2")
else:
    mesh = trimesh.load(files[0])
    print(f"Object 2 bounds: {mesh.bounds}")

