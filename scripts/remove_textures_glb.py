import argparse
import os
import glob
import trimesh
import numpy as np

def remove_textures(glb_path):
    try:
        scene = trimesh.load(glb_path, process=False)
        if isinstance(scene, trimesh.Scene):
            for geom in scene.geometry.values():
                geom.visual = trimesh.visual.ColorVisuals(mesh=geom, face_colors=np.array([200, 200, 200, 255]))
        else:
            scene.visual = trimesh.visual.ColorVisuals(mesh=scene, face_colors=np.array([200, 200, 200, 255]))
        scene.export(glb_path)
        print(f"Processed: {glb_path}")
    except Exception as e:
        print(f"Failed to process {glb_path}: {e}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb_dir", required=True, help="Directory containing GLB files")
    args = parser.parse_args()
    
    glb_files = glob.glob(os.path.join(args.glb_dir, "*.glb"))
    print(f"Found {len(glb_files)} GLB files to strip textures from.")
    
    for f in glb_files:
        remove_textures(f)

if __name__ == "__main__":
    main()
