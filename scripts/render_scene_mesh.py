import argparse
import os
import sys
import numpy as np
import seaborn as sns
from tqdm import tqdm
from simple_3dviz import Mesh, Scene
from simple_3dviz.utils import save_frame
from pyrr import Matrix44

def render_top2down(scene, renderables, frame_path=None):
    scene.clear()
    for r in renderables:
        if isinstance(r, Mesh):
            r.mode = "shading"
        scene.add(r)
    scene.render()
    if frame_path is not None:
        save_frame(frame_path, scene.frame)
    return np.copy(scene.frame)

def main(argv):
    parser = argparse.ArgumentParser(
        description="Render top-down images directly from scene_mesh obj files without needing 3D-FUTURE-model"
    )
    parser.add_argument(
        "--scene_mesh_dir",
        required=True,
        help="Path to the scene_mesh directory (e.g. bedroom/scene_mesh)"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory for rendered images"
    )
    parser.add_argument(
        "--without_floor",
        action="store_true",
        default=True,
        help="Do not render floor (white background)"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Output image size"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=3.1,
        help="Orthographic camera size"
    )

    args = parser.parse_args(argv)

    if not os.path.exists(args.scene_mesh_dir):
        print(f"ERROR: Missing scene_mesh folder: {args.scene_mesh_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    bg = [1, 1, 1, 1] if args.without_floor else [0, 0, 0, 1]
    scene_top2down = Scene(size=(args.image_size, args.image_size), background=bg)
    scene_top2down.up_vector = (0, 0, -1)
    scene_top2down.camera_target = (0, 0, 0)
    scene_top2down.camera_position = (0, 4, 0)
    scene_top2down.light = (0, 4, 0)
    scene_top2down.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )

    color_palette = np.array(sns.color_palette('hls', 23))
    
    scene_folders = [d for d in os.listdir(args.scene_mesh_dir) if d.endswith("_individual_objs")]
    print(f"Found {len(scene_folders)} scenes to render in {args.scene_mesh_dir}")

    for idx, folder_name in enumerate(tqdm(scene_folders, desc="Rendering")):
        scene_id = folder_name.replace("_individual_objs", "")
        individual_objs_dir = os.path.join(args.scene_mesh_dir, folder_name)
        
        obj_files = sorted([f for f in os.listdir(individual_objs_dir) if f.endswith(".obj")])
        if not obj_files:
            continue

        renderables = []
        for obj_idx, obj_file in enumerate(obj_files):
            obj_path = os.path.join(individual_objs_dir, obj_file)
            # Pick a consistent color per object index for now
            color_index = min(obj_idx, len(color_palette) - 1)
            color = color_palette[color_index, :]

            try:
                mesh = Mesh.from_file(obj_path, color=color)
                renderables.append(mesh)
            except Exception as e:
                print(f"  WARNING: Cannot load {obj_path}: {e}")

        if not renderables:
            continue

        out_img_path = os.path.join(args.output_dir, f"{scene_id}.png")
        try:
            render_top2down(scene_top2down, renderables, frame_path=out_img_path)
        except Exception as e:
            print(f"  ERROR: Render failed for {scene_id}: {e}")

    print(f"Done! Images saved to {args.output_dir}")

if __name__ == "__main__":
    main(sys.argv[1:])
