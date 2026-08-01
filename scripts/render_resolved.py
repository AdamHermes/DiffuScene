import argparse
import os
import sys
import json
import numpy as np
import seaborn as sns

from simple_3dviz import Mesh, Scene, Lines
from simple_3dviz.utils import save_frame

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

def load_obj_as_mesh(obj_path, color, delta_t=None):
    try:
        mesh = Mesh.from_file(obj_path, color=color)
        if delta_t is not None and not np.allclose(delta_t, 0.0):
            # Apply translation offset correctly using affine_transform
            mesh.affine_transform(t=delta_t)
        return mesh
    except Exception as e:
        print(f"  WARNING: Cannot load {obj_path}: {e}, skipping...")
        return None

def main(argv):
    parser = argparse.ArgumentParser(
        description="Render top-down images from scene_mesh obj files using resolved collision translations"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Path to the output directory (contains scene_mesh/, collision_params.json, and collision_params_resolved.json)"
    )
    parser.add_argument(
        "--render_imgs_dir",
        default="rendered_resolved",
        help="Output folder name for rendered images (created inside output_dir)"
    )
    parser.add_argument(
        "--scene_mesh_dir",
        default="scene_mesh",
        help="Folder containing original scene meshes"
    )
    parser.add_argument(
        "--orig_json",
        default="collision_params.json",
        help="Original JSON with initial translations"
    )
    parser.add_argument(
        "--resolved_json",
        default="collision_params_resolved.json",
        help="Resolved JSON with updated translations"
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

    orig_json_path = os.path.join(args.output_dir, args.orig_json)
    resolved_json_path = os.path.join(args.output_dir, args.resolved_json)
    scene_mesh_dir = os.path.join(args.output_dir, args.scene_mesh_dir)
    render_out_dir = os.path.join(args.output_dir, args.render_imgs_dir)

    if not os.path.exists(orig_json_path):
        print(f"ERROR: Missing original JSON: {orig_json_path}")
        sys.exit(1)
    if not os.path.exists(resolved_json_path):
        print(f"ERROR: Missing resolved JSON: {resolved_json_path}")
        sys.exit(1)
    if not os.path.exists(scene_mesh_dir):
        print(f"ERROR: Missing scene_mesh folder: {scene_mesh_dir}")
        sys.exit(1)

    os.makedirs(render_out_dir, exist_ok=True)

    with open(orig_json_path, "r", encoding="utf-8") as f:
        orig_data = json.load(f)
    with open(resolved_json_path, "r", encoding="utf-8") as f:
        res_data = json.load(f)

    scene_ids = res_data["scene_ids"]
    class_labels_all = res_data["class_labels"]
    orig_translations = orig_data["translations"]
    res_translations = res_data["translations"]

    print(f"Found {len(scene_ids)} scenes to render.")

    # Lazy import to avoid crash if pyrr is not needed, but we don't need Matrix44 here anyway
    from pyrr import Matrix44
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

    if class_labels_all and class_labels_all[0]:
        n_classes = len(class_labels_all[0][0])
        color_palette = np.array(sns.color_palette('hls', max(n_classes - 2, 1)))
    else:
        color_palette = np.array(sns.color_palette('hls', 23))

    rendered_count = 0
    skipped_count = 0

    for idx, scene_id in enumerate(scene_ids):
        individual_objs_dir = os.path.join(scene_mesh_dir, scene_id + "_individual_objs")

        if not os.path.exists(individual_objs_dir):
            print(f"[{idx+1}/{len(scene_ids)}] SKIP: Missing folder {individual_objs_dir}")
            skipped_count += 1
            continue

        obj_files = sorted([f for f in os.listdir(individual_objs_dir) if f.endswith(".obj")])
        if not obj_files:
            print(f"[{idx+1}/{len(scene_ids)}] SKIP: No .obj files in {individual_objs_dir}")
            skipped_count += 1
            continue

        print(f"[{idx+1}/{len(scene_ids)}] Rendering scene: {scene_id} ({len(obj_files)} objects)")

        scene_class_labels = class_labels_all[idx] if idx < len(class_labels_all) else []
        orig_trans = orig_translations[idx] if idx < len(orig_translations) else []
        res_trans = res_translations[idx] if idx < len(res_translations) else []

        renderables = []
        for obj_idx, obj_file in enumerate(obj_files):
            obj_path = os.path.join(individual_objs_dir, obj_file)

            if obj_idx < len(scene_class_labels):
                one_hot = np.array(scene_class_labels[obj_idx])
                class_index = int(one_hot.argmax())
                color_index = min(class_index, len(color_palette) - 1)
                color = color_palette[color_index, :]
            else:
                color = np.array([0.5, 0.5, 0.5])

            # Calculate translation delta
            delta_t = None
            if obj_idx < len(orig_trans) and obj_idx < len(res_trans):
                delta_t = np.array(res_trans[obj_idx]) - np.array(orig_trans[obj_idx])

            mesh = load_obj_as_mesh(obj_path, color, delta_t)
            if mesh is not None:
                renderables.append(mesh)

        if not renderables:
            print(f"  WARNING: Could not load any meshes for {scene_id}")
            skipped_count += 1
            continue
            


        out_img_path = os.path.join(render_out_dir, f"{scene_id}.png")
        try:
            render_top2down(scene_top2down, renderables, frame_path=out_img_path)
            print(f"  -> Saved: {out_img_path}")
            rendered_count += 1
        except Exception as e:
            print(f"  ERROR: Render failed for {scene_id}: {e}")
            skipped_count += 1

    print(f"\nDone! Rendered: {rendered_count}, Skipped: {skipped_count}")
    print(f"Output images: {render_out_dir}")

if __name__ == "__main__":
    main(sys.argv[1:])
