"""Script to render top-down orthographic ground truth images without texture and without floor
directly from preprocessed dataset (boxes.npz) and pickled 3D-FUTURE models.
"""
import argparse
import os
import sys
import numpy as np
import torch
import seaborn as sns
from pyrr import Matrix44
from simple_3dviz import Scene, Mesh
from simple_3dviz.utils import save_frame
from tqdm import tqdm

from scene_synthesis.datasets.splits_builder import CSVSplitsBuilder
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.datasets.threed_front import CachedThreedFront


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
        description="Render ground truth scenes with no texture, no floor, and white background"
    )
    parser.add_argument(
        "--dataset_dir",
        default="./datasets/3d_front_processed/bedrooms_objfeats_32_64",
        help="Path to preprocessed dataset directory containing scene folders"
    )
    parser.add_argument(
        "--threed_future_pkl",
        default="./datasets/3d_front_processed/threed_future_model_bedroom.pkl",
        help="Path to pickled 3D-FUTURE models"
    )
    parser.add_argument(
        "--path_to_models",
        default=None,
        help="Path to directory containing 3D-FUTURE-model CAD folders (e.g. ./datasets/3D-FUTURE-model)"
    )
    parser.add_argument(
        "--output_dir",
        default="./datasets/3d_front_processed/bedrooms_notexture_nofloor_whiteground",
        help="Path to output directory for rendered images"
    )
    parser.add_argument(
        "--annotation_file",
        default="config/bedroom_threed_front_splits.csv",
        help="Path to splits csv"
    )
    parser.add_argument(
        "--splits",
        default="test",
        help="Comma-separated splits to render, e.g. 'test' or 'train,val,test' (default: 'test')"
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=256,
        help="Image size (default: 256)"
    )
    parser.add_argument(
        "--room_side",
        type=float,
        default=3.1,
        help="Orthographic camera boundary (default: 3.1)"
    )

    args = parser.parse_args(argv)

    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load splits
    splits_builder = CSVSplitsBuilder(args.annotation_file)
    chosen_splits = [s.strip() for s in args.splits.split(",")]
    allowed_scene_ids = set(splits_builder.get_splits(chosen_splits))
    print(f"Loaded {len(allowed_scene_ids)} scene IDs for splits: {chosen_splits}")

    # 2. Match scene directories
    scene_dirs = sorted([
        d for d in os.listdir(args.dataset_dir)
        if os.path.isdir(os.path.join(args.dataset_dir, d)) and "_" in d and d.split("_")[1] in allowed_scene_ids
    ])
    print(f"Found {len(scene_dirs)} matching scenes in {args.dataset_dir}")

    # 3. Load 3D-FUTURE models
    print(f"Loading 3D-FUTURE models from {args.threed_future_pkl}...")
    objects_dataset = ThreedFutureDataset.from_pickled_dataset(args.threed_future_pkl)
    jid_to_model = {obj.model_jid: obj for obj in objects_dataset.objects}
    print(f"Indexed {len(jid_to_model)} 3D models by JID.")

    # 4. Setup simple_3dviz Scene (use xvfb-run on headless servers)
    scene_top2down = Scene(size=(args.image_size, args.image_size), background=[1, 1, 1, 1])
    scene_top2down.up_vector = (0, 0, -1)
    scene_top2down.camera_target = (0, 0, 0)
    scene_top2down.camera_position = (0, 4, 0)
    scene_top2down.light = (0, 4, 0)
    scene_top2down.camera_matrix = Matrix44.orthogonal_projection(
        left=-args.room_side, right=args.room_side,
        bottom=args.room_side, top=-args.room_side,
        near=0.1, far=6
    )

    # 5. Render scenes
    for s_dir in tqdm(scene_dirs, desc="Rendering scenes"):
        room_dir = os.path.join(args.dataset_dir, s_dir)
        boxes_path = os.path.join(room_dir, "boxes.npz")
        if not os.path.exists(boxes_path):
            continue

        data = np.load(boxes_path, allow_pickle=True)
        jids = data["jids"]
        class_labels = data["class_labels"]
        translations = data["translations"]
        sizes = data["sizes"]
        angles = data["angles"]

        # Color palette for classes (excluding start and end tokens)
        num_classes = class_labels.shape[1] - 2
        color_palette = np.array(sns.color_palette("hls", num_classes))
        class_indices = class_labels.argmax(axis=1)

        renderables = []
        for j in range(len(jids)):
            jid = jids[j]
            if isinstance(jid, bytes):
                jid = jid.decode("utf-8")

            if jid not in jid_to_model:
                continue

            furniture = jid_to_model[jid]
            cls_idx = class_indices[j]
            c = color_palette[cls_idx]
            if args.path_to_models:
                model_path = os.path.join(args.path_to_models, jid, "raw_model.obj")
            else:
                model_path = furniture.raw_model_path

            if not os.path.exists(model_path):
                continue

            try:
                raw_mesh = Mesh.from_file(model_path, color=c)
                raw_mesh.scale(furniture.scale)

                bbox = raw_mesh.bbox
                centroid = (bbox[0] + bbox[1]) / 2.0

                trans = translations[j]
                theta = angles[j, 0] if len(angles.shape) > 1 else angles[j]

                R = np.zeros((3, 3))
                R[0, 0] = np.cos(theta)
                R[0, 2] = -np.sin(theta)
                R[2, 0] = np.sin(theta)
                R[2, 2] = np.cos(theta)
                R[1, 1] = 1.0

                raw_mesh.affine_transform(t=-centroid)
                raw_mesh.affine_transform(R=R, t=trans)
                renderables.append(raw_mesh)
            except Exception as e:
                print(f"Failed to load or transform object {jid}: {e}")
                continue

        # Save to both target output_dir (as flat or structured) and inside room_dir
        out_flat_img = os.path.join(args.output_dir, f"{s_dir}.png")
        render_top2down(scene_top2down, renderables, frame_path=out_flat_img)

        # Also copy rendered_scene_notexture_256.png into the scene folder
        out_scene_img = os.path.join(room_dir, f"rendered_scene_notexture_{args.image_size}.png")
        if not os.path.exists(out_scene_img):
            render_top2down(scene_top2down, renderables, frame_path=out_scene_img)

    print(f"Finished rendering ground truth images to {args.output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
