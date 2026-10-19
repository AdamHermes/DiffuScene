# 
# Copyright (C) 2021 NVIDIA Corporation.  All rights reserved.
# Licensed under the NVIDIA Source Code License.
# See LICENSE at https://github.com/nv-tlabs/ATISS.
# Authors: Despoina Paschalidou, Amlan Kar, Maria Shugrina, Karsten Kreis,
#          Andreas Geiger, Sanja Fidler
# 

"""Script for computing the FID score between real and synthesized scenes.
"""
import argparse
import os
import sys

import torch

import numpy as np
from PIL import Image

from cleanfid import fid

import shutil

from scene_synthesis.datasets.splits_builder import CSVSplitsBuilder
from scene_synthesis.datasets.threed_front import CachedThreedFront


class ThreedFrontRenderDataset(object):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset[idx].image_path
        img = Image.open(image_path)
        return img


def main(argv):
    parser = argparse.ArgumentParser(
        description=("Compute the FID scores between the real and the "
                     "synthetic images")
    )
    parser.add_argument(
        "path_to_real_renderings",
        help="Path to the folder containing the real renderings (either flat folder of PNGs or structured dataset folder)"
    )
    parser.add_argument(
        "path_to_synthesized_renderings",
        help="Path to the folder containing the synthesized images"
    )
    parser.add_argument(
        "path_to_annotations",
        nargs="?",
        default=None,
        help="Path to the folder/file containing the annotations (optional if real renderings is a flat folder of PNGs)"
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="if compare all"
    )
    parser.add_argument(
        "--splits",
        default=None,
        help="Comma-separated splits to use for real images, e.g. 'test' or 'train,val' (default: train,val)"
    )
    parser.add_argument(
        "--rendered_scene",
        default=None,
        help="Name of the rendered scene image file to compare (e.g. rendered_scene_notexture_256.png)"
    )

    args = parser.parse_args(argv)

    path_to_test_real = "./test_real/"
    path_to_test_fake = "./test_fake/"
    if os.path.exists(path_to_test_real):
        shutil.rmtree(path_to_test_real)
    os.makedirs(path_to_test_real, exist_ok=True)

    if os.path.exists(path_to_test_fake):
        shutil.rmtree(path_to_test_fake)
    os.makedirs(path_to_test_fake, exist_ok=True)

    try:
        # Check if path_to_real_renderings is a flat folder with PNG images
        real_png_files = sorted([
            f for f in os.listdir(args.path_to_real_renderings)
            if f.endswith(".png")
        ])

        if len(real_png_files) > 0:
            print(f"Detected flat folder with {len(real_png_files)} PNG images in: {args.path_to_real_renderings}")
            if args.path_to_annotations and (args.splits or not args.compare_all):
                splits_builder = CSVSplitsBuilder(args.path_to_annotations)
                chosen_splits = [s.strip() for s in args.splits.split(",")] if args.splits else ["train", "val"]
                allowed_scene_ids = set(splits_builder.get_splits(chosen_splits))
                selected_pngs = []
                for f in real_png_files:
                    base = f[:-4]
                    scene_id = base.split("_")[1] if "_" in base else base
                    if scene_id in allowed_scene_ids:
                        selected_pngs.append(f)
                if len(selected_pngs) > 0:
                    real_png_files = selected_pngs
                    print(f"Filtered to {len(real_png_files)} images matching splits: {chosen_splits}")
                else:
                    print(f"Note: None of filenames matched scene IDs in split {chosen_splits}. Using all {len(real_png_files)} images.")

            for i, f in enumerate(real_png_files):
                shutil.copyfile(
                    os.path.join(args.path_to_real_renderings, f),
                    f"{path_to_test_real}/{i:05d}.png"
                )
            print('number of real images :', len(real_png_files))
        else:
            # Structured 3D-FRONT dataset folder
            config = dict(
                train_stats="dataset_stats.txt",
                room_layout_size="256,256"
            )
            if args.rendered_scene:
                config["rendered_scene"] = args.rendered_scene

            if not args.path_to_annotations:
                raise ValueError("path_to_annotations is required when using a structured dataset folder.")

            splits_builder = CSVSplitsBuilder(args.path_to_annotations)
            if args.splits:
                chosen_splits = [s.strip() for s in args.splits.split(",")]
            elif args.compare_all:
                chosen_splits = ["train", "val", "test"]
            else:
                chosen_splits = ["train", "val"]

            print("Using splits for real images:", chosen_splits)
            test_real = ThreedFrontRenderDataset(CachedThreedFront(
                args.path_to_real_renderings,
                config=config,
                scene_ids=splits_builder.get_splits(chosen_splits)
            ))

            for i, di in enumerate(test_real):
                di.save("{}/{:05d}.png".format(path_to_test_real, i))
            print('number of real images :', len(test_real))

        print("Generating temporary a folder with test_fake images...")
        synthesized_images = sorted([
            os.path.join(args.path_to_synthesized_renderings, oi)
            for oi in os.listdir(args.path_to_synthesized_renderings)
            if oi.endswith(".png")
        ])
        print('number of synthesized images :', len(synthesized_images))

        for i, fi in enumerate(synthesized_images):
            shutil.copyfile(fi, "{}/{:05d}.png".format(path_to_test_fake, i))

        # Compute the FID score
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Computing metrics using device:", device)
        fid_score = fid.compute_fid(path_to_test_real, path_to_test_fake, device=device)
        print('fid score:', fid_score)
        kid_score = fid.compute_kid(path_to_test_real, path_to_test_fake, device=device)
        print('kid score:', kid_score)
    finally:
        if os.path.exists(path_to_test_real):
            shutil.rmtree(path_to_test_real)
        if os.path.exists(path_to_test_fake):
            shutil.rmtree(path_to_test_fake)


if __name__ == "__main__":
    main(None)

# python compute_fid_scores.py /cluster/balrog/jtang/3d_front_processed/bedrooms_notexture_nofloor_whiteground/ /cluster/balrog/jtang/ATISS_exps/diffusion_bedrooms_objfeats_lat32_v/gen_clip_24000/ ../config/bedroom_threed_front_splits.csv
# python compute_fid_scores.py /cluster/balrog/jtang/3d_front_processed/livingrooms_notexture_nofloor_whiteground/ /cluster/balrog/jtang/ATISS_exps/diffusion_livingrooms_permaug_fixedrotaug_unet1d_dim512_nomask_instancond_cosinangle_ddpm_separateclsbbox/gen_top2down_notexture_nofloor-58000/ ../config/livingroom_threed_front_splits.csv 
# python compute_fid_scores.py /cluster/balrog/jtang/3d_front_processed/diningrooms_notexture_nofloor_whiteground/ /cluster/balrog/jtang/ATISS_exps/diffusion_diningrooms_permaug_fixedrotaug_unet1d_dim512_nomask_instancond_cosinangle_ddpm_separateclsbbox_modinstan_objfeats_biou/gen_top2down_notexture_nofloor_retrifeats_combsize-59800/ ../config/diningroom_threed_front_splits.csv 