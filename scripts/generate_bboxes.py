import argparse
import os
import sys
import numpy as np
import torch
import json

from training_utils import load_config
from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.networks import build_network

def main(argv):
    parser = argparse.ArgumentParser(description="Generate bounding boxes using a previously trained model")
    parser.add_argument("config_file", help="Path to experiment config")
    parser.add_argument("output_directory", default="/tmp/", help="Path to output directory")
    parser.add_argument("--weight_file", default=None, help="Path to a pretrained model")
    parser.add_argument("--n_sequences", default=10, type=int, help="Number of sequences to generate")
    parser.add_argument("--scene_id", default=None, help="The scene id to be used for conditioning")
    parser.add_argument("--clip_denoised", action="store_true", help="if clip_denoised")
    parser.add_argument("--fix_order", action="store_true", help="if use fix order")
    args = parser.parse_args(argv)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Running code on", device)

    if not os.path.exists(args.output_directory):
        os.makedirs(args.output_directory)

    config = load_config(args.config_file)
    # Dynamically fix hardcoded paths to point to the local Colab dataset folder
    import os
    colab_dataset_path = "../datasets" # Since script runs from scripts/
    if os.path.exists("../datasets/3D-Front_preprocessed"):
        colab_dataset_path = "../datasets/3D-Front_preprocessed"
    if os.path.exists("../datasets/3d_front_processed"):
        colab_dataset_path = "../datasets/3d_front_processed"
    config["data"]["dataset_directory"] = config["data"]["dataset_directory"].replace("/cluster/balrog/jtang/3d_front_processed", colab_dataset_path)
    if "train_stats_file" in config["network"].get("diffusion_kwargs", {}):
        config["network"]["diffusion_kwargs"]["train_stats_file"] = config["network"]["diffusion_kwargs"]["train_stats_file"].replace("/cluster/balrog/jtang/3d_front_processed", colab_dataset_path)
    if 'text' in config["data"]["encoding_type"] and 'textfix' not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] = config["data"]["encoding_type"].replace('text', 'textfix')
    if "no_prm" not in config["data"]["encoding_type"]:
        config["data"]["encoding_type"] += "_no_prm"

    raw_dataset, dataset = get_dataset_raw_and_encoded(
        config["data"],
        filter_fn=filter_function(config["data"], split=config["validation"].get("splits", ["test"])),
        split=config["validation"].get("splits", ["test"])
    )
    print("Loaded {} scenes with {} object types".format(len(dataset), dataset.n_object_types))
    
    network, _, _ = build_network(
        dataset.feature_size, dataset.n_classes,
        config, args.weight_file, device=device
    )
    network.eval()

    given_scene_id = None
    if args.scene_id:
        for i, di in enumerate(raw_dataset):
            if str(di.scene_id) == args.scene_id:
                given_scene_id = i

    export_data = {
        "scene_ids": [],
        "class_labels": [],
        "translations": [],
        "sizes": [],
        "angles": []
    }

    for i in range(args.n_sequences):
        scene_idx = given_scene_id or (i if args.fix_order and i < len(dataset) else (i % len(dataset) if args.fix_order else np.random.choice(len(dataset))))
        current_scene = raw_dataset[scene_idx]
        samples = dataset[scene_idx]
        print("{} / {}: Generating layout for scene {}".format(i, args.n_sequences, current_scene.scene_id))

        room_mask = current_scene.room_mask.unsqueeze(0).unsqueeze(0) if hasattr(current_scene, 'room_mask') else None
        
        # Simplified generation
        bbox_params = network.generate_layout(
            room_mask=room_mask.to(device) if room_mask is not None else None,
            num_points=config["network"]["sample_num_points"],
            point_dim=config["network"]["point_dim"],
            text=samples['description'] if 'description' in samples.keys() else None,
            device=device,
            clip_denoised=args.clip_denoised,
            batch_seeds=torch.arange(i, i+1),
        )

        boxes = dataset.post_process(bbox_params)
        
        unique_scene_id = "{}_{}_{:03d}".format(current_scene.scene_id, scene_idx, i)
        export_data["scene_ids"].append(unique_scene_id)
        export_data["class_labels"].append(boxes["class_labels"][0].cpu().numpy().tolist())
        export_data["translations"].append(boxes["translations"][0].cpu().numpy().tolist())
        export_data["sizes"].append(boxes["sizes"][0].cpu().numpy().tolist())
        export_data["angles"].append(boxes["angles"][0].cpu().numpy().tolist())

    json_path = os.path.join(args.output_directory, "collision_params.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2)
    print(f"Saved collision parameters to {json_path}")

if __name__ == "__main__":
    main(sys.argv[1:])
