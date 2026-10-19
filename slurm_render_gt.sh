#!/bin/bash
#SBATCH --job-name=diffuscene_render_gt
#SBATCH --output=logs/render_gt_%j.out
#SBATCH --error=logs/render_gt_%j.err
#SBATCH --nodelist=xgpd2            # Node xgpd2 có sẵn xvfb
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=00:30:00

mkdir -p logs

xvfb-run -a python scripts/render_gt_notexture.py \
    --dataset_dir "./datasets/3d_front_processed/diningrooms_objfeats_32_64" \
    --threed_future_pkl "./datasets/3d_front_processed/threed_future_model_diningroom.pkl" \
    --path_to_models "./datasets/3D-FUTURE-model" \
    --output_dir "./datasets/3d_front_processed/diningroom_image" \
    --annotation_file "config/diningroom_threed_front_splits.csv" \
    --splits test
