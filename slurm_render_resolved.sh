#!/bin/bash
#SBATCH --job-name=diffuscene_render_res
#SBATCH --output=logs/render_res_%j.out
#SBATCH --error=logs/render_res_%j.err
#SBATCH --nodelist=xgpd2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

mkdir -p logs

echo "Rendering resolved bedroom..."
xvfb-run -a python scripts/render_resolved.py --output_dir "./models/pretrained_diffusion/bedrooms_uncond/gen_top2down_notexture_nofloor" --orig_json collision_params_merged.json --resolved_json collision_params_resolved.json --render_imgs_dir rendered_resolved

echo "Rendering resolved livingroom..."
xvfb-run -a python scripts/render_resolved.py --output_dir "./models/pretrained_diffusion/livingrooms_uncond/gen_top2down_notexture_nofloor" --orig_json collision_params_merged.json --resolved_json collision_params_resolved.json --render_imgs_dir rendered_resolved

echo "Rendering resolved diningroom..."
xvfb-run -a python scripts/render_resolved.py --output_dir "./models/pretrained_diffusion/diningrooms_uncond/gen_top2down_notexture_nofloor" --orig_json collision_params_merged.json --resolved_json collision_params_resolved.json --render_imgs_dir rendered_resolved

echo "Done!"
