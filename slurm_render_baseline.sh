#!/bin/bash
#SBATCH --job-name=diffuscene_render_base
#SBATCH --output=logs/render_base_%j.out
#SBATCH --error=logs/render_base_%j.err
#SBATCH --nodelist=xgpd2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00

set -euo pipefail

# Submit from the repository root after creating logs/:
#   cd ~/DiffuScene
#   mkdir -p logs
#   sbatch slurm_render_baseline.sh
cd "${SLURM_SUBMIT_DIR:-$(dirname "$(realpath "$0")")}"
mkdir -p logs

# Use the same renderer/camera/colors as slurm_render_resolved.sh.
# Exported scene_mesh/ objects already contain the original transforms.
# Supplying the merged JSON to BOTH inputs makes translation deltas zero;
# collision_params_resolved.json is never opened or required.
# Check inputs for all three rooms before starting rendering.
for model in bedrooms_bert livingrooms_bert diningrooms_bert; do
    output_dir="./models/pretrained_diffusion/$model/finalbase"
    if [[ ! -f "$output_dir/collision_params_merged.json" ]]; then
        echo "ERROR: Missing $output_dir/collision_params_merged.json" >&2
        exit 1
    fi
    if [[ ! -d "$output_dir/scene_mesh" ]]; then
        echo "ERROR: Missing $output_dir/scene_mesh" >&2
        exit 1
    fi
done

for model in bedrooms_bert livingrooms_bert diningrooms_bert; do
    echo "Rendering baseline $model..."
    xvfb-run -a python scripts/render_resolved.py \
        --output_dir "./models/pretrained_diffusion/$model/finalbase" \
        --orig_json collision_params_merged.json \
        --resolved_json collision_params_merged.json \
        --render_imgs_dir rendered_baseline
done

echo "Done! Baseline images saved in each model's finalbase/rendered_baseline/."
