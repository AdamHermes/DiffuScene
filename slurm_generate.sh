#!/bin/bash
#SBATCH --job-name=diffuscene_gen
#SBATCH --output=logs/diffuscene_%A_%a.out
#SBATCH --error=logs/diffuscene_%A_%a.err
#SBATCH --array=0-5                 # Chạy 6 jobs (từ 0 đến 5)
#SBATCH --nodelist=xgpe2
#SBATCH --gres=gpu:1                # Xem giải thích về VRAM ở dưới
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                   # RAM hệ thống
#SBATCH --time=04:00:00

mkdir -p logs

EXP_DIR="../models"
EXP_NAME="pretrained_diffusion/bedrooms_uncond"
CONFIG="../config/uncond/diffusion_bedrooms_instancond_lat32_v.yaml"
# Đã sửa lại file pkl thành bedroom
THREED_FUTURE="../datasets/3d_front_processed/threed_future_model_bedroom.pkl"
WEIGHT_FILE="${EXP_DIR}/${EXP_NAME}/model_30000"
OUT_DIR="${EXP_DIR}/${EXP_NAME}/final"

# Cố định số scene mỗi job. Job cuối sẽ tự động ngắt khi hết data.
SEQ_PER_JOB=30
START_IDX=$((${SLURM_ARRAY_TASK_ID} * ${SEQ_PER_JOB}))

echo "Running task ${SLURM_ARRAY_TASK_ID} with start_index ${START_IDX} on node $(hostname)"

cd scripts

xvfb-run -a python -u generate_diffusion.py "${CONFIG}" "${OUT_DIR}" "${THREED_FUTURE}" \
    --weight_file "${WEIGHT_FILE}" \
    --n_sequences ${SEQ_PER_JOB} \
    --fix_order \
    --start_index ${START_IDX} \
    --render_top2down \
    --save_mesh \
    --no_texture \
    --without_floor \
    --clip_denoised \
    --retrive_objfeats \
    --guidance_config ../config/inference_guidance.yaml \
    --seed 111 \
