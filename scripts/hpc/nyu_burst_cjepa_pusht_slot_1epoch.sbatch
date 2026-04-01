#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/hpc/nyu_burst_cjepa_pusht_slot_1epoch.sbatch
#
# Before submitting, update as needed:
#   REPO_DIR
#   ENV_PREFIX
#   SLOT_PATH
#   OC_CKPT_PATH
#   ACTION_META
#   PROPRIO_META
#   STATE_META
#
# Optional overrides at submit time:
#   sbatch --export=ALL,RUN_NAME=pusht_cjepa_epoch1,BATCH_SIZE=128 \
#     scripts/hpc/nyu_burst_cjepa_pusht_slot_1epoch.sbatch

#SBATCH --job-name=cjepa-1epoch
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --account=csci_ga_3033-2025sp
#SBATCH --partition=c12m85-a100-1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=04:00:00
#SBATCH --requeue

set -euo pipefail

ACCOUNT="${ACCOUNT:-csci_ga_3033-2025sp}"
REPO_DIR="${REPO_DIR:-$HOME/EmbodiedVision/team9-model-code/external/cjepa-main}"
ENV_PREFIX="${ENV_PREFIX:-/scratch/$USER/cjepa-env}"
CACHE_DIR="${CACHE_DIR:-/scratch/$USER/.stable_worldmodel}"
RUN_NAME="${RUN_NAME:-pusht_cjepa_1epoch_nyu}"
CKPT_DIR="${CKPT_DIR:-/scratch/$USER/checkpoints/cjepa-1epoch}"
CKPT_PATH="${CKPT_PATH:-${CKPT_DIR}/${RUN_NAME}_weights.ckpt}"

SLOT_PATH="${SLOT_PATH:-/scratch/$USER/cjepa-data/pusht_videosaur_slots.pkl}"
OC_CKPT_PATH="${OC_CKPT_PATH:-/scratch/$USER/cjepa-data/pusht_videosaur_model.ckpt}"
ACTION_META="${ACTION_META:-/scratch/$USER/cjepa-data/pusht_expert_action_meta.pkl}"
PROPRIO_META="${PROPRIO_META:-/scratch/$USER/cjepa-data/pusht_expert_proprio_meta.pkl}"
STATE_META="${STATE_META:-/scratch/$USER/cjepa-data/pusht_expert_state_meta.pkl}"
WANDB_MODE="${WANDB_MODE:-offline}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-256}"

module purge || true
module load miniconda3 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"
export WANDB_MODE

mkdir -p "${CACHE_DIR}"
mkdir -p "${CKPT_DIR}"

echo "Running 1-epoch CJEPA job at $(date)"
echo "Repo: ${REPO_DIR}"
echo "Env: ${ENV_PREFIX}"
echo "Checkpoint path: ${CKPT_PATH}"
nvidia-smi || true

python src/train/train_causalwm_AP_node_pusht_slot.py \
  wandb.enable=false \
  cache_dir="${CACHE_DIR}" \
  checkpoint_dir="${CKPT_DIR}" \
  resume_ckpt_path="${CKPT_PATH}" \
  output_model_name="${RUN_NAME}" \
  dataset_name="pusht_expert" \
  num_workers="${NUM_WORKERS}" \
  batch_size="${BATCH_SIZE}" \
  trainer.max_epochs=1 \
  trainer.devices=1 \
  num_masked_slots=1 \
  predictor_lr=5e-4 \
  proprio_encoder_lr=1e-4 \
  action_encoder_lr=5e-4 \
  dinowm.history_size=3 \
  dinowm.num_preds=1 \
  dinowm.proprio_embed_dim=128 \
  dinowm.action_embed_dim=128 \
  frameskip=5 \
  videosaur.NUM_SLOTS=4 \
  videosaur.SLOT_DIM=128 \
  predictor.heads=16 \
  embedding_dir="${SLOT_PATH}" \
  model.load_weights="${OC_CKPT_PATH}" \
  action_dir="${ACTION_META}" \
  proprio_dir="${PROPRIO_META}" \
  state_dir="${STATE_META}" \
  use_hungarian_matching=false
