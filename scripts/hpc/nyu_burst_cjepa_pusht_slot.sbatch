#!/usr/bin/env bash
# Submit with:
#   sbatch scripts/hpc/nyu_burst_cjepa_pusht_slot.sbatch
#
# Before submitting, update:
#   ACCOUNT
#   REPO_DIR
#   ENV_PREFIX
#   SLOT_PATH
#   OC_CKPT_PATH
#   ACTION_META
#   PROPRIO_META
#   STATE_META

#SBATCH --job-name=cjepa-pusht
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

ACCOUNT="csci_ga_3033-2025sp"
REPO_DIR="$HOME/EmbodiedVision/team9-model-code/external/cjepa-main"
ENV_PREFIX="/scratch/$USER/cjepa-env"
CACHE_DIR="/scratch/$USER/.stable_worldmodel"
RUN_NAME="pusht_cjepa_nyu"
CKPT_DIR="/scratch/$USER/checkpoints/cjepa"
CKPT_PATH="${CKPT_DIR}/${RUN_NAME}_weights.ckpt"

SLOT_PATH="/scratch/$USER/cjepa-data/pusht_videosaur_slots.pkl"
OC_CKPT_PATH="/scratch/$USER/cjepa-data/pusht_videosaur_model.ckpt"
ACTION_META="/scratch/$USER/cjepa-data/pusht_expert_action_meta.pkl"
PROPRIO_META="/scratch/$USER/cjepa-data/pusht_expert_proprio_meta.pkl"
STATE_META="/scratch/$USER/cjepa-data/pusht_expert_state_meta.pkl"

module purge || true
module load miniconda3 || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${ENV_PREFIX}"

cd "${REPO_DIR}"
export PYTHONPATH="${REPO_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"

mkdir -p "${CACHE_DIR}"
mkdir -p "${CKPT_DIR}"

echo "Checkpoint path: ${CKPT_PATH}"
if [[ -f "${CKPT_PATH}" ]]; then
  echo "Found existing checkpoint. Training will resume from it."
else
  echo "No existing checkpoint found. Training will start fresh."
fi

python src/train/train_causalwm_AP_node_pusht_slot.py \
  cache_dir="${CACHE_DIR}" \
  checkpoint_dir="${CKPT_DIR}" \
  resume_ckpt_path="${CKPT_PATH}" \
  output_model_name="${RUN_NAME}" \
  dataset_name="pusht_expert" \
  num_workers=8 \
  batch_size=256 \
  trainer.max_epochs=30 \
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
