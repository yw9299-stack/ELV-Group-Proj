#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CJEPA_ROOT="${REPO_ROOT}/team9-model-code/external/cjepa-main"
PUSHT_ROOT="${CJEPA_PUSHT_ROOT:-${REPO_ROOT}/team9-model-code/external/dino_wm/datasets/pusht_noise}"
PREP_DIR="${CJEPA_PUSHT_PREP_DIR:-${CJEPA_ROOT}/data/pusht_precomputed}"
SLOT_PATH="${CJEPA_PUSHT_SLOTS:-${PREP_DIR}/pusht_slots.pkl}"
OC_CKPT_PATH="${CJEPA_OC_CKPT:-${PREP_DIR}/pusht_videosaur_model.ckpt}"
ACTION_META="${PREP_DIR}/pusht_expert_action_meta.pkl"
PROPRIO_META="${PREP_DIR}/pusht_expert_proprio_meta.pkl"
STATE_META="${PREP_DIR}/pusht_expert_state_meta.pkl"
OUTPUT_NAME="${CJEPA_OUTPUT_NAME:-pusht_cjepa_epoch1}"
CHECKPOINT_DIR="${PREP_DIR}/checkpoints"
PYTHON_EXE="${CJEPA_PYTHON:-python}"

mkdir -p "${PREP_DIR}" "${CHECKPOINT_DIR}"

if [[ ! -f "${ACTION_META}" || ! -f "${PROPRIO_META}" || ! -f "${STATE_META}" ]]; then
  echo "Meta files missing. Generating from ${PUSHT_ROOT} ..."
  "${PYTHON_EXE}" "${REPO_ROOT}/scripts/prepare_cjepa_pusht_meta.py" --pusht-root "${PUSHT_ROOT}" --out-dir "${PREP_DIR}"
fi

if [[ ! -f "${SLOT_PATH}" ]]; then
  cat >&2 <<EOF
Missing slot embeddings:
  ${SLOT_PATH}

Download official pre-extracted slots or generate them with:
  ${PYTHON_EXE} ${REPO_ROOT}/scripts/extract_pusht_slots_videosaur.py --weight <videosaur_ckpt> --pusht-root "${PUSHT_ROOT}" --save-path "${SLOT_PATH}"
EOF
  exit 1
fi

if [[ ! -f "${OC_CKPT_PATH}" ]]; then
  cat >&2 <<EOF
Missing object-centric checkpoint:
  ${OC_CKPT_PATH}

Download it to that path, or set CJEPA_OC_CKPT to the actual file location.
EOF
  exit 1
fi

cd "${CJEPA_ROOT}"
PYTHONPATH="${CJEPA_ROOT}" "${PYTHON_EXE}" src/train/train_causalwm_AP_node_pusht_slot.py \
  wandb.enable=false \
  output_model_name="${OUTPUT_NAME}" \
  checkpoint_dir="${CHECKPOINT_DIR}" \
  cache_dir="${PREP_DIR}" \
  embedding_dir="${SLOT_PATH}" \
  model.load_weights="${OC_CKPT_PATH}" \
  action_dir="${ACTION_META}" \
  proprio_dir="${PROPRIO_META}" \
  state_dir="${STATE_META}" \
  trainer.max_epochs=1 \
  trainer.accelerator=auto \
  trainer.devices=1 \
  num_workers=1 \
  batch_size=16 \
  num_masked_slots=1 \
  use_hungarian_matching=false
