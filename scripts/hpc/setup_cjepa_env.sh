#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash scripts/hpc/setup_cjepa_env.sh /scratch/$USER/cjepa-env
#
# The argument is optional. If omitted, the conda env will be created at:
#   /scratch/$USER/cjepa-env

ENV_PREFIX="${1:-/scratch/$USER/cjepa-env}"
REPO_DIR="${REPO_DIR:-$HOME/EmbodiedVision/team9-model-code/external/cjepa-main}"

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "REPO_DIR does not exist: ${REPO_DIR}" >&2
  exit 1
fi

module purge || true
module load miniconda3 || true

source "$(conda info --base)/etc/profile.d/conda.sh"

conda create -y -p "${ENV_PREFIX}" python=3.10
conda activate "${ENV_PREFIX}"

conda install -y -c anaconda ffmpeg
python -m pip install --upgrade pip setuptools wheel uv

cd "${REPO_DIR}"

uv pip install seaborn webdataset swig einops torchcodec av accelerate tensorboard tensorboardX hickle pycocotools wget gdown

mkdir -p src/third_party
cd src/third_party

if [[ ! -d stable-pretraining ]]; then
  git clone https://github.com/galilai-group/stable-pretraining.git
fi
cd stable-pretraining
git checkout 92b5841
uv pip install -e .

cd ..
if [[ ! -d stable-worldmodel ]]; then
  git clone https://github.com/galilai-group/stable-worldmodel.git
fi
cd stable-worldmodel
git checkout 221ac82
uv pip install -e .

cd ..
if [[ ! -d nerv ]]; then
  git clone https://github.com/Wuziyi616/nerv.git
fi
cd nerv
git checkout v0.1.0
uv pip install -e .

cd "${REPO_DIR}"
python -c "import lightning, stable_pretraining, stable_worldmodel; print('Environment OK')"

echo "Created env at ${ENV_PREFIX}"
