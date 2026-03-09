# Embodied Vision Group Project

## Current focus

We are currently focusing only on the **wall** task from DINO-WM.

## Suggested setup steps

1. Clone this repository.
2. Set up a Linux environment. WSL Ubuntu is recommended on Windows.
3. Install Miniconda.
4. Clone the original DINO-WM repository into:
   `team9-model-code/external/dino_wm`
5. Enter `team9-model-code/external/dino_wm` and create or update the `dino_wm` conda environment using the repository's updated `environment.yaml`.
6. Download and unpack the DINO-WM datasets.
7. Set `DATASET_DIR` to the parent directory that contains `wall_single/`.
8. Ensure MuJoCo 2.1 is installed under `~/.mujoco/mujoco210`.
9. Export `LD_LIBRARY_PATH` so `mujoco_py` can find MuJoCo before starting Jupyter.
10. Launch Jupyter from the same shell where the environment variables were exported.
11. Start from:
   `notebooks/wall_code_walkthrough.ipynb`

## Project structure

This repository is our main project workspace.

- `notebooks/`
  - code walkthroughs
  - debugging notebooks
  - analysis notes
- `scripts/`
  - helper scripts for running experiments or sanity checks
- `team9-model-code/external/`
  - external reference code
  - the original DINO-WM repository should be placed here
- `team9-model-code/src/`
  - our own project-specific model and experiment code
- `README.md`
  - workspace overview and collaboration notes

## Current status

- WSL Ubuntu environment is set up.
- The `dino_wm` conda environment has been created or updated using the repository's `environment.yaml`.
- `train.py --help` runs successfully in the DINO-WM environment.
- PyCharm interpreter is connected to the WSL `dino_wm` environment.
- The Wall walkthrough notebook has been validated through:
  - Hydra config composition for `env=wall`
  - Wall dataset loading and trajectory slicing
  - model instantiation and single forward pass
  - tensor shape tracing through the encoder, predictor, and decoder
  - Wall environment preparation and short rollout
- Running the notebook requires `DATASET_DIR` and MuJoCo-related `LD_LIBRARY_PATH` to be set before launching Jupyter.
- The initial notebook scaffold is available at:
  - `notebooks/wall_code_walkthrough.ipynb``

## External dependency

The original DINO-WM repository should be cloned separately into:

`team9-model-code/external/dino_wm`

This repository is intended to store our own workspace materials, not to mirror the full upstream DINO-WM repository.

## Collaboration note

The intended division is:

- use `team9-model-code/external/` for upstream reference code
- use `team9-model-code/src/` for our own extensions and experimental code
- use `notebooks/` for structured code reading and experiment notes
- use `scripts/` for reproducible entry points when needed