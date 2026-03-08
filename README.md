# Embodied Vision Group Project

## Current focus

We are currently focusing only on the **wall** task from DINO-WM.

## Suggested setup steps

1. Clone this repository.
2. Set up a Linux environment. WSL Ubuntu is recommended on Windows.
3. Install Miniconda.
4. Clone the original DINO-WM repository into:
   `team9-model-code/external/dino_wm`
5. Enter `team9-model-code/external/dino_wm` and create the `dino_wm` conda environment.
6. Start from:
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

- WSL Ubuntu environment is set up
- The `dino_wm` conda environment is working
- `train.py --help` runs successfully in the DINO-WM environment
- PyCharm interpreter is connected to the WSL `dino_wm` environment
- The initial notebook scaffold has been created:
  - `notebooks/wall_code_walkthrough.ipynb`

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