# Embodied Vision Group Project

## Scope

This repository is the shared workspace for our embodied vision project.

It tracks the actual source code state we use for experiments, including modified code under `team9-model-code/external/`. Large runtime assets such as datasets, checkpoints, logs, and generated outputs are intentionally excluded from git.

## Current focus

- DINO-WM wall-task analysis and walkthroughs
- C-JEPA PushT training
- NYU HPC deployment and reproducibility

## Repository policy

- Keep code that affects experiments under version control, including modified external code.
- Do not commit datasets, pretrained weights, checkpoints, logs, or generated outputs.
- Use external storage such as `/scratch` on HPC for large runtime assets.
- If an external model is modified locally and required for reproduction, that modified code should live in this repository or in a clearly pinned fork.

Related docs:

- [CJEPA PushT Setup](/D:/Embodied Vision group proj/docs/cjepa_pusht_setup.md)
- [CJEPA PushT Setup ZH](/D:/Embodied Vision group proj/docs/cjepa_pusht_setup_zh.md)

## Project structure

- `docs/`
  - setup notes and reproducibility docs
- `notebooks/`
  - code walkthroughs
  - debugging notebooks
  - analysis notes
- `scripts/`
  - helper scripts for local runs and HPC runs
- `team9-model-code/external/`
  - vendored upstream code that we use or modify
  - keep code, exclude datasets and outputs
- `team9-model-code/src/`
  - our own project-specific model and experiment code

## Notes

- Local and HPC workflows should use the same tracked code state.
- If a result depends on a local modification to an external model, that modification should not live only on one machine.
