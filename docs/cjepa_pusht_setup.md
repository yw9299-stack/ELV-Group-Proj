# CJEPA PushT Setup on a New Computer

This document records the deployment path that was actually used to get the PushT CJEPA training pipeline running in WSL on a new Windows machine.

It is intentionally practical rather than minimal. The goal is to reach a state where the PushT slot-based training script starts real training, not just where imports succeed.

## Scope

This setup is for:

- Windows host
- WSL2 Ubuntu
- NVIDIA GPU available through WSL
- PyCharm on Windows using a WSL Conda interpreter
- `team9-model-code/external/cjepa-main`
- PushT slot-based training:
  - `src/train/train_causalwm_AP_node_pusht_slot.py`

This document does **not** cover full object-centric model training from scratch. For PushT CJEPA training, we use official pre-extracted slot embeddings and the official Videosaur checkpoint.

## 1. Required Components

You need all of the following:

- This workspace repository
- WSL Ubuntu
- Miniconda inside WSL
- A Conda environment for CJEPA
- A recent enough Windows NVIDIA driver so WSL CUDA works
- The DINO-WM PushT dataset at:
  - `team9-model-code/external/dino_wm/datasets/pusht_noise`
- Official precomputed PushT slot embeddings
- Official PushT Videosaur checkpoint
- Generated PushT action/proprio/state metadata pickles

## 2. Directory Layout

Expected relevant layout:

```text
Embodied Vision group proj/
├── docs/
│   └── cjepa_pusht_setup.md
├── scripts/
│   ├── prepare_cjepa_pusht_meta.py
│   ├── extract_pusht_slots_videosaur.py
│   └── run_cjepa_pusht_1epoch.sh
└── team9-model-code/
    └── external/
        ├── dino_wm/
        │   └── datasets/
        │       └── pusht_noise/
        └── cjepa-main/
            └── data/
                └── pusht_precomputed/
```

Final precomputed directory:

```text
team9-model-code/external/cjepa-main/data/pusht_precomputed/
├── pusht_expert_action_meta.pkl
├── pusht_expert_proprio_meta.pkl
├── pusht_expert_state_meta.pkl
├── pusht_slots.pkl
└── pusht_videosaur_model.ckpt
```

## 3. Windows and WSL Prerequisites

### 3.1 Install or update WSL

Use WSL2 with Ubuntu.

### 3.2 Install Miniconda inside WSL

Create and use a dedicated Conda environment for CJEPA.

### 3.3 Update the Windows NVIDIA driver

This matters even if training is launched from WSL. WSL uses the Windows host driver.

For this setup, using the latest NVIDIA **Studio Driver** is recommended over Game Ready, because this machine is being used for CUDA / PyTorch / reproducible research rather than games.

After updating, reboot Windows and verify:

On Windows:

```cmd
nvidia-smi
```

Inside WSL:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.version.cuda)"
```

If WSL cannot use CUDA, training may still start but later fail when tensors move to GPU.

## 4. Clone and Open the Workspace

Clone the workspace on Windows, for example:

```text
D:\Embodied Vision group proj
```

From WSL this will appear as:

```text
/mnt/d/Embodied Vision group proj
```

## 5. Prepare the CJEPA Environment

Activate the environment you want PyCharm and WSL to use, for example:

```bash
conda activate cjepa
```

At minimum, the environment must support:

- `torch`
- `torchvision`
- `lightning`
- `transformers`
- `timm`
- `stable_pretraining`
- `stable_worldmodel`

This repository also contains vendored third-party code under:

- `src/third_party/videosaur`
- `src/third_party/stable-worldmodel`
- `src/third_party/stable-pretraining`

## 6. Put the PushT Dataset in Place

For this workflow, the dataset is expected at:

```text
team9-model-code/external/dino_wm/datasets/pusht_noise
```

That directory must contain:

```text
pusht_noise/
├── train/
│   ├── states.pth
│   ├── rel_actions.pth
│   ├── velocities.pth
│   ├── seq_lengths.pkl
│   └── obses/
└── val/
    ├── states.pth
    ├── rel_actions.pth
    ├── velocities.pth
    ├── seq_lengths.pkl
    └── obses/
```

This setup assumes that dataset is already present locally.

## 7. Generate Metadata Pickles

We generate CJEPA-specific metadata from `pusht_noise` using:

- [prepare_cjepa_pusht_meta.py](/mnt/d/Embodied%20Vision%20group%20proj/scripts/prepare_cjepa_pusht_meta.py)

Run:

```bash
cd "/mnt/d/Embodied Vision group proj"
python scripts/prepare_cjepa_pusht_meta.py
```

This writes:

- `pusht_expert_action_meta.pkl`
- `pusht_expert_proprio_meta.pkl`
- `pusht_expert_state_meta.pkl`

to:

```text
/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed
```

## 8. Download Official PushT Slots

Do **not** recompute slots unless you specifically want to reproduce the object-centric extraction path.

For PushT CJEPA training, the fastest and most stable route is to use the official pre-extracted slot embeddings:

- `pusht_videosaur_slots.pkl`

Download and save it as:

```bash
mkdir -p "/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed" && \
wget -O "/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed/pusht_slots.pkl" \
"https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_videosaur_slots.pkl"
```

## 9. Download Official Videosaur Checkpoint

The PushT slot-based training path also needs the object-centric checkpoint because the model builder still loads the Videosaur model structure and weights.

Download:

- `pusht_videosaur_model.ckpt`

Save it as:

```bash
wget -O "/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed/pusht_videosaur_model.ckpt" \
"https://huggingface.co/HazelNam/CJEPA/resolve/main/pusht_videosaur_model.ckpt"
```

## 10. Verify Precomputed Assets

Run:

```bash
ls -lh "/mnt/d/Embodied Vision group proj/team9-model-code/external/cjepa-main/data/pusht_precomputed"
```

You should see:

- `pusht_expert_action_meta.pkl`
- `pusht_expert_proprio_meta.pkl`
- `pusht_expert_state_meta.pkl`
- `pusht_slots.pkl`
- `pusht_videosaur_model.ckpt`

## 11. Start Training

Use the WSL-friendly wrapper:

- [run_cjepa_pusht_1epoch.sh](/mnt/d/Embodied%20Vision%20group%20proj/scripts/run_cjepa_pusht_1epoch.sh)

Run:

```bash
bash "/mnt/d/Embodied Vision group proj/scripts/run_cjepa_pusht_1epoch.sh"
```

This script:

- checks for metadata pickles
- generates them if needed
- checks for `pusht_slots.pkl`
- checks for `pusht_videosaur_model.ckpt`
- launches:
  - `src/train/train_causalwm_AP_node_pusht_slot.py`

## 12. PyCharm Configuration

Recommended setup:

- Open the Windows-side project in PyCharm
- Use a **WSL interpreter**
- Point the interpreter to the WSL Conda environment, for example:
  - `/home/<user>/miniconda3/envs/cjepa/bin/python`

For one-click execution:

1. Create a Shell Script / Bash run configuration
2. Script path:
   - `/mnt/d/Embodied Vision group proj/scripts/run_cjepa_pusht_1epoch.sh`
3. Interpreter:
   - `/bin/bash`
4. Working directory:
   - `/mnt/d/Embodied Vision group proj`

## 13. Why So Many Steps Were Needed

This pipeline was not plug-and-play on a fresh machine because several issues appeared in sequence:

- the metadata needed by PushT slot training was not already generated from `pusht_noise`
- the slot training script needs official precomputed `slots.pkl`
- the Videosaur model builder still expects an object-centric checkpoint
- `torchcodec` in the active environment was installed but unusable
- the installed `stable_worldmodel` API no longer exposed `swm.wm.dinowm.Embedder`
- CUDA driver compatibility initially blocked loading GPU-saved checkpoints

The repository was adjusted locally to make the stack workable in this environment.

## 14. Local Code Adjustments That Were Needed

The following local changes were required to make the pipeline practical on this machine:

- `scripts/prepare_cjepa_pusht_meta.py`
  - added to generate PushT metadata pickles from `pusht_noise`
- `scripts/extract_pusht_slots_videosaur.py`
  - added for optional slot extraction
  - supports fallback video backends
- `scripts/run_cjepa_pusht_1epoch.sh`
  - added as the WSL/PyCharm one-click launch wrapper
- `src/third_party/videosaur/videosaur/data/pipelines.py`
  - made tolerant to broken `torchcodec` imports
- `src/third_party/videosaur/videosaur/models.py`
  - changed checkpoint loading to `map_location="cpu"`
- `src/train/train_causalwm_AP_node_pusht_slot.py`
  - switched from `swm.wm.dinowm.Embedder` to the repository-local `Embedder`

These changes are deployment-oriented compatibility fixes. They do not change the overall PushT slot-training objective.

## 15. Optional: Recompute Slots Yourself

This is optional and not needed for the normal deployment path.

If you later want to reproduce the slot extraction path yourself, use:

- [extract_pusht_slots_videosaur.py](/mnt/d/Embodied%20Vision%20group%20proj/scripts/extract_pusht_slots_videosaur.py)

However, this path is more fragile because it depends on:

- video decoding backend availability
- object-centric checkpoint availability
- compatible CUDA / PyTorch / TorchCodec stack

For deployment on a new machine, using the official pre-extracted PushT slots is strongly preferred.

## 16. Sanity Check for Success

The setup should be considered successful when the training run shows all of the following:

- metadata pickles load successfully
- slot embeddings load successfully
- Videosaur weights load successfully
- Lightning enters `Epoch 0/0`
- the step counter advances beyond `0/...`

Example success signal:

```text
Epoch 0/0 ... 865/119186
```

At that point, the training stack is genuinely running.
