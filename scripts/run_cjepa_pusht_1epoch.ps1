$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$CJEPARoot = Join-Path $RepoRoot "team9-model-code\external\cjepa-main"
$PushTRoot = if ($env:CJEPA_PUSHT_ROOT) { $env:CJEPA_PUSHT_ROOT } else { Join-Path $RepoRoot "team9-model-code\external\dino_wm\datasets\pusht_noise" }
$PrepDir = if ($env:CJEPA_PUSHT_PREP_DIR) { $env:CJEPA_PUSHT_PREP_DIR } else { Join-Path $CJEPARoot "data\pusht_precomputed" }
$SlotPath = if ($env:CJEPA_PUSHT_SLOTS) { $env:CJEPA_PUSHT_SLOTS } else { Join-Path $PrepDir "pusht_slots.pkl" }
$ActionMeta = Join-Path $PrepDir "pusht_expert_action_meta.pkl"
$ProprioMeta = Join-Path $PrepDir "pusht_expert_proprio_meta.pkl"
$StateMeta = Join-Path $PrepDir "pusht_expert_state_meta.pkl"
$OutputName = if ($env:CJEPA_OUTPUT_NAME) { $env:CJEPA_OUTPUT_NAME } else { "pusht_cjepa_epoch1" }
$CheckpointDir = Join-Path $PrepDir "checkpoints"
$PythonExe = if ($env:CJEPA_PYTHON) { $env:CJEPA_PYTHON } else { "python" }

New-Item -ItemType Directory -Force -Path $PrepDir | Out-Null
New-Item -ItemType Directory -Force -Path $CheckpointDir | Out-Null

if (-not (Test-Path $ActionMeta) -or -not (Test-Path $ProprioMeta) -or -not (Test-Path $StateMeta)) {
    Write-Host "Meta files missing. Generating from $PushTRoot ..."
    & $PythonExe (Join-Path $RepoRoot "scripts\prepare_cjepa_pusht_meta.py") --pusht-root $PushTRoot --out-dir $PrepDir
}

if (-not (Test-Path $SlotPath)) {
    Write-Error @"
Missing slot embeddings:
  $SlotPath

Generate them with:
  $PythonExe scripts\extract_pusht_slots_videosaur.py --weight <videosaur_ckpt> --pusht-root `"$PushTRoot`" --save-path `"$SlotPath`"
"@
}

Push-Location $CJEPARoot
try {
    $env:PYTHONPATH = $CJEPARoot
    & $PythonExe src/train/train_causalwm_AP_node_pusht_slot.py `
        wandb.enable=false `
        output_model_name=$OutputName `
        checkpoint_dir=$CheckpointDir `
        cache_dir=$PrepDir `
        embedding_dir=$SlotPath `
        action_dir=$ActionMeta `
        proprio_dir=$ProprioMeta `
        state_dir=$StateMeta `
        trainer.max_epochs=1 `
        trainer.accelerator=auto `
        trainer.devices=1 `
        num_workers=1 `
        batch_size=16 `
        num_masked_slots=1 `
        use_hungarian_matching=false
}
finally {
    Pop-Location
}
