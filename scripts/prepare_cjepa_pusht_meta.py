import argparse
import pickle
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PUSHT_ROOT = REPO_ROOT / "team9-model-code" / "external" / "dino_wm" / "datasets" / "pusht_noise"
DEFAULT_OUT_DIR = REPO_ROOT / "team9-model-code" / "external" / "cjepa-main" / "data" / "pusht_precomputed"


def build_split(split_root: Path, action_scale: float) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    states = torch.load(split_root / "states.pth", map_location="cpu").float()
    rel_actions = torch.load(split_root / "rel_actions.pth", map_location="cpu").float()
    velocities = torch.load(split_root / "velocities.pth", map_location="cpu").float()

    with open(split_root / "seq_lengths.pkl", "rb") as f:
        seq_lengths = pickle.load(f)

    action_dict: dict[str, np.ndarray] = {}
    proprio_dict: dict[str, np.ndarray] = {}
    state_dict: dict[str, np.ndarray] = {}

    for episode_idx, seq_len in enumerate(seq_lengths):
        seq_len = int(seq_len)
        key = f"{episode_idx}_pixels.mp4"

        state = states[episode_idx, :seq_len].numpy()
        velocity = velocities[episode_idx, :seq_len].numpy()
        action = (rel_actions[episode_idx, :seq_len] / action_scale).numpy()
        proprio = np.concatenate([state[:, :2], velocity], axis=-1)
        full_state = np.concatenate([state, velocity], axis=-1)

        action_dict[key] = action.astype(np.float32, copy=False)
        proprio_dict[key] = proprio.astype(np.float32, copy=False)
        state_dict[key] = full_state.astype(np.float32, copy=False)

    return action_dict, proprio_dict, state_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PushT action/proprio/state meta pickles for CJEPA from dino_wm pusht_noise.")
    parser.add_argument(
        "--pusht-root",
        type=Path,
        default=DEFAULT_PUSHT_ROOT,
        help="Path to the pusht_noise dataset root containing train/ and val/.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory where the meta pickle files will be written.",
    )
    parser.add_argument(
        "--action-scale",
        type=float,
        default=100.0,
        help="Scale factor used to convert rel_actions.pth back to environment action units.",
    )
    args = parser.parse_args()

    pusht_root = args.pusht_root.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    required = [
        pusht_root / "train" / "states.pth",
        pusht_root / "train" / "rel_actions.pth",
        pusht_root / "train" / "velocities.pth",
        pusht_root / "train" / "seq_lengths.pkl",
        pusht_root / "val" / "states.pth",
        pusht_root / "val" / "rel_actions.pth",
        pusht_root / "val" / "velocities.pth",
        pusht_root / "val" / "seq_lengths.pkl",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required PushT files:\n" + "\n".join(missing))

    train_action, train_proprio, train_state = build_split(pusht_root / "train", args.action_scale)
    val_action, val_proprio, val_state = build_split(pusht_root / "val", args.action_scale)

    action_file = {"train": train_action, "val": val_action}
    proprio_file = {"train": train_proprio, "val": val_proprio}
    state_file = {"train": train_state, "val": val_state}

    action_path = out_dir / "pusht_expert_action_meta.pkl"
    proprio_path = out_dir / "pusht_expert_proprio_meta.pkl"
    state_path = out_dir / "pusht_expert_state_meta.pkl"

    with open(action_path, "wb") as f:
        pickle.dump(action_file, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(proprio_path, "wb") as f:
        pickle.dump(proprio_file, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(state_path, "wb") as f:
        pickle.dump(state_file, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {action_path}")
    print(f"Wrote {proprio_path}")
    print(f"Wrote {state_path}")
    print(f"train episodes: {len(train_action)}")
    print(f"val episodes: {len(val_action)}")


if __name__ == "__main__":
    main()
