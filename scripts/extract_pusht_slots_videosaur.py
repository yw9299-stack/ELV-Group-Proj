import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms

VIDEO_BACKEND = None
try:
    from torchcodec.decoders import VideoDecoder as TorchCodecVideoDecoder

    VIDEO_BACKEND = "torchcodec"
except Exception:
    TorchCodecVideoDecoder = None

try:
    import decord

    decord.bridge.set_bridge("torch")
    VIDEO_BACKEND = VIDEO_BACKEND or "decord"
except ImportError:
    decord = None

try:
    import cv2

    VIDEO_BACKEND = VIDEO_BACKEND or "opencv"
except ImportError:
    cv2 = None


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CJEPA_ROOT = REPO_ROOT / "team9-model-code" / "external" / "cjepa-main"
DEFAULT_PUSHT_ROOT = REPO_ROOT / "team9-model-code" / "external" / "dino_wm" / "datasets" / "pusht_noise"
DEFAULT_VIDEOSAUR_CONFIG = DEFAULT_CJEPA_ROOT / "src" / "third_party" / "videosaur" / "configs" / "videosaur" / "pusht_dinov2_hf.yml"
DEFAULT_SAVE_PATH = DEFAULT_CJEPA_ROOT / "data" / "pusht_precomputed" / "pusht_slots.pkl"


def sorted_episode_paths(obs_dir: Path) -> list[Path]:
    def key_fn(path: Path) -> int:
        return int(path.stem.split("_")[-1])

    return sorted(obs_dir.glob("episode_*.mp4"), key=key_fn)


def output_key(video_path: Path) -> str:
    episode_idx = int(video_path.stem.split("_")[-1])
    return f"{episode_idx}_pixels.mp4"


def read_video_frames(video_path: Path, device: str) -> torch.Tensor:
    if TorchCodecVideoDecoder is not None:
        decoder = TorchCodecVideoDecoder(str(video_path))
        return decoder[:].to(device)

    if decord is not None:
        reader = decord.VideoReader(str(video_path), num_threads=1)
        frames = reader.get_batch(range(len(reader))).to(device)
        return frames

    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        ok, frame = cap.read()
        while ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(torch.from_numpy(frame))
            ok, frame = cap.read()
        cap.release()
        if not frames:
            raise RuntimeError(f"Failed to decode any frames from {video_path}")
        return torch.stack(frames, dim=0).to(device)

    raise ImportError(
        "None of torchcodec, decord, or opencv-python is available. "
        "Install one of them in the active environment."
    )


@torch.no_grad()
def extract_video_slots(model, video_path: Path, device: str, resize_to: int) -> np.ndarray:
    frames = read_video_frames(video_path, device=device)
    tfm = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((resize_to, resize_to)),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    video = tfm(frames).unsqueeze(0)
    encoder_out = model.encoder(video)
    features = encoder_out["features"]
    slots_init = model.initializer(batch_size=1).to(device)
    processor_out = model.processor(slots_init, features)
    return processor_out["state"][0].detach().cpu().numpy()


def extract_split(model, split_root: Path, device: str, resize_to: int, limit: int | None) -> dict[str, np.ndarray]:
    obs_dir = split_root / "obses"
    if not obs_dir.exists():
        raise FileNotFoundError(f"Missing obs directory: {obs_dir}")

    videos = sorted_episode_paths(obs_dir)
    if limit is not None:
        videos = videos[:limit]

    split_slots: dict[str, np.ndarray] = {}
    for idx, video_path in enumerate(videos, start=1):
        print(f"[{split_root.name}] {idx}/{len(videos)} {video_path.name}")
        split_slots[output_key(video_path)] = extract_video_slots(model, video_path, device=device, resize_to=resize_to)
    return split_slots


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract PushT slot embeddings from dino_wm pusht_noise videos using a Videosaur checkpoint.")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=DEFAULT_CJEPA_ROOT,
        help="Path to the cjepa-main repository root.",
    )
    parser.add_argument(
        "--pusht-root",
        type=Path,
        default=DEFAULT_PUSHT_ROOT,
        help="Path to the pusht_noise dataset root containing train/ and val/.",
    )
    parser.add_argument(
        "--videosaur-config",
        type=Path,
        default=DEFAULT_VIDEOSAUR_CONFIG,
        help="Path to the Videosaur YAML config.",
    )
    parser.add_argument("--weight", type=Path, required=True, help="Path to the pretrained Videosaur/Object-Centric checkpoint.")
    parser.add_argument(
        "--save-path",
        type=Path,
        default=DEFAULT_SAVE_PATH,
        help="Output pickle path.",
    )
    parser.add_argument("--device", default="cuda", help="Torch device, e.g. cuda or cpu.")
    parser.add_argument("--resize-to", type=int, default=196, help="Input resolution for Videosaur.")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on videos per split for smoke testing.")
    args = parser.parse_args()

    repo_root = args.repo_root.expanduser().resolve()
    sys.path.insert(0, str(repo_root))

    from src.third_party.videosaur.videosaur import configuration, models

    conf = configuration.load_config(str(args.videosaur_config.expanduser().resolve()))
    model = models.build(conf.model, conf.optimizer)
    ckpt = torch.load(args.weight.expanduser().resolve(), map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict)
    model = model.eval().to(args.device)

    pusht_root = args.pusht_root.expanduser().resolve()
    save_path = args.save_path.expanduser().resolve()
    save_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Video backend: {VIDEO_BACKEND or 'unavailable'}")

    slots = {
        "train": extract_split(model, pusht_root / "train", device=args.device, resize_to=args.resize_to, limit=args.limit),
        "val": extract_split(model, pusht_root / "val", device=args.device, resize_to=args.resize_to, limit=args.limit),
    }

    with open(save_path, "wb") as f:
        pickle.dump(slots, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Wrote {save_path}")
    print(f"train videos: {len(slots['train'])}")
    print(f"val videos: {len(slots['val'])}")


if __name__ == "__main__":
    main()
