import argparse
import sys
import types
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


DATA_OPTS = {
    "fstride": 1,
    "sample_type": "all",
    "height_rng": [0, float("inf")],
    "squarify_ratio": 0,
    "data_split_type": "random",
    "seq_type": "intention",
    "min_track_size": 0,
    "max_size_observe": 15,
    "seq_overlap_rate": 0.5,
    "balance": True,
    "crop_type": "context",
    "crop_mode": "pad_resize",
    "encoder_input_type": [],
    "decoder_input_type": ["bbox"],
    "output_type": ["intent"],
}


def load_jaad(allow_cv2_stub=False):
    try:
        from utils.jaad_data import JAAD
        return JAAD
    except ModuleNotFoundError as exc:
        if exc.name != "cv2" or not allow_cv2_stub:
            raise
        sys.modules["cv2"] = types.ModuleType("cv2")
        from utils.jaad_data import JAAD
        return JAAD


def collect_required_frames(data_path, splits, reverse_step):
    JAAD = load_jaad(allow_cv2_stub=True)
    dataset = JAAD(data_path=str(data_path))
    data_type = {
        "encoder_input_type": DATA_OPTS["encoder_input_type"],
        "decoder_input_type": DATA_OPTS["decoder_input_type"],
        "output_type": DATA_OPTS["output_type"],
    }

    required = defaultdict(set)
    idx = DATA_OPTS["max_size_observe"] - reverse_step
    if idx < 0 or idx >= DATA_OPTS["max_size_observe"]:
        raise ValueError(
            f"reverse_step={reverse_step} selects index {idx}, outside the "
            f"0..{DATA_OPTS['max_size_observe'] - 1} observation range"
        )

    for split in splits:
        seq = dataset.generate_data_trajectory_sequence(split, **DATA_OPTS)
        samples = dataset.get_train_val_data(
            seq,
            data_type,
            DATA_OPTS["max_size_observe"],
            DATA_OPTS["seq_overlap_rate"],
        )

        for sample in samples["images"]:
            frame_path = Path(sample[idx])
            vid = frame_path.parts[-2]
            frame_num = int(frame_path.stem)
            required[vid].add(frame_num)

    return required


def extract_frames(data_path, required, dry_run):
    if not dry_run:
        try:
            import cv2
        except ImportError as exc:
            raise SystemExit(
                "OpenCV is required to extract frames. Install it with "
                "`python3 -m pip install opencv-python` and rerun this command."
            ) from exc

    total = sum(len(frames) for frames in required.values())
    existing = 0
    written = 0
    missing_videos = []

    print(f"Required frames: {total}")
    for vid, frames in sorted(required.items()):
        out_dir = data_path / "images" / vid
        existing += sum((out_dir / f"{frame:05d}.png").is_file() for frame in frames)

    print(f"Already extracted: {existing}")
    print(f"Need to extract: {total - existing}")

    if dry_run:
        return

    for vid, frames in sorted(required.items()):
        video_path = data_path / "JAAD_clips" / f"{vid}.mp4"
        out_dir = data_path / "images" / vid
        out_dir.mkdir(parents=True, exist_ok=True)

        if not video_path.is_file():
            missing_videos.append(str(video_path))
            continue

        frames_to_write = {
            frame for frame in frames if not (out_dir / f"{frame:05d}.png").is_file()
        }
        if not frames_to_write:
            continue

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            missing_videos.append(str(video_path))
            continue

        frame_num = 0
        ok, image = cap.read()
        while ok and frames_to_write:
            if frame_num in frames_to_write:
                cv2.imwrite(str(out_dir / f"{frame_num:05d}.png"), image)
                frames_to_write.remove(frame_num)
                written += 1
                if written % 500 == 0:
                    print(f"Written {written} frames...")
            ok, image = cap.read()
            frame_num += 1
        cap.release()

        if frames_to_write:
            print(f"Warning: {video_path} missing {len(frames_to_write)} requested frames")

    print(f"New frames written: {written}")
    if missing_videos:
        print("Missing/unreadable videos:")
        for path in missing_videos:
            print(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--reverse-step", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    data_path = Path(args.data_path).resolve()
    required = collect_required_frames(data_path, args.splits, args.reverse_step)
    extract_frames(data_path, required, args.dry_run)


if __name__ == "__main__":
    main()
