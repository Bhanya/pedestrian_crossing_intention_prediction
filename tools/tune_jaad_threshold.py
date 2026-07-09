import argparse
import csv
import os
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.EfficientPIE_baseline import EfficientPIE
from models.EfficientPIE_bbox import EfficientPIEBBox
from models.EfficientPIE_bbox_trajectory import EfficientPIEBBoxTrajectory
from utils.jaad_data import JAAD
from utils.my_dataset import MyDataSet


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


def build_dataset(data_path, split, use_bbox_features, use_bbox_trajectory):
    data_type = {
        "encoder_input_type": DATA_OPTS["encoder_input_type"],
        "decoder_input_type": DATA_OPTS["decoder_input_type"],
        "output_type": DATA_OPTS["output_type"],
    }
    jaad_dataset = JAAD(data_path=data_path)
    seq = jaad_dataset.generate_data_trajectory_sequence(split, **DATA_OPTS)
    seq_for_dataset = jaad_dataset.get_train_val_data(
        seq,
        data_type,
        DATA_OPTS["max_size_observe"],
        DATA_OPTS["seq_overlap_rate"],
    )
    transform = transforms.Compose([
        transforms.Resize([300, 300]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return MyDataSet(
        images_seq=seq_for_dataset,
        data_opts=DATA_OPTS,
        transform=transform,
        return_bbox_features=use_bbox_features,
        return_bbox_trajectory=use_bbox_trajectory,
    )


def build_model(use_bbox_features, use_bbox_trajectory, device):
    if use_bbox_features and use_bbox_trajectory:
        raise ValueError("Choose only one: --use-bbox-features or --use-bbox-trajectory")
    if use_bbox_trajectory:
        return EfficientPIEBBoxTrajectory(num_classes=2).to(device)
    if use_bbox_features:
        return EfficientPIEBBox(num_classes=2).to(device)
    return EfficientPIE(num_classes=2).to(device)


@torch.no_grad()
def collect_predictions(model, dataloader, device):
    model.eval()
    probs = []
    labels_all = []
    for step, data in enumerate(dataloader, start=1):
        if len(data) == 3:
            images, bbox_context, labels = data
            logits = model(images.to(device), bbox_context.to(device))
        else:
            images, labels = data
            logits = model(images.to(device))
        batch_probs = torch.softmax(logits, dim=1)[:, 1]
        probs.extend(batch_probs.cpu())
        labels_all.extend(labels.cpu())
        if step == 1 or step % 50 == 0:
            print(f"Collected predictions for batch {step}/{len(dataloader)}")
    return torch.stack(probs), torch.stack(labels_all).long()


def compute_metrics(probs, labels, threshold):
    preds = (probs >= threshold).long()
    tp = ((preds == 1) & (labels == 1)).sum().item()
    tn = ((preds == 0) & (labels == 0)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()

    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--use-bbox-features", action="store_true")
    parser.add_argument("--use-bbox-trajectory", action="store_true")
    parser.add_argument("--start", type=float, default=0.05)
    parser.add_argument("--end", type=float, default=0.95)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--output", default="./logs/threshold_tuning.csv")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = build_dataset(
        args.data_path,
        args.split,
        args.use_bbox_features,
        args.use_bbox_trajectory,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=device.type == "cuda",
        num_workers=0 if device.type == "cpu" else min(os.cpu_count(), args.batch_size, 8),
        collate_fn=MyDataSet.collate_fn,
    )

    model = build_model(args.use_bbox_features, args.use_bbox_trajectory, device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    probs, labels = collect_predictions(model, dataloader, device)

    thresholds = []
    current = args.start
    while current <= args.end + 1e-9:
        thresholds.append(round(current, 6))
        current += args.step

    rows = [compute_metrics(probs, labels, threshold) for threshold in thresholds]
    best = max(rows, key=lambda row: (row["f1"], row["recall"], row["accuracy"]))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved threshold sweep to {args.output}")
    print(
        "Best threshold by validation F1: "
        f"{best['threshold']:.2f} | "
        f"acc {best['accuracy']:.4f}, precision {best['precision']:.4f}, "
        f"recall {best['recall']:.4f}, f1 {best['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
