import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate segmentation masks from files and report mIoU."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory with predicted masks (.png).",
    )
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Directory with ground-truth masks (.png).",
    )
    parser.add_argument(
        "--split_file",
        type=str,
        default=None,
        help="Optional txt file (image_path label_path) to define evaluation subset.",
    )
    parser.add_argument(
        "--num_classes",
        type=int,
        default=3,
        help="Number of classes.",
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=255,
        help="Ignore label index in ground-truth.",
    )
    return parser.parse_args()


def read_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask.astype(np.int64)


def fast_hist(gt: np.ndarray, pred: np.ndarray, num_classes: int, ignore_index: int):
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    gt = gt[valid]
    pred = pred[valid]
    if gt.size == 0:
        return np.zeros((num_classes, num_classes), dtype=np.int64), 0
    invalid_pred = (pred < 0) | (pred >= num_classes)
    if np.any(invalid_pred):
        invalid_values = np.unique(pred[invalid_pred])
        raise ValueError(
            f"Pred mask has invalid class ids: {invalid_values.tolist()} (valid range: 0..{num_classes - 1})"
        )
    hist = np.bincount(num_classes * gt + pred, minlength=num_classes**2)
    return hist.reshape(num_classes, num_classes), valid.sum()


def collect_gt_files(gt_dir: Path, split_file: Path | None):
    if split_file is None:
        return sorted(gt_dir.rglob("*.png"))

    gt_files = []
    for line in split_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        rel_gt = parts[1].replace("\\", "/")
        gt_files.append(gt_dir.parent / rel_gt)
    return gt_files


def build_pred_index(pred_dir: Path):
    pred_map = {}
    for p in pred_dir.rglob("*.png"):
        if p.name in pred_map:
            raise ValueError(
                f"Duplicate prediction filename found: {p.name}\n"
                f"  First: {pred_map[p.name]}\n"
                f"  Second: {p}\n"
                "Use unique filenames or evaluate one folder at a time."
            )
        pred_map[p.name] = p
    return pred_map


def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    gt_dir = Path(args.gt_dir)
    split_file = Path(args.split_file) if args.split_file else None

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not gt_dir.exists():
        raise FileNotFoundError(f"Ground-truth directory not found: {gt_dir}")
    if split_file and not split_file.exists():
        raise FileNotFoundError(f"Split file not found: {split_file}")

    gt_files = collect_gt_files(gt_dir, split_file)
    if not gt_files:
        raise RuntimeError("No ground-truth files found for evaluation.")

    pred_index = build_pred_index(pred_dir)

    confusion = np.zeros((args.num_classes, args.num_classes), dtype=np.int64)
    total_valid_pixels = 0
    evaluated = 0
    missing = 0

    for gt_file in gt_files:
        if not gt_file.exists():
            missing += 1
            print(f"SKIP GT missing: {gt_file}")
            continue

        pred_file = pred_index.get(gt_file.name)
        if pred_file is None:
            missing += 1
            print(f"SKIP PRED missing: {gt_file.name}")
            continue

        gt = read_mask(gt_file)
        pred = read_mask(pred_file)

        if gt.shape != pred.shape:
            print(f"SKIP SHAPE mismatch: {gt_file.name} gt={gt.shape} pred={pred.shape}")
            missing += 1
            continue

        hist, valid_count = fast_hist(gt, pred, args.num_classes, args.ignore_index)
        confusion += hist
        total_valid_pixels += int(valid_count)
        evaluated += 1

    if evaluated == 0:
        raise RuntimeError("No files were evaluated. Check paths and filenames.")

    tp = np.diag(confusion).astype(np.float64)
    gt_sum = confusion.sum(axis=1).astype(np.float64)
    pred_sum = confusion.sum(axis=0).astype(np.float64)

    iou = tp / np.maximum(gt_sum + pred_sum - tp, 1.0)
    class_acc = tp / np.maximum(gt_sum, 1.0)
    miou = float(np.nanmean(iou))
    macc = float(np.nanmean(class_acc))
    pixacc = float(tp.sum() / np.maximum(confusion.sum(), 1.0))

    print("\n=== Evaluation Summary ===")
    print(f"Evaluated files: {evaluated}")
    print(f"Missing/Skipped: {missing}")
    print(f"Valid pixels: {total_valid_pixels:,}")
    print(f"Pixel Acc: {pixacc:.6f}")
    print(f"Mean Acc: {macc:.6f}")
    print(f"mIoU: {miou:.6f}")

    for c in range(args.num_classes):
        print(f"Class {c} IoU: {iou[c]:.6f} | Acc: {class_acc[c]:.6f}")


if __name__ == "__main__":
    main()
