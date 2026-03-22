import argparse
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upscale predictions from 640x640 to original image size."
    )
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory with predictions (e.g., pseudo_color_prediction/).",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=True,
        help="Directory with original images.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save upscaled predictions.",
    )
    parser.add_argument(
        "--interp",
        type=str,
        default="nearest",
        choices=["nearest", "bilinear", "bicubic"],
        help="Interpolation method for upscaling (default: nearest).",
    )
    return parser.parse_args()


def get_original_size(image_path: Path) -> tuple:
    """Get (H, W) of original image."""
    img = Image.open(image_path)
    return img.size[::-1]  # PIL returns (W, H), we need (H, W)


def upscale_mask(pred_path: Path, target_size: tuple, interp: str) -> np.ndarray:
    """Load prediction and upscale to target size."""
    pred = Image.open(pred_path)
    palette = pred.getpalette() if pred.mode == "P" else None
    pred_arr = np.array(pred)

    # Handle RGB images -> take first channel
    if pred_arr.ndim == 3:
        pred_arr = pred_arr[:, :, 0]

    # Upscale using OpenCV
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
    }
    upscaled = cv2.resize(pred_arr, (target_size[1], target_size[0]), interpolation=interp_map[interp])
    return upscaled, palette


def save_mask(mask: np.ndarray, output_path: Path, palette=None):
    """Save mask as image."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if palette is not None:
        img = Image.fromarray(mask, mode="P")
        img.putpalette(palette)
        img.save(output_path)
    else:
        Image.fromarray(mask).save(output_path)


def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)

    if not pred_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {pred_dir}")
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    # Find all prediction files
    pred_files = sorted([p for p in pred_dir.rglob("*.png")])
    if not pred_files:
        raise RuntimeError(f"No PNG files found in {pred_dir}")

    print(f"Found {len(pred_files)} predictions to upscale.")
    print(f"Interpolation method: {args.interp}")

    processed = 0
    skipped = 0
    total_upscaled_pixels = 0

    for idx, pred_file in enumerate(pred_files, start=1):
            # Get just the filename (without extension) and try to match with different extensions
            stem = pred_file.stem  # filename without extension
        
            # Try different image extensions (prioritize jpg, then others)
            img_file = None
            for ext in [".jpg", ".jpeg", ".JPG", ".JPEG", ".png", ".PNG"]:
                img_candidate = image_dir / f"{stem}{ext}"
                if img_candidate.exists():
                    img_file = img_candidate
                    break
        
            if img_file is None:
                print(f"SKIP: {pred_file.name} (no corresponding image)")
                skipped += 1
                continue

            # Load and upscale
            try:
                orig_size = get_original_size(img_file)
                upscaled, palette = upscale_mask(pred_file, orig_size, args.interp)
            
                output_file = output_dir / pred_file.name
                output_file.parent.mkdir(parents=True, exist_ok=True)
                save_mask(upscaled, output_file, palette)
            
                pred_size = upscaled.shape
                total_upscaled_pixels += pred_size[0] * pred_size[1]
                processed += 1
            
                if idx % 50 == 0 or idx == len(pred_files):
                    print(f"Processed {idx}/{len(pred_files)} predictions")
            except Exception as e:
                print(f"ERROR {pred_file.name}: {e}")
                skipped += 1

    print(f"\nDone.")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Total pixels upscaled: {total_upscaled_pixels:,}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
