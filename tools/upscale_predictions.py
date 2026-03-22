import argparse
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
    parser.add_argument(
        "--save_overlay",
        action="store_true",
        help="Also save overlay images (prediction mask blended on original image).",
    )
    parser.add_argument(
        "--overlay_dir",
        type=str,
        default=None,
        help="Directory to save overlay images. If omitted, uses <output_dir>/added_prediction.",
    )
    parser.add_argument(
        "--overlay_weight",
        type=float,
        default=0.6,
        help="Mask blend weight in overlay (0.0-1.0). Default: 0.6.",
    )
    parser.add_argument(
        "--custom_color",
        nargs="+",
        type=int,
        default=None,
        help="Optional color palette as flat RGB list, e.g. --custom_color 0 0 0 255 0 0 0 0 255",
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


def build_palette(custom_color=None) -> np.ndarray:
    """Build palette as [N, 3] RGB uint8."""
    if custom_color is not None:
        if len(custom_color) % 3 != 0:
            raise ValueError("--custom_color length must be a multiple of 3.")
        palette = np.array(custom_color, dtype=np.uint8).reshape(-1, 3)
        return palette

    # Default 3-class palette: background black, class-1 orange, class-2 blue.
    return np.array(
        [
            [0, 0, 0],
            [255, 0, 0],
            [0, 0, 255],
        ],
        dtype=np.uint8,
    )


def colorize_mask(mask: np.ndarray, palette: np.ndarray) -> np.ndarray:
    """Map class indices to RGB colors."""
    max_cls = int(mask.max()) if mask.size else 0
    if max_cls >= len(palette):
        extra = max_cls + 1 - len(palette)
        palette = np.vstack([palette, np.zeros((extra, 3), dtype=np.uint8)])
    return palette[mask]


def save_overlay_image(
    image_path: Path,
    mask: np.ndarray,
    overlay_path: Path,
    palette: np.ndarray,
    overlay_weight: float,
):
    """Save overlay image by blending original image and colorized mask."""
    overlay_path.parent.mkdir(parents=True, exist_ok=True)
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.uint8)
    color_mask = colorize_mask(mask, palette)
    blended = cv2.addWeighted(image, 1.0 - overlay_weight, color_mask, overlay_weight, 0.0)
    Image.fromarray(blended).save(overlay_path)


def main():
    args = parse_args()

    pred_dir = Path(args.pred_dir)
    image_dir = Path(args.image_dir)
    output_dir = Path(args.output_dir)
    overlay_dir = Path(args.overlay_dir) if args.overlay_dir else output_dir / "added_prediction"

    if not (0.0 <= args.overlay_weight <= 1.0):
        raise ValueError("--overlay_weight must be in [0.0, 1.0].")

    palette = build_palette(args.custom_color)

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
    overlays_saved = 0
    total_upscaled_pixels = 0

    for idx, pred_file in enumerate(pred_files, start=1):
        # Get just the filename (without extension) and try to match with different extensions.
        stem = pred_file.stem

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

        try:
            orig_size = get_original_size(img_file)
            upscaled, pred_palette = upscale_mask(pred_file, orig_size, args.interp)

            output_file = output_dir / pred_file.name
            save_mask(upscaled, output_file, pred_palette)

            if args.save_overlay:
                overlay_file = overlay_dir / pred_file.name
                save_overlay_image(
                    image_path=img_file,
                    mask=upscaled,
                    overlay_path=overlay_file,
                    palette=palette,
                    overlay_weight=args.overlay_weight,
                )
                overlays_saved += 1

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
    if args.save_overlay:
        print(f"  Overlays saved: {overlays_saved}")
        print(f"  Overlay directory: {overlay_dir}")
    print(f"  Total pixels upscaled: {total_upscaled_pixels:,}")
    print(f"  Output directory: {output_dir}")


if __name__ == "__main__":
    main()
