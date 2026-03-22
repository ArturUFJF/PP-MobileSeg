import argparse
import json
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from paddleseg.cvlibs import Config, SegBuilder
from paddleseg.utils import visualize


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect dataset loading and augmentation pipeline.')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file.')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'val', 'test'],
                        help='Dataset split to inspect.')
    parser.add_argument('--num_samples', type=int, default=20,
                        help='How many samples to export.')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Start index in file_list.')
    parser.add_argument('--output_dir', type=str,
                        default='output/dataset_debug',
                        help='Where to save debug outputs.')
    parser.add_argument('--repeats_per_sample', type=int, default=3,
                        help='How many stochastic augmentation passes to run for each sample index.')
    parser.add_argument('--save_steps', action='store_true',
                        help='Save intermediate image/mask after each transform op.')
    parser.add_argument('--seed', type=int, default=2026,
                        help='Random seed for reproducible augmentations.')
    parser.add_argument('--opts', nargs='+', default=None,
                        help='Override config options.')
    return parser.parse_args()


def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)


def load_raw_pair(image_path, label_path):
    img_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError(f'Cannot read image: {image_path}')
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    label = None
    if label_path is not None and os.path.exists(label_path):
        label = np.asarray(Image.open(label_path))
        if label.ndim == 3:
            label = label[:, :, 0]
    return img_rgb, label


def maybe_get_normalize_params(transforms):
    for op in transforms:
        if hasattr(op, 'mean') and hasattr(op, 'std'):
            mean = np.asarray(op.mean, dtype=np.float32).reshape([1, 1, -1])
            std = np.asarray(op.std, dtype=np.float32).reshape([1, 1, -1])
            return mean, std
    return None, None


def extract_transform_ops(dataset):
    # Dataset.transforms is a Compose object in PaddleSeg datasets.
    compose = getattr(dataset, 'transforms', None)
    if compose is None:
        return []
    return getattr(compose, 'transforms', [])


def to_visual_image_any(img_arr, mean=None, std=None):
    img = np.asarray(img_arr)

    # Accept both CHW and HWC layouts.
    if img.ndim == 2:
        img = img[:, :, np.newaxis]
    elif img.ndim == 3 and img.shape[0] in [1, 3] and img.shape[2] not in [1, 3]:
        img = np.transpose(img, (1, 2, 0))

    img = img.astype(np.float32)

    # Apply denormalization only when image already looks normalized.
    # Before Normalize, values are usually in [0, 255] and denormalizing would
    # saturate to white.
    looks_normalized = (
        float(np.min(img)) >= -10.0 and float(np.max(img)) <= 10.0)

    if mean is not None and std is not None and img.shape[2] in [1, 3] and looks_normalized:
        if mean.shape[2] == 1 and img.shape[2] == 3:
            mean = np.repeat(mean, 3, axis=2)
            std = np.repeat(std, 3, axis=2)
        img = (img * std + mean) * 255.0
    else:
        img_min = float(np.min(img))
        img_max = float(np.max(img))
        if img_max > img_min:
            img = 255.0 * (img - img_min) / (img_max - img_min)
        else:
            img = np.zeros_like(img)

    img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)
    return img


def colorize_mask(mask, color_map):
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id in np.unique(mask):
        if cls_id < 0:
            continue
        c = color_map[int(cls_id) % 256]
        out[mask == cls_id] = c
    return out


def overlay(img_rgb, color_mask, alpha=0.55):
    if img_rgb.shape[:2] != color_mask.shape[:2]:
        color_mask = cv2.resize(
            color_mask,
            (img_rgb.shape[1], img_rgb.shape[0]),
            interpolation=cv2.INTER_NEAREST)
    return cv2.addWeighted(img_rgb, 1.0 - alpha, color_mask, alpha, 0)


def mask_stats(mask):
    if mask is None:
        return None
    uniques, counts = np.unique(mask, return_counts=True)
    stats = {}
    for u, c in zip(uniques.tolist(), counts.tolist()):
        stats[int(u)] = int(c)
    return stats


def save_rgb(path, img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), img_bgr)


def apply_transforms_with_optional_steps(raw_img, raw_mask, transform_ops,
                                         save_steps=False,
                                         sample_dir=None,
                                         mean=None,
                                         std=None,
                                         color_map=None):
    data = {
        'img': raw_img.astype(np.float32).copy(),
        'trans_info': [],
        'gt_fields': []
    }
    if raw_mask is not None:
        data['label'] = raw_mask.copy()
        data['gt_fields'].append('label')

    if save_steps and sample_dir is not None:
        step0 = sample_dir / 'steps' / '00_input'
        step0.mkdir(parents=True, exist_ok=True)
        save_rgb(step0 / 'img.png', raw_img)
        if raw_mask is not None and color_map is not None:
            m_color = colorize_mask(raw_mask.astype(np.int32), color_map)
            save_rgb(step0 / 'overlay.png', overlay(raw_img, m_color))

    for op_idx, op in enumerate(transform_ops, start=1):
        data = op(data)
        if save_steps and sample_dir is not None:
            op_name = op.__class__.__name__
            step_dir = sample_dir / 'steps' / f'{op_idx:02d}_{op_name}'
            step_dir.mkdir(parents=True, exist_ok=True)
            img_vis = to_visual_image_any(data['img'], mean=mean, std=std)
            save_rgb(step_dir / 'img.png', img_vis)
            if 'label' in data and color_map is not None:
                mask = np.asarray(data['label'])
                if mask.ndim == 3:
                    mask = mask[:, :, 0]
                mask = mask.astype(np.int32)
                m_color = colorize_mask(mask, color_map)
                save_rgb(step_dir / 'overlay.png', overlay(img_vis, m_color))

    return data


def main():
    args = parse_args()
    set_seed(args.seed)

    cfg = Config(args.config, opts=args.opts)
    builder = SegBuilder(cfg)

    if args.split == 'train':
        dataset = builder.train_dataset
    elif args.split == 'val':
        dataset = builder.val_dataset
    else:
        dataset = builder.test_dataset

    transform_ops = extract_transform_ops(dataset)

    mean, std = maybe_get_normalize_params(transform_ops)
    color_map_flat = visualize.get_color_map_list(256)
    color_map = np.asarray(color_map_flat, dtype=np.uint8).reshape(-1, 3)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    report = {
        'config': args.config,
        'split': args.split,
        'seed': args.seed,
        'dataset_len': len(dataset),
        'start_idx': args.start_idx,
        'num_samples': args.num_samples,
        'samples': []
    }

    end_idx = min(len(dataset), args.start_idx + args.num_samples)
    for idx in range(args.start_idx, end_idx):
        image_path, label_path = dataset.file_list[idx]
        raw_img, raw_mask = load_raw_pair(image_path, label_path)

        for rep in range(max(1, args.repeats_per_sample)):
            sample_dir = out_root / f'sample_{idx:05d}_rep_{rep:02d}'
            sample_dir.mkdir(parents=True, exist_ok=True)

            data = apply_transforms_with_optional_steps(
                raw_img=raw_img,
                raw_mask=raw_mask,
                transform_ops=transform_ops,
                save_steps=args.save_steps,
                sample_dir=sample_dir,
                mean=mean,
                std=std,
                color_map=color_map)

            aug_img = to_visual_image_any(data['img'], mean=mean, std=std)
            aug_mask = data.get('label', None)

            if aug_mask is not None:
                aug_mask = np.asarray(aug_mask)
                if aug_mask.ndim == 3:
                    aug_mask = aug_mask[:, :, 0]
                aug_mask = aug_mask.astype(np.int32)

            save_rgb(sample_dir / 'raw_image.png', raw_img)
            save_rgb(sample_dir / 'aug_image.png', aug_img)

            if raw_mask is not None:
                raw_mask_color = colorize_mask(raw_mask.astype(np.int32), color_map)
                save_rgb(sample_dir / 'raw_mask_color.png', raw_mask_color)
                save_rgb(sample_dir / 'raw_overlay.png', overlay(raw_img, raw_mask_color))
                cv2.imwrite(str(sample_dir / 'raw_mask_id.png'), raw_mask.astype(np.uint16))

            if aug_mask is not None:
                aug_mask_color = colorize_mask(aug_mask, color_map)
                save_rgb(sample_dir / 'aug_mask_color.png', aug_mask_color)
                save_rgb(sample_dir / 'aug_overlay.png', overlay(aug_img, aug_mask_color))
                cv2.imwrite(str(sample_dir / 'aug_mask_id.png'), aug_mask.astype(np.uint16))

            raw_aug_equal = None
            if raw_mask is not None and aug_mask is not None and raw_mask.shape == aug_mask.shape:
                raw_aug_equal = bool(np.array_equal(raw_mask, aug_mask))

            sample_info = {
                'idx': idx,
                'rep': rep,
                'image_path': image_path,
                'label_path': label_path,
                'raw_shape': list(raw_img.shape),
                'aug_shape': list(data['img'].shape),
                'raw_mask_stats': mask_stats(raw_mask),
                'aug_mask_stats': mask_stats(aug_mask),
                'raw_mask_equals_aug_mask': raw_aug_equal,
                'transform_ops': [op.__class__.__name__ for op in transform_ops],
            }
            with open(sample_dir / 'info.json', 'w', encoding='utf-8') as f:
                json.dump(sample_info, f, indent=2, ensure_ascii=False)

            report['samples'].append(sample_info)

    with open(out_root / 'report.json', 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f'Saved debug artifacts to: {out_root}')


if __name__ == '__main__':
    main()
