"""Custom dataset that loads semantic labels and per-pixel area masks together.

The file list (train/val) must provide three space separated columns:
```
relative/path/to/image.jpg relative/path/to/semantic.png relative/path/to/area_mask.raw
```
The area mask is expected to be stored as a raw binary array (default float32)
with the same spatial resolution as the semantic mask.
"""

import os
from typing import List, Optional

import numpy as np
from PIL import Image
import paddle

from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose


@manager.DATASETS.add_component
class LeafAreaDataset(paddle.io.Dataset):
    """Dataset that returns image, semantic label and a per-pixel area mask.

    Args:
        transforms (list): Transformations applied to the image and ground truths.
        dataset_root (str): Root directory for relative paths listed in the split files.
        num_classes (int): Number of semantic classes (PNG labels).
        mode (str): One of ("train", "val", "test").
        train_path (str): Path to txt with train samples. Each line must contain
            three columns (image, semantic label, area mask) when ``mode`` is "train".
        val_path (str): Path to txt with validation samples. Behaviour mirrors ``train_path``.
        test_path (str): Path to txt for test samples. Area mask column is optional
            and ignored.
        separator (str): Separator token inside the txt files. Default space.
        ignore_index (int): Ignore label value for semantic mask.
        img_channels (int): Number of channels of input imagery.
        area_mask_dtype (str): Numpy dtype string used when reading the raw area mask.
        area_mask_scale (float): Optional multiplicative scale applied to the loaded
            area mask values (useful to convert units).
    """

    def __init__(
        self,
        transforms: List,
        dataset_root: str,
        num_classes: int,
        mode: str = "train",
        train_path: Optional[str] = None,
        val_path: Optional[str] = None,
        test_path: Optional[str] = None,
        separator: str = " ",
        ignore_index: int = 255,
        img_channels: int = 3,
        area_mask_dtype: str = "float32",
        area_mask_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms, img_channels=img_channels)
        self.num_classes = num_classes
        self.mode = mode.lower()
        self.separator = separator
        self.ignore_index = ignore_index
        self.area_mask_dtype = np.dtype(area_mask_dtype)
        self.area_mask_scale = area_mask_scale
        self.file_list = []

        if self.mode not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported mode `{mode}`. Use train/val/test.")
        if not os.path.isdir(self.dataset_root):
            raise FileNotFoundError(f"dataset_root not found: {self.dataset_root}")
        if not isinstance(transforms, list) or len(transforms) == 0:
            raise ValueError("`transforms` must be a non-empty list")

        list_path = self._resolve_list_path(train_path, val_path, test_path)
        self._parse_file_list(list_path)

    # ---------------------------------------------------------------------
    def _resolve_list_path(self, train_path, val_path, test_path):
        if self.mode == "train":
            if train_path is None:
                raise ValueError("train_path must be provided for training mode")
            return train_path
        if self.mode == "val":
            if val_path is None:
                raise ValueError("val_path must be provided for val mode")
            return val_path
        if test_path is None:
            raise ValueError("test_path must be provided for test mode")
        return test_path

    def _parse_file_list(self, list_path: str) -> None:
        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Split file not found: {list_path}")

        with open(list_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                parts = line.strip().split(self.separator)
                if self.mode in {"train", "val"}:
                    if len(parts) != 3:
                        raise ValueError(
                            f"Expected 3 columns in `{list_path}` line {line_idx}, got {len(parts)}"
                        )
                    img_rel, lbl_rel, area_rel = parts
                    img_path = os.path.join(self.dataset_root, img_rel)
                    lbl_path = os.path.join(self.dataset_root, lbl_rel)
                    area_path = os.path.join(self.dataset_root, area_rel)
                    self.file_list.append((img_path, lbl_path, area_path))
                else:  # test mode
                    if len(parts) < 1:
                        continue
                    img_rel = parts[0]
                    lbl_path = None
                    area_path = None
                    if len(parts) > 1:
                        lbl_path = os.path.join(self.dataset_root, parts[1])
                    self.file_list.append((os.path.join(self.dataset_root, img_rel), lbl_path, area_path))

    # ---------------------------------------------------------------------
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx: int):
        data = {"trans_info": []}
        img_path, lbl_path, area_path = self.file_list[idx]

        data["img"] = img_path
        if lbl_path is not None:
            data["label"] = lbl_path
        data["gt_fields"] = []

        if self.mode in {"train", "val"} and lbl_path is not None and area_path is not None:
            data["gt_fields"].append("label")
            area_array = self._load_area_mask(area_path, lbl_path)
            data["area_mask"] = area_array
            data["gt_fields"].append("area_mask")
        elif self.mode == "test" and area_path is not None:
            # Optional area mask for evaluation/inference convenience.
            data["area_mask"] = self._load_area_mask(area_path, lbl_path)

        data = self.transforms(data)

        if "area_mask" in data:
            area = data["area_mask"]
            if area.ndim == 2:
                area = area[np.newaxis, ...]
            elif area.ndim == 3 and area.shape[2] != 1:
                area = np.transpose(area, (2, 0, 1))
            elif area.ndim == 3:  # H, W, 1
                area = np.transpose(area, (2, 0, 1))
            data["area_mask"] = area.astype("float32")

        return data

    # ------------------------------------------------------------------
    def _load_area_mask(self, area_path: str, lbl_path: Optional[str]) -> np.ndarray:
        """Load a raw mask and reshape it to match the semantic label dimensions."""

        if not os.path.exists(area_path):
            raise FileNotFoundError(f"Area mask not found: {area_path}")

        if lbl_path is None:
            raise ValueError("Semantic label path is required to infer mask shape")

        with Image.open(lbl_path) as label_img:
            width, height = label_img.size

        expected_size = width * height
        mask = np.fromfile(area_path, dtype=self.area_mask_dtype)
        if mask.size != expected_size:
            raise ValueError(
                f"Area mask {area_path} has {mask.size} values but expected {expected_size}"
            )
        mask = mask.reshape((height, width)) * self.area_mask_scale
        return mask.astype("float32")