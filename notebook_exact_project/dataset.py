import random
from glob import glob
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def list_sorted(pattern: str) -> List[str]:
    return sorted(glob(pattern))


def _pad_to_min_size(img: np.ndarray, min_h: int, min_w: int, pad_value: int = 0) -> np.ndarray:
    """Symmetric pad to ensure img has at least (min_h, min_w)."""
    h, w = img.shape[:2]
    pad_h = max(min_h - h, 0)
    pad_w = max(min_w - w, 0)

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    if pad_h == 0 and pad_w == 0:
        return img

    return cv2.copyMakeBorder(
        img,
        top, bottom, left, right,
        borderType=cv2.BORDER_CONSTANT,
        value=pad_value if img.ndim == 2 else (pad_value, pad_value, pad_value),
    )


def _crop(img: np.ndarray, crop_h: int, crop_w: int, y: int, x: int) -> np.ndarray:
    return img[y:y + crop_h, x:x + crop_w]


class DriveDataset(Dataset):
    """
    Dataset aligned with the notebook's logic (simple read -> normalize -> tensor),
    with ONE change to support raw 565x584 data while keeping 512x512 training:

    - Instead of resizing (which changes vessel scale), we do **pad + crop** to 512x512:
        1) Symmetric pad (only if needed) to ensure at least 512x512.
        2) Crop to exactly 512x512 (center-crop by default; optional random crop).

    This preserves local structures better than resize and is closer to the common
    preprocessing used in many vessel-seg notebooks that assume 512x512 inputs.
    """

    def __init__(
            self,
            images_path: List[str],
            masks_path: List[str],
            size: Tuple[int, int] = (512, 512),
            random_crop: bool = False,
            seed: int = 42,
    ):
        assert len(images_path) == len(masks_path), "images and masks length mismatch"
        self.images_path = images_path
        self.masks_path = masks_path
        self.n_samples = len(images_path)
        self.size = size  # (H, W)
        self.random_crop = random_crop
        self._rng = random.Random(seed)

    def _get_crop_xy(self, h: int, w: int, crop_h: int, crop_w: int) -> Tuple[int, int]:
        """Get (y, x) for cropping; center by default, optionally random."""
        if h == crop_h and w == crop_w:
            return 0, 0

        max_y = max(h - crop_h, 0)
        max_x = max(w - crop_w, 0)

        if self.random_crop:
            y = self._rng.randint(0, max_y) if max_y > 0 else 0
            x = self._rng.randint(0, max_x) if max_x > 0 else 0
        else:
            y = max_y // 2
            x = max_x // 2
        return y, x

    def _read_pair(self, img_path: str, mask_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR
        if image is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {mask_path}")

        crop_h, crop_w = self.size

        # 1) Pad (only if smaller than target)
        image = _pad_to_min_size(image, crop_h, crop_w, pad_value=0)
        mask = _pad_to_min_size(mask, crop_h, crop_w, pad_value=0)

        # 2) Crop to target (center or random), same coords for both
        h, w = image.shape[:2]
        y, x = self._get_crop_xy(h, w, crop_h, crop_w)

        image = _crop(image, crop_h, crop_w, y, x)
        mask = _crop(mask, crop_h, crop_w, y, x)

        # --- Notebook-style normalization/format ---
        image = image / 255.0
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)  # CHW

        # Keep mask crisp; ensure in [0,1]
        mask = (mask / 255.0).astype(np.float32)
        mask = np.expand_dims(mask, axis=0)  # 1HW

        return torch.from_numpy(image), torch.from_numpy(mask)

    def __getitem__(self, index: int):
        return self._read_pair(self.images_path[index], self.masks_path[index])

    def __len__(self) -> int:
        return self.n_samples
