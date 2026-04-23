

from __future__ import annotations

import os
import argparse
from typing import Tuple

import cv2
import yaml
import numpy as np
import torch

from model import build_unet


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def pad_if_needed(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Symmetric pad to make H>=target_h and W>=target_w. Pad with 0 (black)."""
    h, w = image.shape[:2]
    pad_h = max(target_h - h, 0)
    pad_w = max(target_w - w, 0)
    if pad_h == 0 and pad_w == 0:
        return image

    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    image = cv2.copyMakeBorder(image, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return image


def center_crop(image: np.ndarray, crop_h: int, crop_w: int) -> np.ndarray:
    h, w = image.shape[:2]
    if h < crop_h or w < crop_w:
        image = pad_if_needed(image, crop_h, crop_w)
        h, w = image.shape[:2]

    y1 = max((h - crop_h) // 2, 0)
    x1 = max((w - crop_w) // 2, 0)
    return image[y1:y1 + crop_h, x1:x1 + crop_w]


def preprocess(image_rgb: np.ndarray, size: Tuple[int, int]) -> torch.Tensor:
    """Match dataset logic: pad(if needed) + center crop + /255 + CHW."""
    th, tw = size
    image_rgb = pad_if_needed(image_rgb, th, tw)
    image_rgb = center_crop(image_rgb, th, tw)

    x = image_rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))  # CHW
    x = torch.from_numpy(x).unsqueeze(0)  # 1CHW
    return x


def load_model(cfg: dict, device: torch.device) -> torch.nn.Module:
    model = build_unet(in_channels=cfg["model"]["in_channels"], out_channels=cfg["model"]["out_channels"])
    ckpt_path = cfg["paths"]["checkpoint_path"]
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    # support either direct state_dict or wrapped dict
    if isinstance(state, dict) and "model" in state:
        state_dict = state["model"]
    else:
        state_dict = state
    model.load_state_dict(state_dict, strict=True)

    # --- required by user ---
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model.to(device)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="config.yml")
    ap.add_argument("--image", type=str,default="inputs/fundus.png", help="Path to a fundus RGB image")
    ap.add_argument("--out", type=str, default="outputs/vessel_mask.png", help="Output mask path (PNG recommended)")
    ap.add_argument("--threshold", type=float, default=0.5, help="Binarization threshold on sigmoid(prob)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    size = tuple(cfg["data"]["size"])  # [H, W]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(cfg, device)

    img = _read_rgb(args.image)
    x = preprocess(img, size).to(device)

    with torch.no_grad():
        logits = model(x)  # 1x1xHxW
        prob = torch.sigmoid(logits)
        mask = (prob >= args.threshold).to(torch.uint8)  # 0/1

    mask_np = (mask.squeeze(0).squeeze(0).cpu().numpy() * 255).astype(np.uint8)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    cv2.imwrite(args.out, mask_np)  # single-channel png

    print(f"[OK] Saved binary vessel mask to: {args.out}")


if __name__ == "__main__":
    main()
