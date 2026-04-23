import os
# 跟 train.py 一样，限制 CPU 线程，避免推理时 CPU 爆炸（尤其是 cv2/BLAS）:contentReference[oaicite:2]{index=2}
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("cv2_NUM_THREADS", "0")

import time
import glob
import argparse
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import cv2


# -----------------------------
# from train.py (same logic) :contentReference[oaicite:3]{index=3}
# -----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_path(p: str, base_dir: str) -> str:
    if p is None:
        return p
    p = os.path.expanduser(p)
    if os.path.isabs(p):
        return p
    return os.path.normpath(os.path.join(base_dir, p))


def dynamic_import(module_path: str):
    import importlib.util
    spec = importlib.util.spec_from_file_location("dyn_mod", module_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)
    return mod


def build_student(student_cfg: Dict[str, Any]) -> nn.Module:
    """
    支持两种方式：
    - build_fn：model_py 里提供 build_model() -> nn.Module
    - class_name：model_py 里提供 class（nn.Module），用 kwargs 初始化
    """
    model_py = student_cfg["model_py"]
    mod = dynamic_import(model_py)

    build_fn = student_cfg.get("build_fn", "")
    class_name = student_cfg.get("class_name", "")
    kwargs = student_cfg.get("kwargs", {}) or {}

    if build_fn:
        if not hasattr(mod, build_fn):
            raise AttributeError(f"Student build_fn '{build_fn}' not found in {model_py}")
        model = getattr(mod, build_fn)(**kwargs) if kwargs else getattr(mod, build_fn)()
        if not isinstance(model, nn.Module):
            raise TypeError("build_fn must return torch.nn.Module")
        return model

    if class_name:
        if not hasattr(mod, class_name):
            raise AttributeError(f"Student class_name '{class_name}' not found in {model_py}")
        cls = getattr(mod, class_name)
        model = cls(**kwargs)
        if not isinstance(model, nn.Module):
            raise TypeError("class_name must be torch.nn.Module")
        return model

    raise ValueError("student_cfg must set either build_fn or class_name.")


# -----------------------------
# image io / padding
# -----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def imwrite_rgb_uint8(path: str, rgb_uint8: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    bgr = cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)


def to_tensor_01(rgb: np.ndarray) -> torch.Tensor:
    x = rgb.astype(np.float32) / 255.0
    x = torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)  # 1,C,H,W
    return x


def pad_to_multiple(x: torch.Tensor, multiple: int = 16) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # x: 1,C,H,W
    if multiple <= 1:
        return x, (x.shape[-2], x.shape[-1])
    h, w = x.shape[-2], x.shape[-1]
    ph = (multiple - (h % multiple)) % multiple
    pw = (multiple - (w % multiple)) % multiple
    if ph == 0 and pw == 0:
        return x, (h, w)
    # pad right/bottom reflect
    x_pad = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="reflect")
    return x_pad, (h, w)


@torch.no_grad()
def forward_model(
    model: nn.Module,
    x: torch.Tensor,
    *,
    use_amp: bool,
    pad_multiple: int,
) -> torch.Tensor:
    x_pad, (h0, w0) = pad_to_multiple(x, pad_multiple)
    with torch.cuda.amp.autocast(enabled=use_amp):
        y = model(x_pad)
    y = y.float()
    return y[..., :h0, :w0]


def make_hann2d(tile: int, device: torch.device) -> torch.Tensor:
    # (tile, tile)
    w1 = torch.hann_window(tile, periodic=False, device=device).clamp(min=1e-4)
    w2 = w1[:, None] * w1[None, :]
    return w2


@torch.no_grad()
def forward_tiled(
    model: nn.Module,
    x: torch.Tensor,
    *,
    tile: int,
    overlap: int,
    use_amp: bool,
    pad_multiple: int,
) -> torch.Tensor:
    """
    x: 1,C,H,W
    使用 overlap + Hann 权重融合，减小拼接缝。
    """
    assert x.dim() == 4 and x.size(0) == 1
    b, c, h, w = x.shape
    device = x.device
    stride = max(1, tile - overlap)

    # 右/下补齐到能覆盖完整网格
    n_y = int(np.ceil(max(1, h - tile) / stride)) + 1
    n_x = int(np.ceil(max(1, w - tile) / stride)) + 1
    pad_h = max(0, (n_y - 1) * stride + tile - h)
    pad_w = max(0, (n_x - 1) * stride + tile - w)
    x_pad = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    _, _, hp, wp = x_pad.shape

    out_acc = torch.zeros((1, c, hp, wp), device=device, dtype=torch.float32)
    w_acc = torch.zeros((1, 1, hp, wp), device=device, dtype=torch.float32)

    w2 = make_hann2d(tile, device=device).view(1, 1, tile, tile)

    for iy in range(n_y):
        y0 = iy * stride
        y1 = y0 + tile
        for ix in range(n_x):
            x0 = ix * stride
            x1 = x0 + tile

            patch = x_pad[..., y0:y1, x0:x1]
            pred = forward_model(model, patch, use_amp=use_amp, pad_multiple=pad_multiple)  # 1,C,tile,tile
            out_acc[..., y0:y1, x0:x1] += pred * w2
            w_acc[..., y0:y1, x0:x1] += w2

    out = out_acc / w_acc.clamp(min=1e-8)
    out = out[..., :h, :w]
    return out


def list_images_from_dir(input_dir: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
    paths = sorted(list(set(paths)))
    return paths


def list_images_from_file(list_file: str, base_dir: Optional[str] = None) -> List[str]:
    base_dir = base_dir or os.path.dirname(os.path.abspath(list_file))
    paths: List[str] = []
    with open(list_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            # 只取第一列（常见：txt 每行一个路径，或 “inp gt” 这种）
            p0 = s.split()[0]
            if not os.path.isabs(p0):
                p0 = os.path.normpath(os.path.join(base_dir, p0))
            paths.append(p0)
    return paths


def load_ckpt_to_model(model: nn.Module, ckpt_path: str, use_ema: bool = True) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        if use_ema and ("ema" in ckpt):
            sd = ckpt["ema"]
        elif "model" in ckpt:
            sd = ckpt["model"]
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            # 有些人直接 torch.save(model.state_dict())
            # 或者 dict 里只有一堆权重键
            maybe_weight_keys = [k for k in ckpt.keys() if isinstance(ckpt[k], torch.Tensor)]
            if len(maybe_weight_keys) > 10:
                sd = ckpt
            else:
                raise KeyError(f"Unsupported checkpoint dict keys: {list(ckpt.keys())[:20]}")
    else:
        raise TypeError("Checkpoint must be a dict.")
    model.load_state_dict(sd, strict=True)


def main():
    parser = argparse.ArgumentParser("Batch inference: pick 20 images and run sequentially")
    # 默认沿用 train.py 的 config 默认值 :contentReference[oaicite:4]{index=4}
    parser.add_argument("--config", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/train.yml")
    parser.add_argument("--ckpt", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/work_fundus_mamba/checkpoints/latest.pth", help="if empty -> use work_dir/checkpoints/latest.pth")
    parser.add_argument("--use_ema", type=int, default=1, help="1: use ema weights if present; 0: use raw model")
    parser.add_argument("--device", type=str, default="cuda", help="cuda / cpu / cuda:0 ...")

    # 数据来源：优先 list_file；否则用 input_dir 扫描
    parser.add_argument("--list_file", type=str, default="", help="txt file containing image paths (one per line or first column)")
    parser.add_argument("--input_dir", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/input", help="fallback: directory to scan images")
    parser.add_argument("--out_dir", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/work_fundus_mamba/infer_out", help="output directory")
    parser.add_argument("--num", type=int, default=120, help="how many images to infer")
    parser.add_argument("--start", type=int, default=0, help="start index after sorting")
    parser.add_argument("--shuffle", type=int, default=0, help="1: shuffle before picking")

    # 推理设置（给你做了相对安全的默认）
    parser.add_argument("--use_amp", type=int, default=1)
    parser.add_argument("--pad_to", type=int, default=16, help="pad H/W to multiple for U-Net-like nets")
    parser.add_argument("--tile", type=int, default=1024, help="0 = no tiling; otherwise tile size")
    parser.add_argument("--overlap", type=int, default=128, help="overlap pixels for tiled inference")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))

    # work_dir / ckpt_dir 命名跟 train.py 一致（work_dir/checkpoints/latest.pth）:contentReference[oaicite:5]{index=5}
    work_dir = resolve_path(cfg["work_dir"], base_dir)
    ckpt_default = os.path.join(work_dir, "checkpoints", "latest.pth")
    ckpt_path = resolve_path(args.ckpt, base_dir) if args.ckpt.strip() else ckpt_default

    device = torch.device(args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu")

    # student model（复用 train.py 的动态构建方式）:contentReference[oaicite:6]{index=6}
    student_cfg = cfg["model"]["student"].copy()
    student_cfg["model_py"] = resolve_path(student_cfg["model_py"], base_dir)
    model = build_student(student_cfg).to(device)
    model.eval()

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    load_ckpt_to_model(model, ckpt_path, use_ema=bool(args.use_ema))

    use_amp = bool(args.use_amp) and (device.type == "cuda")
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    # collect images
    paths: List[str] = []
    if args.list_file.strip() and os.path.exists(args.list_file):
        paths = list_images_from_file(args.list_file, base_dir=os.path.dirname(os.path.abspath(args.list_file)))
    else:
        paths = list_images_from_dir(args.input_dir)

    if len(paths) == 0:
        raise RuntimeError(f"No images found. list_file='{args.list_file}' input_dir='{args.input_dir}'")

    if args.shuffle:
        rng = np.random.default_rng(1234)
        rng.shuffle(paths)
    else:
        paths = sorted(paths)

    sel = paths[args.start: args.start + args.num]
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[Infer] device={device} amp={use_amp} tile={args.tile} overlap={args.overlap} pad_to={args.pad_to}")
    print(f"[Infer] config={args.config}")
    print(f"[Infer] ckpt={ckpt_path} (use_ema={bool(args.use_ema)})")
    print(f"[Infer] total_found={len(paths)} pick={len(sel)} start={args.start}")
    t_all = time.time()

    for i, p in enumerate(sel):
        name = os.path.basename(p)
        out_path = os.path.join(args.out_dir, name)

        t0 = time.time()
        rgb = imread_rgb(p)
        x = to_tensor_01(rgb).to(device, non_blocking=True)

        with torch.no_grad():
            if args.tile and args.tile > 0:
                y = forward_tiled(
                    model,
                    x,
                    tile=int(args.tile),
                    overlap=int(args.overlap),
                    use_amp=use_amp,
                    pad_multiple=int(args.pad_to),
                )
            else:
                y = forward_model(model, x, use_amp=use_amp, pad_multiple=int(args.pad_to))

        y = y.clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()  # HWC
        y8 = (y * 255.0 + 0.5).astype(np.uint8)
        imwrite_rgb_uint8(out_path, y8)

        dt = time.time() - t0
        print(f"[{i+1:02d}/{len(sel):02d}] {name}  time={dt:.3f}s  -> {out_path}")

    print(f"[Done] total_time={time.time()-t_all:.2f}s  saved_to={args.out_dir}")


if __name__ == "__main__":
    main()