# infer.py
import os

# 跟 train.py 一致：默认锁一张卡（需要就改）
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
# 严格限制底层计算库的线程数，防止 CPU 爆炸
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("cv2_NUM_THREADS", "0")

import argparse
import glob
from typing import Dict, Any, List, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cv2
except ImportError as e:
    raise ImportError("Please install opencv-python (cv2) for image I/O") from e


# -----------------------------
# train.py style helpers
# -----------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def resolve_path(p: str, base_dir: str) -> str:
    if p is None:
        return p
    p = os.path.expanduser(str(p))
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
    与 train.py 同逻辑：
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
        fn = getattr(mod, build_fn)
        model = fn(**kwargs) if kwargs else fn()
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


def strip_module_prefix(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not state:
        return state
    k0 = next(iter(state.keys()))
    if k0.startswith("module."):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def pick_ckpt_auto(ckpt_dir: str) -> str:
    """
    优先级：
    1) best_ema_psnr.pth
    2) latest.pth
    3) interrupt.pth
    4) epoch_*.pth (最大 epoch)
    """
    cands = [
        os.path.join(ckpt_dir, "best_ema_psnr.pth"),
        os.path.join(ckpt_dir, "latest.pth"),
        os.path.join(ckpt_dir, "interrupt.pth"),
    ]
    for p in cands:
        if os.path.exists(p):
            return p

    ep_files = sorted(glob.glob(os.path.join(ckpt_dir, "epoch_*.pth")))
    if ep_files:
        # 取 epoch 最大的
        def ep_num(fp: str) -> int:
            base = os.path.splitext(os.path.basename(fp))[0]
            try:
                return int(base.split("_")[-1])
            except Exception:
                return -1
        ep_files = sorted(ep_files, key=ep_num)
        return ep_files[-1]

    raise FileNotFoundError(
        f"No checkpoint found in {ckpt_dir}. "
        f"Expected one of: best_ema_psnr.pth / latest.pth / interrupt.pth / epoch_*.pth"
    )


def _has_images(dir_path: str) -> bool:
    exts = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    for e in exts:
        if glob.glob(os.path.join(dir_path, e)):
            return True
    return False


def pick_input_auto(work_dir: str, data_root: str, cfg: Dict[str, Any]) -> str:
    """
    先看 cfg 里是否提供了 infer 输入（若存在且有效就用），否则按常见目录猜：
    - work_dir/infer_in
    - work_dir/test_pho/input
    - work_dir/test/input
    - work_dir/test_images
    - data_root/input
    - data_root/test/input
    - data_root/test
    - data_root
    """
    # 允许你在 train.yml 里写 infer: { input_dir: "xxx" }
    infer_cfg = cfg.get("infer", {}) or {}
    key_order = ["input_dir", "input", "in_dir", "test_dir"]
    for k in key_order:
        v = infer_cfg.get(k, "")
        if v:
            v_abs = resolve_path(v, os.path.dirname(os.path.abspath(args_default_config())))
            if os.path.isdir(v_abs) and _has_images(v_abs):
                return v_abs
            if os.path.isfile(v_abs):
                return v_abs

    candidates = [
        os.path.join(work_dir, "infer_in"),
        os.path.join(work_dir, "test_pho", "input"),
        os.path.join(work_dir, "test", "input"),
        os.path.join(work_dir, "test_images"),
        os.path.join(data_root, "input"),
        os.path.join(data_root, "test", "input"),
        os.path.join(data_root, "test"),
        data_root,
    ]
    for p in candidates:
        if os.path.isdir(p) and _has_images(p):
            return p

    # 最后兜底：递归找一张图（限制扫描量）
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
    found = []
    max_files = 20
    max_depth = 4
    base_depth = data_root.rstrip(os.sep).count(os.sep)
    for root, dirs, files in os.walk(data_root):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth > max_depth:
            dirs[:] = []
            continue
        for fn in files:
            if fn.lower().endswith(exts):
                found.append(os.path.join(root, fn))
                if len(found) >= max_files:
                    break
        if len(found) >= max_files:
            break

    if found:
        return os.path.dirname(found[0])

    raise FileNotFoundError(
        "Cannot auto-detect input images. Please either:\n"
        f"1) Put images into: {os.path.join(work_dir, 'infer_in')}\n"
        "or\n"
        "2) Run with --input /path/to/images_or_dir\n"
    )


# -----------------------------
# Image IO
# -----------------------------
def imread_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Failed to read image: {path}")
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    if img.shape[2] > 3:
        img = img[:, :, :3]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype == np.uint16:
        img_f = img.astype(np.float32) / 65535.0
    else:
        img_f = img.astype(np.float32) / 255.0
    return np.clip(img_f, 0.0, 1.0)


def imsave_rgb01(path: str, img01: np.ndarray):
    img01 = np.clip(img01, 0.0, 1.0)
    img_u8 = (img01 * 255.0 + 0.5).astype(np.uint8)
    img_bgr = cv2.cvtColor(img_u8, cv2.COLOR_RGB2BGR)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img_bgr)


# -----------------------------
# Tiling (quality-first)
# -----------------------------
def make_positions(length: int, tile: int, step: int) -> List[int]:
    if length <= tile:
        return [0]
    pos = list(range(0, length - tile, step))
    pos.append(length - tile)
    return pos


def make_1d_weight(tile: int, overlap: int, at_start: bool, at_end: bool) -> torch.Tensor:
    w = torch.ones(tile, dtype=torch.float32)
    if overlap <= 0:
        return w
    ramp = torch.linspace(0.0, 1.0, steps=overlap, dtype=torch.float32)
    if not at_start:
        w[:overlap] = ramp
    if not at_end:
        w[-overlap:] = ramp.flip(0)
    return w


def make_2d_weight(tile: int, overlap: int, *, at_left: bool, at_right: bool, at_top: bool, at_bottom: bool) -> torch.Tensor:
    wx = make_1d_weight(tile, overlap, at_left, at_right)
    wy = make_1d_weight(tile, overlap, at_top, at_bottom)
    return torch.ger(wy, wx)  # (tile,tile)


def pad_to_multiple(x: torch.Tensor, multiple: int, mode: str = "reflect") -> Tuple[torch.Tensor, Tuple[int, int]]:
    if multiple <= 1:
        return x, (0, 0)
    _, _, h, w = x.shape
    pad_b = (multiple - (h % multiple)) % multiple
    pad_r = (multiple - (w % multiple)) % multiple
    if pad_b == 0 and pad_r == 0:
        return x, (0, 0)
    x = F.pad(x, pad=(0, pad_r, 0, pad_b), mode=mode)
    return x, (pad_b, pad_r)


def _unwrap_pred(pred):
    if isinstance(pred, (tuple, list)):
        return pred[0]
    if isinstance(pred, dict):
        # 常见：{"pred": tensor} / {"out": tensor}
        for k in ("pred", "out", "output"):
            if k in pred and torch.is_tensor(pred[k]):
                return pred[k]
        # 找第一个 tensor
        for v in pred.values():
            if torch.is_tensor(v):
                return v
    return pred


@torch.no_grad()
def forward_full(model: nn.Module, x: torch.Tensor, use_amp: bool) -> torch.Tensor:
    with torch.cuda.amp.autocast(enabled=use_amp):
        pred = model(x)
    pred = _unwrap_pred(pred)
    return pred


@torch.no_grad()
def forward_tiled(model: nn.Module, x: torch.Tensor, tile: int, overlap: int, use_amp: bool) -> torch.Tensor:
    assert x.ndim == 4 and x.shape[0] == 1
    device = x.device
    _, c, h, w = x.shape

    step = max(1, tile - overlap)
    ys = make_positions(h, tile, step)
    xs = make_positions(w, tile, step)

    out_acc = torch.zeros((1, c, h, w), device=device, dtype=torch.float32)
    w_acc = torch.zeros((1, 1, h, w), device=device, dtype=torch.float32)

    for y0 in ys:
        for x0 in xs:
            y1, x1 = y0 + tile, x0 + tile
            patch = x[:, :, y0:y1, x0:x1]

            w2 = make_2d_weight(
                tile, overlap,
                at_left=(x0 == 0),
                at_right=(x1 == w),
                at_top=(y0 == 0),
                at_bottom=(y1 == h),
            ).to(device)
            w2 = w2.unsqueeze(0).unsqueeze(0)  # (1,1,tile,tile)

            with torch.cuda.amp.autocast(enabled=use_amp):
                pred = model(patch)
            pred = _unwrap_pred(pred)
            pred_f = pred.float()

            out_acc[:, :, y0:y1, x0:x1] += pred_f * w2
            w_acc[:, :, y0:y1, x0:x1] += w2

    return out_acc / torch.clamp(w_acc, min=1e-8)


# -----------------------------
# Defaults (match your train.py)
# -----------------------------
def args_default_config() -> str:
    # train.py 里 --config 的 default 值
    return "/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/train.yml"


def main():
    parser = argparse.ArgumentParser()

    # 全部给 default：可以直接 python infer.py
    parser.add_argument("--config", type=str, default=args_default_config(), help="train.yml path")
    parser.add_argument("--ckpt", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/work_fundus_mamba/checkpoints/latest.pth", help="(optional) checkpoint path; empty -> auto pick from work_dir/checkpoints")
    parser.add_argument("--which", type=str, default="ema", choices=["ema", "model"], help="load ckpt['ema'] or ckpt['model']")
    parser.add_argument("--input", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/input/00031.PNG", help="image path or dir; empty -> auto detect")
    parser.add_argument("--output_dir", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/work_fundus_mamba", help="empty -> work_dir/infer_out")

    # 质量优先：默认 tile 推理（2560^2 不容易 OOM）
    parser.add_argument("--tile", type=int, default=512)
    parser.add_argument("--overlap", type=int, default=64)
    parser.add_argument("--pad_multiple", type=int, default=16)

    # 默认按 train.yml 里的 use_amp（如果不存在，默认 True）
    parser.add_argument("--amp", type=int, default=-1, help="-1: follow train.yml; 0: off; 1: on")

    parser.add_argument("--suffix", type=str, default="", help="empty -> _ema/_model")

    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args.config))
    cfg = load_yaml(args.config)

    # resolve paths
    work_dir = resolve_path(cfg.get("work_dir", "."), base_dir)
    data_root = resolve_path(cfg.get("data", {}).get("data_root", work_dir), base_dir)
    ckpt_dir = os.path.join(work_dir, "checkpoints")

    # output dir
    out_dir = args.output_dir.strip() or os.path.join(work_dir, "infer_out")
    os.makedirs(out_dir, exist_ok=True)

    # amp default
    if args.amp == -1:
        use_amp = bool(cfg.get("train", {}).get("use_amp", True))
    else:
        use_amp = bool(args.amp)

    # speed toggles (same spirit as train.py)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    student_cfg = cfg["model"]["student"].copy()
    student_cfg["model_py"] = resolve_path(student_cfg["model_py"], base_dir)
    model = build_student(student_cfg).to(device).eval()

    # pick ckpt
    ckpt_path = args.ckpt.strip()
    if not ckpt_path:
        ckpt_path = pick_ckpt_auto(ckpt_dir)
    else:
        ckpt_path = resolve_path(ckpt_path, base_dir)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    if args.which in ckpt:
        state = ckpt[args.which]
    else:
        # fallback
        state = ckpt.get("ema", None) or ckpt.get("model", None)
        if state is None:
            raise KeyError(f"Checkpoint has no 'ema'/'model'. keys={list(ckpt.keys())}")

    state = strip_module_prefix(state)
    model.load_state_dict(state, strict=True)

    # pick input
    in_path = args.input.strip()
    if not in_path:
        in_path = pick_input_auto(work_dir, data_root, cfg)

    # collect images
    img_paths: List[str] = []
    if os.path.isdir(in_path):
        patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff"]
        for pat in patterns:
            img_paths.extend(glob.glob(os.path.join(in_path, pat)))
        img_paths = sorted(list(set(img_paths)))
    else:
        img_paths = [in_path]

    if not img_paths:
        raise FileNotFoundError(f"No images found under: {in_path}")

    suf = args.suffix if args.suffix else f"_{args.which}"
    print(f"[Infer] device={device} amp={use_amp} which={args.which}")
    print(f"[Infer] config={args.config}")
    print(f"[Infer] ckpt={ckpt_path}")
    print(f"[Infer] input={in_path} (n={len(img_paths)})")
    print(f"[Infer] out_dir={out_dir}")
    print(f"[Infer] tile={args.tile} overlap={args.overlap} pad_multiple={args.pad_multiple}")

    for ip in img_paths:
        img01 = imread_rgb(ip)
        h0, w0 = img01.shape[:2]

        x = torch.from_numpy(img01).permute(2, 0, 1).unsqueeze(0).contiguous().to(device=device, dtype=torch.float32)
        x_pad, (pad_b, pad_r) = pad_to_multiple(x, multiple=args.pad_multiple, mode="reflect")

        if x_pad.shape[-2] <= args.tile and x_pad.shape[-1] <= args.tile:
            pred = forward_full(model, x_pad, use_amp=use_amp).float()
        else:
            pred = forward_tiled(model, x_pad, tile=args.tile, overlap=args.overlap, use_amp=use_amp)

        # crop pad
        if pad_b > 0:
            pred = pred[:, :, :-pad_b, :]
        if pad_r > 0:
            pred = pred[:, :, :, :-pad_r]

        pred01 = pred.clamp(0, 1).squeeze(0).permute(1, 2, 0).detach().cpu().numpy()
        if pred01.shape[0] != h0 or pred01.shape[1] != w0:
            raise RuntimeError(f"Shape mismatch: pred={pred01.shape}, input={(h0, w0)}")

        name = os.path.splitext(os.path.basename(ip))[0]
        op = os.path.join(out_dir, f"{name}{suf}.png")
        imsave_rgb01(op, pred01)
        print(f"[OK] {os.path.basename(ip)} -> {op}")


if __name__ == "__main__":
    main()