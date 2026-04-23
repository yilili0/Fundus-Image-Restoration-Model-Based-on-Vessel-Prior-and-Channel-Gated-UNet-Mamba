#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Sanity tests for fundus image restoration model (scale=1).

Includes:
  1) Shape + residual magnitude check
  2) Gradient flow / NaN-Inf grad check
  3) Peak GPU memory test (single size + optional size sweep)
  3c) Peak GPU memory batch sweep (fixed size)
  4) Hallucinated texture tendency test using simple synthetic inputs

Outputs:
  - Console logs
  - Optional CSV log (recommended for experiment tracking)

Assumes `model.py` is importable and defines `MambaRealSR11`.
"""

import os
import argparse
import time
from typing import Dict, Tuple, List, Optional

import torch
import torch.nn.functional as F

# ---- Import your model here ----
# Put this script in the same folder as model.py, or adjust PYTHONPATH / import accordingly.
from models.model import MambaRealSR11


def set_seed(seed: int = 1234):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_str: str):
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


@torch.no_grad()
def residual_stats(y: torch.Tensor, x: torch.Tensor) -> Dict[str, float]:
    diff = (y - x)
    return {
        "mean_abs": diff.abs().mean().item(),
        "max_abs": diff.abs().max().item(),
        "mean": diff.mean().item(),
        "std": diff.std(unbiased=False).item(),
    }


def hf_energy_laplacian(img: torch.Tensor) -> float:
    """
    High-frequency energy via discrete Laplacian magnitude (cheap & robust).
    img: [B, C, H, W]
    """
    lap = (
        img[:, :, :-2, 1:-1] + img[:, :, 2:, 1:-1] +
        img[:, :, 1:-1, :-2] + img[:, :, 1:-1, 2:] -
        4.0 * img[:, :, 1:-1, 1:-1]
    )
    return lap.abs().mean().item()


def make_smooth_ramp(batch: int, ch: int, h: int, w: int, device) -> torch.Tensor:
    yy = torch.linspace(0, 1, steps=h, device=device).view(1, 1, h, 1)
    xx = torch.linspace(0, 1, steps=w, device=device).view(1, 1, 1, w)
    ramp = 0.5 * yy + 0.5 * xx
    ramp = ramp.expand(batch, ch, h, w).contiguous()
    return ramp


def make_single_dot(batch: int, ch: int, h: int, w: int, device) -> torch.Tensor:
    x = torch.zeros(batch, ch, h, w, device=device)
    x[:, :, h // 2, w // 2] = 1.0
    return x


def print_header(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _dtype_from_str(s: str):
    s = s.lower()
    if s in ("fp16", "float16"):
        return torch.float16
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    return torch.float16


class _nullcontext:
    def __enter__(self): return None
    def __exit__(self, exc_type, exc, tb): return False


def _autocast_ctx(device, amp_dtype: str):
    if device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=_dtype_from_str(amp_dtype))
    return _nullcontext()


def _append_csv(csv_path: str, row: Dict[str, object], header_order: List[str]):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True) if os.path.dirname(csv_path) else None
    is_new = (not os.path.exists(csv_path)) or (os.path.getsize(csv_path) == 0)
    line = ",".join(str(row.get(k, "")) for k in header_order)
    with open(csv_path, "a", encoding="utf-8") as f:
        if is_new:
            f.write(",".join(header_order) + "\n")
        f.write(line + "\n")


@torch.no_grad()
def test_shape_and_residual(model: torch.nn.Module, device, size: int, batch: int, amp_dtype: str):
    print_header("TEST 1) Shape + residual magnitude")
    model.eval()
    x = torch.randn(batch, 3, size, size, device=device)

    with _autocast_ctx(device, amp_dtype):
        y = model(x)

    print(f"Input  shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")
    assert y.shape == x.shape, "Output shape != input shape (for restoration scale=1)."

    stats = residual_stats(y, x)
    print("Residual (y-x) stats:", stats)


def test_gradients(model: torch.nn.Module, device, size: int, batch: int, amp_dtype: str):
    print_header("TEST 2) Gradient flow + NaN/Inf checks")
    model.train()
    x = torch.randn(batch, 3, size, size, device=device)

    model.zero_grad(set_to_none=True)

    if device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler(enabled=True)
        with _autocast_ctx(device, amp_dtype):
            y = model(x)
            loss = y.mean()
        scaler.scale(loss).backward()
        # No optimizer step required; we just want grads
    else:
        y = model(x)
        loss = y.mean()
        loss.backward()

    bad = 0
    total = 0
    none_grad = 0
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is None:
            none_grad += 1
            continue
        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
            print("[BAD GRAD]", n)
            bad += 1

    print(f"Total trainable param tensors: {total}")
    print(f"Params with grad=None:        {none_grad}")
    print(f"Params with NaN/Inf grads:    {bad}")
    assert bad == 0, "Found NaN/Inf gradients."


def test_peak_memory_once(model: torch.nn.Module, device, size: int, batch: int, amp_dtype: str, csv_path: Optional[str] = None):
    print_header(f"TEST 3) Peak GPU memory (size={size}, batch={batch})")
    if device.type != "cuda":
        print("CUDA not available. Skipping peak memory test.")
        return

    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randn(batch, 3, size, size, device=device)
    model.zero_grad(set_to_none=True)

    scaler = torch.cuda.amp.GradScaler(enabled=True)
    start = time.time()
    with _autocast_ctx(device, amp_dtype):
        y = model(x)
        loss = (y ** 2).mean()
    scaler.scale(loss).backward()
    torch.cuda.synchronize()
    elapsed = time.time() - start

    peak_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
    reserved_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)

    print(f"AMP={amp_dtype}  Peak allocated: {peak_gb:.3f} GB | Peak reserved: {reserved_gb:.3f} GB | Time: {elapsed:.2f}s")

    if csv_path:
        row = {
            "mode": "single",
            "size": size,
            "batch": batch,
            "amp": amp_dtype,
            "peak_allocated_gb": round(peak_gb, 4),
            "peak_reserved_gb": round(reserved_gb, 4),
            "time_s": round(elapsed, 3),
            "status": "ok",
        }
        _append_csv(
            csv_path,
            row,
            header_order=["mode", "size", "batch", "amp", "peak_allocated_gb", "peak_reserved_gb", "time_s", "status"],
        )


def test_peak_memory_sweep(model: torch.nn.Module, device, sizes: List[int], batch: int, amp_dtype: str, csv_path: Optional[str] = None):
    print_header("TEST 3b) Peak GPU memory size sweep")
    if device.type != "cuda":
        print("CUDA not available. Skipping memory sweep.")
        return

    for sz in sizes:
        try:
            test_peak_memory_once(model, device, sz, batch, amp_dtype, csv_path=csv_path)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            msg = str(e).splitlines()[-1]
            print(f"[OOM] size={sz}, batch={batch} -> {msg}")
            if csv_path:
                row = {
                    "mode": "size_sweep",
                    "size": sz,
                    "batch": batch,
                    "amp": amp_dtype,
                    "peak_allocated_gb": "",
                    "peak_reserved_gb": "",
                    "time_s": "",
                    "status": "oom",
                }
                _append_csv(
                    csv_path,
                    row,
                    header_order=["mode", "size", "batch", "amp", "peak_allocated_gb", "peak_reserved_gb", "time_s", "status"],
                )
            break


def test_peak_memory_batch_sweep(model: torch.nn.Module, device, size: int, batches: List[int], amp_dtype: str, csv_path: Optional[str] = None):
    print_header("TEST 3c) Peak GPU memory batch sweep (fixed size)")
    if device.type != "cuda":
        print("CUDA not available. Skipping batch sweep.")
        return

    for bs in batches:
        try:
            test_peak_memory_once(model, device, size, bs, amp_dtype, csv_path=csv_path)
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            msg = str(e).splitlines()[-1]
            print(f"[OOM] batch={bs}, size={size} -> {msg}")
            if csv_path:
                row = {
                    "mode": "batch_sweep",
                    "size": size,
                    "batch": bs,
                    "amp": amp_dtype,
                    "peak_allocated_gb": "",
                    "peak_reserved_gb": "",
                    "time_s": "",
                    "status": "oom",
                }
                _append_csv(
                    csv_path,
                    row,
                    header_order=["mode", "size", "batch", "amp", "peak_allocated_gb", "peak_reserved_gb", "time_s", "status"],
                )
            break


@torch.no_grad()
def test_hf_tendency(model: torch.nn.Module, device, size: int, batch: int, amp_dtype: str):
    print_header("TEST 4) Hallucinated texture tendency (HF energy on simple inputs)")
    model.eval()

    tests: List[Tuple[str, torch.Tensor]] = []
    tests.append(("zeros", torch.zeros(batch, 3, size, size, device=device)))
    tests.append(("ones", torch.ones(batch, 3, size, size, device=device)))
    tests.append(("smooth_ramp", make_smooth_ramp(batch, 3, size, size, device)))
    tests.append(("single_dot", make_single_dot(batch, 3, size, size, device)))

    for name, x in tests:
        with _autocast_ctx(device, amp_dtype):
            y = model(x)

        hf_in = hf_energy_laplacian(x)
        hf_out = hf_energy_laplacian(y)
        hf_res = hf_energy_laplacian(y - x)

        print(f"[{name:10s}] HF_in={hf_in:.6f}  HF_out={hf_out:.6f}  HF_res(y-x)={hf_res:.6f}")


def build_model(args, device):
    model = MambaRealSR11(
        inp_channels=3,
        out_channels=3,
        scale=1,                     # fundus restoration
        dim=args.dim,
        num_blocks=args.num_blocks,
        num_refinement_blocks=args.num_refine,
        heads=args.heads,
        ffn_expansion_factor=args.ffn_expansion,
        bias=args.bias,
        LayerNorm_type="WithBias",
    ).to(device)

    if args.ckpt and os.path.isfile(args.ckpt):
        ckpt = torch.load(args.ckpt, map_location="cpu")
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"Loaded ckpt: {args.ckpt}")
        print(f"Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}")
    else:
        if args.ckpt:
            print(f"[Warn] ckpt not found: {args.ckpt}. Using random init.")

    return model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=str, default="auto", help="auto/cuda/cuda:0/cpu")
    p.add_argument("--seed", type=int, default=1234)

    # model config
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--num_blocks", type=int, nargs=4, default=[6, 2, 2, 1])
    p.add_argument("--num_refine", type=int, default=6)
    p.add_argument("--heads", type=int, nargs=4, default=[1, 2, 4, 8])
    p.add_argument("--ffn_expansion", type=float, default=2.66)
    p.add_argument("--bias", action="store_true")
    p.add_argument("--ckpt", type=str, default="", help="optional checkpoint path")

    # tests (single-run mode; if run_all_256 is enabled, these are ignored)
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--amp", type=str, default="fp16", choices=["fp16", "bf16"], help="AMP dtype (cuda only)")
    p.add_argument("--mem_sweep", action="store_true", help="Run size sweep for memory.")
    p.add_argument("--sweep_sizes", type=int, nargs="+", default=[256, 320, 384, 448, 512])

    p.add_argument("--batch_sweep", action="store_true", help="Sweep batch sizes for a fixed patch size.")
    p.add_argument("--sweep_batches", type=int, nargs="+", default=[1, 2, 4, 6, 8])

    p.add_argument("--csv", type=str, default="", help="optional csv output path, e.g. logs/sanity.csv")

    # convenience: run all 256x256 presets (DEFAULT: enabled)
    p.add_argument(
        "--run_all_256",
        action="store_true",
        default=True,
        help="Run the full 256x256 preset suite (DEFAULT enabled). "
             "Writes multiple csv files into --outdir."
    )
    p.add_argument(
        "--no_run_all_256",
        action="store_true",
        help="Disable the default run_all_256 preset suite and use single-run flags instead."
    )
    p.add_argument(
        "--outdir",
        type=str,
        default="logs_256",
        help="Output directory for preset CSV logs when using run_all_256 (default: logs_256)."
    )
    p.add_argument(
        "--all_batches",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 6, 7, 8],
        help="Batch list for the preset batch sweep when using run_all_256."
    )
    p.add_argument("--skip_bf16", action="store_true",
                   help="When using run_all_256, skip the bf16 sweep (useful if bf16 not supported).")
    p.add_argument("--skip_basic", action="store_true",
                   help="When using run_all_256, skip the final basic sanity run.")

    args = p.parse_args()

    # 如果用户显式传了 --no_run_all_256，就关闭默认全套
    if getattr(args, "no_run_all_256", False):
        args.run_all_256 = False

    return args

def run_all_256_presets(args, device):
    """
    Equivalent of run_all_256.sh, but inside Python:
      1) FP16 batch sweep -> {outdir}/sanity_256_fp16.csv
      2) BF16 batch sweep -> {outdir}/sanity_256_bf16.csv (unless --skip_bf16)
      3) Basic sanity (shape/residual + grad + single peak mem + HF)
         -> {outdir}/sanity_256_basic_fp16.csv (unless --skip_basic)

    Notes:
      - Uses size=256 fixed
      - Uses batch=1 for non-sweep tests; sweep uses args.all_batches
    """
    fixed_size = 256
    outdir = args.outdir
    os.makedirs(outdir, exist_ok=True)

    # --- 1) FP16 batch sweep ---
    print_header("RUN_ALL_256) 1/3 FP16 batch sweep")
    model = build_model(args, device)
    test_peak_memory_batch_sweep(
        model,
        device,
        size=fixed_size,
        batches=args.all_batches,
        amp_dtype="fp16",
        csv_path=os.path.join(outdir, "sanity_256_fp16.csv"),
    )

    # --- 2) BF16 batch sweep ---
    if not args.skip_bf16:
        print_header("RUN_ALL_256) 2/3 BF16 batch sweep")
        model = build_model(args, device)
        test_peak_memory_batch_sweep(
            model,
            device,
            size=fixed_size,
            batches=args.all_batches,
            amp_dtype="bf16",
            csv_path=os.path.join(outdir, "sanity_256_bf16.csv"),
        )
    else:
        print_header("RUN_ALL_256) 2/3 BF16 batch sweep (skipped)")

    # --- 3) Basic sanity FP16 ---
    if not args.skip_basic:
        print_header("RUN_ALL_256) 3/3 Basic sanity (FP16, batch=1, size=256)")
        model = build_model(args, device)

        test_shape_and_residual(model, device, fixed_size, batch=1, amp_dtype="fp16")
        test_gradients(model, device, fixed_size, batch=1, amp_dtype="fp16")
        test_peak_memory_once(
            model,
            device,
            size=fixed_size,
            batch=1,
            amp_dtype="fp16",
            csv_path=os.path.join(outdir, "sanity_256_basic_fp16.csv"),
        )
        test_hf_tendency(model, device, fixed_size, batch=1, amp_dtype="fp16")
    else:
        print_header("RUN_ALL_256) 3/3 Basic sanity (skipped)")

    print("\nRUN_ALL_256 presets finished.")
def main():
    args = parse_args()
    set_seed(args.seed)

    device = get_device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA capability:", torch.cuda.get_device_capability(0))

    # 默认：直接 python test_model.py 就会跑全套 256
    if args.run_all_256:
        run_all_256_presets(args, device)
        return

    # 否则：走原本的单项测试模式
    model = build_model(args, device)

    test_shape_and_residual(model, device, args.size, args.batch, args.amp)
    test_gradients(model, device, args.size, args.batch, args.amp)
    test_peak_memory_once(model, device, args.size, args.batch, args.amp, csv_path=args.csv if args.csv else None)

    if args.mem_sweep:
        test_peak_memory_sweep(model, device, args.sweep_sizes, args.batch, args.amp,
                               csv_path=args.csv if args.csv else None)

    if args.batch_sweep:
        test_peak_memory_batch_sweep(model, device, args.size, args.sweep_batches, args.amp,
                                     csv_path=args.csv if args.csv else None)

    test_hf_tendency(model, device, args.size, args.batch, args.amp)

    print("\nAll tests finished.")

if __name__ == "__main__":
    main()
