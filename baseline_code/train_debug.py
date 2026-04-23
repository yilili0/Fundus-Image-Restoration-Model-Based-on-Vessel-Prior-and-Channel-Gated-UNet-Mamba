
# -*- coding: utf-8 -*-
"""
train_debug.py (对齐 train.py 的 NaN/Inf 短扫 + 定点复现版)
============================================================

你刚才跑到：
  [done] no non-finite found within max_steps.
这通常说明：在“当前 checkpoint 的 global_step 起点”往后 400 step 内没触发 NaN/Inf。
但你 log.jsonl 显示：
- 第一次 NaN 出现在 global_step=32050（epoch 31 step 49）
- 永久 NaN 从 global_step=34200（epoch 33 step 199）开始

所以如果你 resume 的 global_step 比 32050 小很多（例如 ep30 常见在 ~30000），
max_steps=400 根本还没跑到 32050，自然抓不到。

本脚本在“严格复用 train.py 构建方式”的前提下，新增两个调试能力：
1) --target_gs: 直接跑到某个 global_step（比如 32050/34200），便于精准复现
2) --val_every: 按 train.yml 的 val_fixed_samples 做快速 val（可选用 EMA），因为你日志里 val 可能更早暴露 NaN

推荐用法：
A) 精准复现第一次 NaN（如果你从 ep30 恢复）：
   python train_debug.py --config train.yml --resume /path/to/ep30.pth --target_gs 32100 --max_steps 4000

B) 精准复现永久 NaN 起点：
   python train_debug.py --config train.yml --resume /path/to/ep30.pth --target_gs 34250 --max_steps 6000

C) 同时监控 val（每 200 step 跑 2 个 val batch）：
   python train_debug.py --config train.yml --resume /path/to/ep30.pth --target_gs 34250 --val_every 200 --val_batches 2

D) 判断是否 AMP 数值不稳：
   python train_debug.py --config train.yml --resume /path/to/ep30.pth --target_gs 34250 --amp 0
   python train_debug.py --config train.yml --resume /path/to/ep30.pth --target_gs 34250 --amp 1 --amp_dtype bf16
"""

import os
import math
import time
import argparse
import random
from typing import Dict, Any, Optional, Tuple

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.dataset import (
    PipelineConfig,
    make_train_val_split,
    FundusPairInfinitePatchDataset,
    FundusPairValFixedSamples,
)

import losses.loss as loss_mod


# -----------------------------
# Utils (对齐 train.py)
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataloader_kwargs(num_workers: int):
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return {k: v for k, v in kwargs.items() if v is not None}


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
    spec.loader.exec_module(mod)  # type: ignore
    return mod


def build_student(student_cfg: Dict[str, Any]) -> nn.Module:
    model_py = student_cfg["model_py"]
    mod = dynamic_import(model_py)

    build_fn = student_cfg.get("build_fn", "")
    class_name = student_cfg.get("class_name", "")
    kwargs = student_cfg.get("kwargs", {}) or {}

    if build_fn:
        fn = getattr(mod, build_fn)
        m = fn(**kwargs) if kwargs else fn()
        assert isinstance(m, nn.Module)
        return m
    if class_name:
        cls = getattr(mod, class_name)
        m = cls(**kwargs)
        assert isinstance(m, nn.Module)
        return m
    raise ValueError("student_cfg must have build_fn or class_name")


def make_lr_lambda(
    *,
    steps_per_epoch: int,
    total_epochs: int,
    warmup_epochs: int,
    start_lr: float,
    target_lr: float,
    min_lr: float,
    use_cosine: bool,
):
    total_steps = total_epochs * steps_per_epoch
    warmup_steps = max(1, warmup_epochs * steps_per_epoch)
    start_factor = start_lr / target_lr
    min_factor = min_lr / target_lr

    def lr_lambda(step: int):
        if step < warmup_steps:
            t = step / float(warmup_steps)
            return start_factor + t * (1.0 - start_factor)
        if not use_cosine:
            return 1.0
        t = (step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * t))
        return min_factor + (1.0 - min_factor) * cosine

    return lr_lambda


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def apply_stage_schedule(
    criterion: loss_mod.FundusCompositeLoss,
    progress: float,
    *,
    stage1_pct: float,
    stage2_pct: float,
    stage2_ramp: float,
    base_w_ffl: float,
    base_w_vessel: float,
) -> str:
    progress = clamp01(progress)
    if progress < stage1_pct:
        criterion.weights.w_ffl = 0.0
        criterion.weights.w_vessel = 0.0
        return "S1_pix_struct"
    if progress < stage2_pct:
        p2 = (progress - stage1_pct) / max(1e-8, (stage2_pct - stage1_pct))
        ramp = clamp01(p2 / max(1e-6, stage2_ramp))
        criterion.weights.w_ffl = base_w_ffl * ramp
        criterion.weights.w_vessel = base_w_vessel * ramp
        return "S2_add_freq_vseg"
    criterion.weights.w_ffl = base_w_ffl
    criterion.weights.w_vessel = base_w_vessel
    return "S3_add_vfeat"


# -----------------------------
# Debug helpers
# -----------------------------
def _autocast_ctx(enabled: bool, dtype: str):
    if not enabled:
        return torch.cuda.amp.autocast(enabled=False)
    dt = torch.float16 if dtype.lower() in ["fp16", "float16"] else torch.bfloat16
    try:
        return torch.autocast(device_type="cuda", dtype=dt, enabled=True)
    except Exception:
        return torch.cuda.amp.autocast(enabled=True, dtype=dt)


def is_finite_tensor(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


def tensor_stats(x: torch.Tensor) -> Dict[str, Any]:
    x_det = x.detach()
    finite = torch.isfinite(x_det)
    if finite.any():
        xf = x_det[finite]
        return dict(
            shape=list(x_det.shape),
            dtype=str(x_det.dtype),
            device=str(x_det.device),
            finite_ratio=float(finite.float().mean().item()),
            min=float(xf.min().item()),
            max=float(xf.max().item()),
            mean=float(xf.mean().item()),
            absmax=float(xf.abs().max().item()),
        )
    return dict(shape=list(x_det.shape), dtype=str(x_det.dtype), device=str(x_det.device), finite_ratio=0.0)


def check_model_params_finite(model: nn.Module) -> Tuple[bool, Optional[str]]:
    for n, p in model.named_parameters():
        if p is None:
            continue
        if not torch.isfinite(p.detach()).all():
            return False, n
    return True, None


def check_model_grads_finite(model: nn.Module) -> Tuple[bool, Optional[str]]:
    for n, p in model.named_parameters():
        if p is None or p.grad is None:
            continue
        if not torch.isfinite(p.grad.detach()).all():
            return False, n
    return True, None


def save_debug_dump(dump_dir: str, tag: str, payload: Dict[str, Any]):
    os.makedirs(dump_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    pt_path = os.path.join(dump_dir, f"debug_{tag}_{ts}.pt")
    torch.save(payload, pt_path)
    print(f"[dump] saved -> {pt_path}")

    # optional: png
    try:
        from PIL import Image

        def save_img(t: torch.Tensor, path: str):
            x = t.detach().float().cpu()
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
            x = (x * 255.0).byte().permute(1, 2, 0).numpy()
            Image.fromarray(x).save(path)

        for k in ["xin", "xgt", "pred01"]:
            if k in payload and isinstance(payload[k], torch.Tensor):
                img_path = os.path.join(dump_dir, f"{tag}_{k}_{ts}.png")
                save_img(payload[k][0], img_path)
                print(f"[dump] saved -> {img_path}")
    except Exception:
        pass


def locate_first_bad_module(model: nn.Module, xin: torch.Tensor, amp: bool, amp_dtype: str) -> Optional[str]:
    bad_name: Optional[str] = None
    handles = []

    def hook_factory(name: str):
        def _hook(_m, _inp, out):
            nonlocal bad_name
            if bad_name is not None:
                return
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and (not torch.isfinite(t).all()):
                bad_name = name
        return _hook

    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            handles.append(m.register_forward_hook(hook_factory(name)))

    try:
        with torch.no_grad():
            with _autocast_ctx(amp, amp_dtype):
                _ = model(xin)
    finally:
        for h in handles:
            h.remove()
    return bad_name


@torch.no_grad()
def forward_once(model: nn.Module, xin: torch.Tensor, amp: bool, amp_dtype: str) -> torch.Tensor:
    model.eval()
    with _autocast_ctx(amp, amp_dtype):
        y = model(xin)
    model.train()
    return y


@torch.no_grad()
def ema_update(ema_model: nn.Module, model: nn.Module, decay: float):
    # in-place EMA update (对齐常见写法)
    for ema_p, p in zip(ema_model.parameters(), model.parameters()):
        ema_p.data.mul_(decay).add_(p.data, alpha=(1.0 - decay))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="train.yml")
    p.add_argument("--resume", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/work_fundus_mamba/checkpoints/epoch_30.pth")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--mode", type=str, default="scan", choices=["scan", "forward", "compare_amp"])
    p.add_argument("--max_steps", type=int, default=4000)
    p.add_argument("--target_gs", type=int, default=31800, help="跑到指定 global_step（>=0 生效），用于精准复现")
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--amp_dtype", type=str, default="bp16", choices=["fp16", "bf16"])
    p.add_argument("--dump_dir", type=str, default="./debug_dumps")
    p.add_argument("--do_step", type=int, default=1)
    p.add_argument("--find_module", type=int, default=1)

    # val check
    p.add_argument("--val_every", type=int, default=0, help="每 N step 跑一次 val（0=关闭）")
    p.add_argument("--val_batches", type=int, default=1, help="每次 val 跑多少个 batch（越大越慢）")
    p.add_argument("--val_use_ema", type=int, default=1, help="val 用 EMA 模型（1）还是当前模型（0）")
    args = p.parse_args()

    cfg = load_yaml(args.config)
    base_dir = os.path.dirname(os.path.abspath(args.config))

    # align backend flags with train.py / train.yml
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    if bool(cfg.get("allow_tf32", True)):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    seed = int(cfg.get("seed", 1234))
    seed_everything(seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    use_amp = bool(args.amp)
    print(f"[env] device={device} mode={args.mode} amp={use_amp} dtype={args.amp_dtype} max_steps={args.max_steps}")

    # -------- dirs & splits --------
    work_dir = resolve_path(cfg["work_dir"], base_dir)
    data_root = resolve_path(cfg["data"]["data_root"], base_dir)
    split_dir = os.path.join(work_dir, "splits")
    os.makedirs(split_dir, exist_ok=True)

    train_txt = os.path.join(split_dir, "train.txt")
    val_txt = os.path.join(split_dir, "val.txt")
    if not (os.path.exists(train_txt) and os.path.exists(val_txt)):
        val_ratio = float(cfg["data"].get("val_ratio", 20 / 120))
        make_train_val_split(data_root, split_dir, val_ratio=val_ratio, seed=seed)

    # -------- datasets --------
    patch_size = int(cfg["data"]["patch_size"])
    pipe_cfg = PipelineConfig(patch_size=patch_size)

    train_ds = FundusPairInfinitePatchDataset(
        data_root=data_root,
        cfg=pipe_cfg,
        seed=seed,
        list_file=train_txt,
        augment=bool(cfg["data"].get("augment", True)),
    )
    train_cfg = cfg["train"]
    batch_size = int(train_cfg["batch_size"])
    num_workers = int(train_cfg.get("num_workers", 0))

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,      # IterableDataset：不要 shuffle
        **make_dataloader_kwargs(num_workers),
    )
    train_iter = iter(train_loader)

    # val loader（可选）
    val_loader = None
    if args.val_every and args.val_every > 0:
        val_ds = FundusPairValFixedSamples(
            data_root=data_root,
            cfg=pipe_cfg,
            seed=int(cfg["data"].get("val_seed", seed + 17)),
            list_file=val_txt,
            num_samples=int(cfg["data"].get("val_num_samples", 256)),
        )
        val_bs = int(cfg["data"].get("val_batch_size", batch_size))
        val_workers = min(num_workers, 2)  # debug 时别太大，避免拖慢
        val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, **make_dataloader_kwargs(val_workers))

    # -------- model / ema --------
    student_cfg = cfg["model"]["student"].copy()
    student_cfg["model_py"] = resolve_path(student_cfg["model_py"], base_dir)
    model = build_student(student_cfg).to(device)

    import copy
    ema_model = copy.deepcopy(model).to(device)
    ema_model.eval()
    for p_ in ema_model.parameters():
        p_.requires_grad_(False)

    ema_decay = float(train_cfg.get("ema_decay", 0.999))

    # -------- teacher --------
    teacher_cfg = cfg["model"].get("teacher", {})
    teacher_enable = bool(teacher_cfg.get("enable", False))
    teacher = None
    if teacher_enable:
        teacher = loss_mod.build_vessel_teacher_from_files(
            model_py_path=resolve_path(teacher_cfg["model_py"], base_dir),
            checkpoint_path=resolve_path(teacher_cfg["ckpt"], base_dir),
            device=device,
            strict=True,
            in_channels=int(teacher_cfg.get("in_channels", 3)),
            out_channels=int(teacher_cfg.get("out_channels", 1)),
        )

    # -------- loss --------
    loss_weights_cfg = cfg["loss"]["weights"]
    weights = loss_mod.CompositeWeights(
        w_charb=float(loss_weights_cfg.get("w_charb", 1.0)),
        w_msssim=float(loss_weights_cfg.get("w_msssim", 0.1)),
        w_ffl=float(loss_weights_cfg.get("w_ffl", 0.05)),
        w_vessel=float(loss_weights_cfg.get("w_vessel", 0.02 if teacher_enable else 0.0)),
    )

    stage_cfg = cfg["loss"]["stages"]
    stage1_pct = float(stage_cfg.get("stage1_pct", 0.20))
    stage2_pct = float(stage_cfg.get("stage2_pct", 0.80))
    stage2_ramp = float(stage_cfg.get("stage2_ramp", 0.10))

    schedule = loss_mod.RampSchedule(
        vessel_start=stage1_pct,
        vessel_full=stage2_pct,
        feat_start=stage2_pct,
        feat_full=1.0,
    )

    vessel_cfg = loss_mod.VesselTeacherConfig(
        mean=None,
        std=None,
        mask_gamma=float(cfg["loss"]["vessel"].get("mask_gamma", 2.0)),
        hard_thresh=None,
        feature_layers=[s.strip() for s in str(cfg["loss"]["vessel"].get("feature_layers", "")).split(",") if s.strip()],
        force_teacher_fp32=bool(cfg["loss"]["vessel"].get("force_teacher_fp32", True)),
    )

    criterion = loss_mod.FundusCompositeLoss(
        assume_range=str(cfg["loss"].get("assume_range", "0_1")),
        weights=weights,
        schedule=schedule,
        use_roi_mask_for_pixel=bool(cfg["loss"].get("use_roi_mask_for_pixel", False)),
        roi_black_thr_255=int(cfg["loss"].get("roi_black_thr_255", 10)),
        vessel_teacher=teacher,
        vessel_cfg=vessel_cfg,
        vessel_w_bce=float(cfg["loss"]["vessel"].get("w_bce", 1.0)),
        vessel_w_dice=float(cfg["loss"]["vessel"].get("w_dice", 1.0)),
        vessel_w_feat=float(cfg["loss"]["vessel"].get("w_feat", 0.2)),
    ).to(device)

    # -------- optimizer / scheduler / scaler --------
    opt_cfg = cfg["optim"]
    lr = float(opt_cfg.get("lr", 2e-4))
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    epochs = int(train_cfg["epochs"])
    steps_per_epoch = int(train_cfg["steps_per_epoch"])
    warmup_epochs = int(opt_cfg.get("warmup_epochs", 5))
    start_lr = float(opt_cfg.get("start_lr", 1e-6))
    min_lr = float(opt_cfg.get("min_lr", 1e-6))
    use_cosine = bool(opt_cfg.get("use_cosine", True))

    lr_lambda = make_lr_lambda(
        steps_per_epoch=steps_per_epoch,
        total_epochs=epochs,
        warmup_epochs=warmup_epochs,
        start_lr=start_lr,
        target_lr=lr,
        min_lr=min_lr,
        use_cosine=use_cosine,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    grad_clip = float(train_cfg.get("grad_clip", 0.5))

    # -------- resume --------
    resume_path = resolve_path(args.resume, base_dir)
    global_step = 0
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        if "ema" in ckpt:
            ema_model.load_state_dict(ckpt["ema"], strict=True)
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        if "scheduler" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler"])
        if "scaler" in ckpt and ckpt["scaler"] is not None:
            scaler.load_state_dict(ckpt["scaler"])
        global_step = int(ckpt.get("global_step", 0))
        print(f"[Resume] global_step={global_step}")

    # -------- loop --------
    model.train()
    total_steps = epochs * steps_per_epoch
    target_gs = args.target_gs if args.target_gs >= 0 else None

    for dbg_step in range(args.max_steps):
        if target_gs is not None and global_step >= target_gs:
            print(f"[hit] reached target_gs={target_gs} at dbg_step={dbg_step}")
            break

        try:
            xin, xgt, meta = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            xin, xgt, meta = next(train_iter)

        xin = xin.to(device, non_blocking=True)
        xgt = xgt.to(device, non_blocking=True)

        # input
        if not is_finite_tensor(xin) or not is_finite_tensor(xgt):
            print(f"[BAD] input non-finite @dbg_step={dbg_step} gs={global_step}")
            print("  xin:", tensor_stats(xin))
            print("  xgt:", tensor_stats(xgt))
            save_debug_dump(args.dump_dir, "bad_input", {"xin": xin.cpu(), "xgt": xgt.cpu(), "meta": meta, "global_step": global_step})
            return

        progress = min(1.0, (max(global_step, 1) - 1) / max(total_steps - 1, 1))
        stage = apply_stage_schedule(
            criterion,
            float(progress),
            stage1_pct=stage1_pct,
            stage2_pct=stage2_pct,
            stage2_ramp=stage2_ramp,
            base_w_ffl=float(loss_weights_cfg.get("w_ffl", 0.05)),
            base_w_vessel=float(loss_weights_cfg.get("w_vessel", 0.0)),
        )

        if args.mode == "compare_amp":
            pred_amp = forward_once(model, xin, amp=use_amp, amp_dtype=args.amp_dtype)
            pred_fp32 = forward_once(model, xin, amp=False, amp_dtype=args.amp_dtype)
            print(f"[compare] pred_amp_finite={is_finite_tensor(pred_amp)} pred_fp32_finite={is_finite_tensor(pred_fp32)}")
            print("  amp :", tensor_stats(pred_amp))
            print("  fp32:", tensor_stats(pred_fp32))
            return

        optimizer.zero_grad(set_to_none=True)

        # forward
        with _autocast_ctx(use_amp, args.amp_dtype):
            pred = model(xin)

        if not is_finite_tensor(pred):
            bad_mod = locate_first_bad_module(model, xin, amp=use_amp, amp_dtype=args.amp_dtype) if bool(args.find_module) else None
            pred_fp32 = forward_once(model, xin, amp=False, amp_dtype=args.amp_dtype)
            print(f"[BAD] pred non-finite @dbg_step={dbg_step} gs={global_step} progress={progress:.4f} stage={stage}")
            print(f"  first_bad_module: {bad_mod}")
            print("  pred_amp:", tensor_stats(pred))
            print("  pred_fp32 finite?", is_finite_tensor(pred_fp32), "stats:", tensor_stats(pred_fp32))
            save_debug_dump(
                args.dump_dir,
                "bad_pred",
                {
                    "xin": xin.cpu(),
                    "xgt": xgt.cpu(),
                    "pred": pred.detach().cpu(),
                    "pred_fp32": pred_fp32.detach().cpu(),
                    "meta": meta,
                    "global_step": global_step,
                    "progress": float(progress),
                    "stage": stage,
                    "first_bad_module": bad_mod,
                },
            )
            return

        # loss
        pred_f = pred.float() if use_amp else pred
        loss, stats = criterion(pred_f, xgt, progress=float(progress))

        if not torch.isfinite(loss):
            print(f"[BAD] loss non-finite @dbg_step={dbg_step} gs={global_step} progress={progress:.4f} stage={stage}")
            flat = {k: (float(v.detach().cpu().item()) if torch.is_tensor(v) and v.numel() == 1 else None) for k, v in stats.items()}
            print("  stats:", flat)
            save_debug_dump(
                args.dump_dir,
                "bad_loss",
                {
                    "xin": xin.cpu(),
                    "xgt": xgt.cpu(),
                    "pred": pred.detach().cpu(),
                    "pred01": torch.nan_to_num(pred_f.detach().cpu()).clamp(0, 1),
                    "loss": loss.detach().cpu(),
                    "stats": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in stats.items()},
                    "meta": meta,
                    "global_step": global_step,
                    "progress": float(progress),
                    "stage": stage,
                },
            )
            return

        if args.mode == "forward":
            if dbg_step % args.log_every == 0:
                print(f"[OK-forward] dbg_step={dbg_step} gs={global_step} stage={stage} loss={float(loss.item()):.6f}")
            global_step += 1
            continue

        # backward + clip + grad check
        scaler.scale(loss).backward()
        if grad_clip and grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        okg, badg = check_model_grads_finite(model)
        if not okg:
            print(f"[BAD] grad non-finite @dbg_step={dbg_step} gs={global_step} bad_param={badg} stage={stage}")
            save_debug_dump(
                args.dump_dir,
                "bad_grad",
                {
                    "xin": xin.cpu(),
                    "xgt": xgt.cpu(),
                    "pred": pred.detach().cpu(),
                    "loss": loss.detach().cpu(),
                    "stats": {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in stats.items()},
                    "bad_grad_param": badg,
                    "meta": meta,
                    "global_step": global_step,
                    "progress": float(progress),
                    "stage": stage,
                },
            )
            return

        # step + param check
        if bool(args.do_step):
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            okp, badp = check_model_params_finite(model)
            if not okp:
                print(f"[BAD] PARAM non-finite AFTER STEP @dbg_step={dbg_step} gs={global_step} bad_param={badp}")
                save_debug_dump(args.dump_dir, "bad_param", {"bad_param": badp, "global_step": global_step, "progress": float(progress), "stage": stage})
                return
        else:
            scaler.update()

        # EMA update（对齐训练）
        ema_update(ema_model, model, decay=ema_decay)

        # optional quick val
        if val_loader is not None and args.val_every > 0 and (dbg_step % args.val_every == 0):
            vmodel = ema_model if bool(args.val_use_ema) else model
            vmodel.eval()
            with torch.no_grad():
                vb = 0
                for vxin, vxgt, vmeta in val_loader:
                    vxin = vxin.to(device, non_blocking=True)
                    with _autocast_ctx(use_amp, args.amp_dtype):
                        vpred = vmodel(vxin)
                    if not is_finite_tensor(vpred):
                        tag = "bad_val_pred"
                        print(f"[BAD] {tag} @dbg_step={dbg_step} gs={global_step} (val)")
                        save_debug_dump(args.dump_dir, tag, {"vxin": vxin.detach().cpu(), "vpred": vpred.detach().cpu(), "global_step": global_step})
                        return
                    vb += 1
                    if vb >= int(args.val_batches):
                        break
            vmodel.train()

        if dbg_step % args.log_every == 0:
            print(f"[OK] dbg_step={dbg_step} gs={global_step} progress={progress:.4f} stage={stage} loss={float(loss.item()):.6f}")

        global_step += 1

    print("[done] no non-finite found within max_steps / before target_gs.")


if __name__ == "__main__":
    main()
