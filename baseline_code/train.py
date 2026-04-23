
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# 严格限制底层计算库的线程数，防止 CPU 爆炸
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["cv2_NUM_THREADS"] = "0"  # 如果代
import math
import time
import json
import random
import argparse
from dataclasses import asdict
from typing import Dict, Any, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# your dataset.py
from data.dataset import (
    PipelineConfig,
    make_train_val_split,
    FundusPairInfinitePatchDataset,
    FundusPairValFixedSamples,
)

# your loss.py (uploaded)
import losses.loss as loss_mod


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def atomic_save(obj: Dict[str, Any], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def make_dataloader_kwargs(num_workers: int):
    kwargs = dict(
        num_workers=num_workers,
        pin_memory=True if num_workers > 0 else False,
        drop_last=True,
        persistent_workers=True if num_workers > 0 else False,
        prefetch_factor=2 if num_workers > 0 else None,
    )
    return {k: v for k, v in kwargs.items() if v is not None}


def psnr01(pred01: torch.Tensor, gt01: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    mse = torch.mean((pred01 - gt01) ** 2, dim=(1, 2, 3))
    return 10.0 * torch.log10(1.0 / (mse + eps))


@torch.no_grad()
def update_ema(ema_model: nn.Module, model: nn.Module, decay: float):
    msd = model.state_dict()
    esd = ema_model.state_dict()
    for k, v in esd.items():
        if k in msd:
            v.copy_(v * decay + msd[k].detach() * (1.0 - decay))


def first_nonfinite_grad(model: nn.Module) -> Optional[str]:
    """返回第一个出现 non-finite 的梯度参数名；若全部正常则返回 None。"""
    for n, p in model.named_parameters():
        if p is None or p.grad is None:
            continue
        if not torch.isfinite(p.grad).all():
            return n
    return None



def copy_model(model: nn.Module) -> nn.Module:
    import copy
    return copy.deepcopy(model)


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        cfg = {}
    return cfg


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


# -----------------------------
# Stage schedule: control compute (skip teacher in stage1) + smooth enabling
# -----------------------------
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
    """
    Stage1: w_ffl=0, w_vessel=0 -> FFL + teacher 都不会计算（重要：loss.py 只有 w_vessel>0 才会跑 teacher）
    Stage2: ramp 打开 w_ffl & w_vessel（平滑）
    Stage3: full on；vfeat 的 ramp 由 loss.py 内部 schedule(feat_factor) 控制
    """
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
# Validation (quick)
# -----------------------------
@torch.no_grad()
def run_validation(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool, amp_dtype: torch.dtype, steps: int) -> Dict[str, float]:
    model.eval()
    psnr_sum = 0.0
    l1_sum = 0.0
    n = 0

    it = iter(loader)
    for _ in range(steps):
        try:
            xin, xgt, _meta = next(it)
        except StopIteration:
            break
        xin = xin.to(device, non_blocking=True)
        xgt = xgt.to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            pred = model(xin)

        pred01 = pred.float().clamp(0, 1)
        gt01 = xgt.float().clamp(0, 1)

        psnr_sum += float(psnr01(pred01, gt01).mean().item())
        l1_sum += float((pred01 - gt01).abs().mean().item())
        n += 1

    if n == 0:
        return {"val_psnr": float("nan"), "val_l1": float("nan")}
    return {"val_psnr": psnr_sum / n, "val_l1": l1_sum / n}


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/train.yml", help="path to train.yml")
    parser.add_argument("--resume", type=str, default="", help="override resume checkpoint")
    args_cli = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(args_cli.config))
    cfg = load_yaml(args_cli.config)

    # -------- resolve paths --------
    work_dir = resolve_path(cfg["work_dir"], base_dir)
    data_root = resolve_path(cfg["data"]["data_root"], base_dir)

    os.makedirs(work_dir, exist_ok=True)
    ckpt_dir = os.path.join(work_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    log_path = os.path.join(work_dir, "log.jsonl")

    # -------- device & seed --------
    seed = int(cfg.get("seed", 1234))
    seed_everything(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # speed toggles (safe defaults)
    torch.backends.cudnn.benchmark = bool(cfg.get("cudnn_benchmark", True))
    torch.backends.cuda.matmul.allow_tf32 = bool(cfg.get("allow_tf32", True))

    # -------- split --------
    split_dir = os.path.join(work_dir, "split")
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
    val_ds = FundusPairValFixedSamples(
        data_root=data_root,
        cfg=pipe_cfg,
        list_file=val_txt,
        seed=int(cfg["data"].get("val_seed", 2026)),
        num_samples=int(cfg["data"].get("val_num_samples", 2000)),
    )

    # -------- loaders --------
    batch_size = int(cfg["train"]["batch_size"])
    num_workers = int(cfg["train"].get("num_workers", 0))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        **make_dataloader_kwargs(num_workers),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["data"].get("val_batch_size", batch_size)),
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )

    # -------- student model --------
    student_cfg = cfg["model"]["student"].copy()
    student_cfg["model_py"] = resolve_path(student_cfg["model_py"], base_dir)
    model = build_student(student_cfg).to(device)

    # -------- EMA shadow --------
    ema_decay = float(cfg["train"].get("ema_decay", 0.999))
    ema_model = copy_model(model).to(device)
    for p in ema_model.parameters():
        p.requires_grad_(False)
    ema_model.eval()

    # -------- teacher (optional) --------
    teacher = None
    teacher_cfg = cfg["model"].get("teacher", {})
    teacher_enable = bool(teacher_cfg.get("enable", False))
    if teacher_enable:
        teacher_model_py = resolve_path(teacher_cfg["model_py"], base_dir)
        teacher_ckpt = resolve_path(teacher_cfg["ckpt"], base_dir)
        teacher = loss_mod.build_vessel_teacher_from_files(
            model_py_path=teacher_model_py,
            checkpoint_path=teacher_ckpt,
            device=device,
            strict=True,
            in_channels=int(teacher_cfg.get("in_channels", 3)),
            out_channels=int(teacher_cfg.get("out_channels", 1)),
        )

    # -------- loss config --------
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

    # IMPORTANT: align internal ramp with stage boundaries
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
        feature_layers=[s.strip() for s in cfg["loss"]["vessel"].get("feature_layers", "e1.conv,e2.conv,e3.conv").split(",") if s.strip()],
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

    base_w_ffl = float(weights.w_ffl)
    base_w_vessel = float(weights.w_vessel)

    # -------- optimizer / scheduler / AMP --------
    opt_cfg = cfg["optim"]
    lr = float(opt_cfg["lr"])
    weight_decay = float(opt_cfg.get("weight_decay", 0.0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay)

    train_cfg = cfg["train"]
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

    use_amp = bool(train_cfg.get("use_amp", True))
    amp_dtype_str = str(train_cfg.get("amp_dtype", "fp16")).lower()  # 'fp16' or 'bf16'
    use_bf16 = use_amp and amp_dtype_str in ("bf16", "bfloat16")
    autocast_dtype = torch.bfloat16 if use_bf16 else torch.float16
    if use_bf16:
        print("[AMP] using bf16 autocast (GradScaler disabled)")
    else:
        print("[AMP] using fp16 autocast" if use_amp else "[AMP] disabled")
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and (not use_bf16)))

    grad_clip = float(train_cfg.get("grad_clip", 0.5))

    # -------- resume --------
    resume_path = args_cli.resume.strip() or str(train_cfg.get("resume", "")).strip()
    if resume_path:
        resume_path = resolve_path(resume_path, base_dir)

    start_epoch = 0
    global_step = 0
    best_psnr = -1e9
    if resume_path and os.path.exists(resume_path):
        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=True)
        ema_model.load_state_dict(ckpt["ema"], strict=True)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt.get("scaler", scaler.state_dict()))
        start_epoch = int(ckpt.get("epoch", 0))
        global_step = int(ckpt.get("global_step", 0))
        best_psnr = float(ckpt.get("best_psnr", best_psnr))
        print(f"[Resume] epoch={start_epoch} step={global_step} best_psnr={best_psnr:.3f}")

    # -------- logging helper --------
    def log_json(d: Dict[str, Any]):
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    total_steps = epochs * steps_per_epoch
    print(f"[Train] device={device} work_dir={work_dir}")
    print(f"[Train] epochs={epochs} steps/ep={steps_per_epoch} total_steps={total_steps}")
    print(f"[LR] warmup_epochs={warmup_epochs} start_lr={start_lr} lr={lr} min_lr={min_lr} cosine={use_cosine}")
    print(f"[Safe] amp={use_amp} grad_clip={grad_clip} ema_decay={ema_decay}")
    print(f"[Stage] stage1={stage1_pct} stage2={stage2_pct} stage2_ramp={stage2_ramp}")
    print(f"[Teacher] enable={teacher_enable}")

    # -------- training loop --------
    val_every = int(train_cfg.get("val_every", 1))
    val_steps = int(cfg["data"].get("val_steps", max(1, int(cfg["data"].get("val_num_samples", 2000)) // int(cfg["data"].get("val_batch_size", batch_size)))))
    save_every_epoch = bool(train_cfg.get("save_every_epoch", True))
    save_every_n = int(train_cfg.get("save_every_n", 1))  # 新增：每 N 个 epoch 独立保存一次
    try:
        it = iter(train_loader)
        for epoch in range(start_epoch, epochs):
            model.train()
            t0 = time.time()


            for step_in_epoch in range(steps_per_epoch):
                global_step += 1
                try:
                    xin, xgt, _meta = next(it)
                except StopIteration:
                    it = iter(train_loader)
                    xin, xgt, _meta = next(it)

                xin = xin.to(device, non_blocking=True)
                xgt = xgt.to(device, non_blocking=True)

                progress = min(1.0, (global_step - 1) / float(max(1, total_steps - 1)))

                # stage schedule (skip teacher compute in stage1)
                stage_name = apply_stage_schedule(
                    criterion,
                    progress,
                    stage1_pct=stage1_pct,
                    stage2_pct=stage2_pct,
                    stage2_ramp=stage2_ramp,
                    base_w_ffl=base_w_ffl,
                    base_w_vessel=base_w_vessel,
                )

                optimizer.zero_grad(set_to_none=True)

                # student forward in AMP
                with torch.cuda.amp.autocast(enabled=use_amp, dtype=autocast_dtype):
                    pred = model(xin)

                # loss/teacher in FP32 for stability
                pred_f = pred.float() if use_amp else pred
                loss, stats = criterion(pred_f, xgt, progress=progress)

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if grad_clip and grad_clip > 0:
                    if scaler.is_enabled():
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

                bad_grad_param = first_nonfinite_grad(model)
                did_opt_step = True
                if bad_grad_param is not None:
                    did_opt_step = False
                    print(f"[WARN] non-finite grad detected @global_step={global_step} param={bad_grad_param} -> skip optimizer.step()")
                    if scaler.is_enabled():
                        # scaler.step 会自动跳过并在 scaler.update() 里降 scale
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.zero_grad(set_to_none=True)
                else:
                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                if did_opt_step:
                    scheduler.step()
                    update_ema(ema_model, model, decay=ema_decay)

                # log
                log_every = int(train_cfg.get("log_every", 50))
                if global_step % log_every == 0 or step_in_epoch == 0:
                    lr_now = optimizer.param_groups[0]["lr"]
                    log = {
                        "epoch": epoch,
                        "step_in_epoch": step_in_epoch,
                        "global_step": global_step,
                        "progress": float(progress),
                        "stage": stage_name,
                        "lr": float(lr_now),
                        "loss_total": float(loss.item()),
                        "w_ffl": float(criterion.weights.w_ffl),
                        "w_vessel": float(criterion.weights.w_vessel),
                    }
                    for k in ["charb", "msssim", "ffl", "vessel", "vessel_factor", "feat_factor",
                              "v_vseg_bce", "v_vseg_dice", "v_vfeat_raw"]:
                        if k in stats and torch.is_tensor(stats[k]):
                            log[k] = float(stats[k].item())
                    log_json(log)

                    print(
                        f"[Ep {epoch:03d} {step_in_epoch:04d}/{steps_per_epoch}] "
                        f"stage={stage_name} loss={log['loss_total']:.4f} lr={lr_now:.2e} "
                        f"w_ffl={log['w_ffl']:.3g} w_vessel={log['w_vessel']:.3g} "
                        f"vf={log.get('vessel_factor', 0.0):.2f} ff={log.get('feat_factor', 0.0):.2f}"
                    )

            epoch_time = time.time() - t0

            # validate
            if (epoch + 1) % val_every == 0:
                v_raw = run_validation(model, val_loader, device, use_amp=use_amp, amp_dtype=autocast_dtype, steps=val_steps)
                v_ema = run_validation(ema_model, val_loader, device, use_amp=use_amp, amp_dtype=autocast_dtype, steps=val_steps)

                print(
                    f"[Val Ep {epoch:03d}] raw: psnr={v_raw['val_psnr']:.3f} l1={v_raw['val_l1']:.4f} | "
                    f"ema: psnr={v_ema['val_psnr']:.3f} l1={v_ema['val_l1']:.4f} | "
                    f"time={epoch_time:.1f}s"
                )
                log_json({"epoch": epoch, "global_step": global_step, "epoch_time_s": float(epoch_time),
                          "val_raw": v_raw, "val_ema": v_ema})

                # save latest
                latest_path = os.path.join(ckpt_dir, "latest.pth")
                atomic_save({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_psnr": best_psnr,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": cfg,
                }, latest_path)

                # best by EMA psnr
                if v_ema["val_psnr"] > best_psnr:
                    best_psnr = v_ema["val_psnr"]
                    best_path = os.path.join(ckpt_dir, "best_ema_psnr.pth")
                    atomic_save({
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "best_psnr": best_psnr,
                        "model": model.state_dict(),
                        "ema": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": cfg,
                        "val_ema": v_ema,
                    }, best_path)
                    print(f"[Best] saved: {best_path} (best_psnr={best_psnr:.3f})")

            elif save_every_epoch:
                latest_path = os.path.join(ckpt_dir, "latest.pth")
                atomic_save({
                    "epoch": epoch + 1,
                    "global_step": global_step,
                    "best_psnr": best_psnr,
                    "model": model.state_dict(),
                    "ema": ema_model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "config": cfg,
                }, latest_path)
                # --- 新增：每 N 个 epoch 单独保存一份不被覆盖的权重 ---
            if (epoch + 1) % save_every_n == 0:
                epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch + 1}.pth")
                atomic_save({
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "best_psnr": best_psnr,
                        "model": model.state_dict(),
                        "ema": ema_model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "scaler": scaler.state_dict(),
                        "config": cfg,
                    }, epoch_path)
                print(f"[Save] Saved interval checkpoint: {epoch_path}")
    except KeyboardInterrupt:
        print("\n[Interrupt] saving checkpoint...")
        interrupt_path = os.path.join(ckpt_dir, "interrupt.pth")
        atomic_save({
            "epoch": epoch,
            "global_step": global_step,
            "best_psnr": best_psnr,
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "config": cfg,
        }, interrupt_path)
        print(f"[Interrupt] saved: {interrupt_path}")
        raise


if __name__ == "__main__":
    main()