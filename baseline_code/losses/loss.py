
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Basic helpers
# -------------------------
def set_requires_grad(model: nn.Module, flag: bool) -> None:
    for p in model.parameters():
        p.requires_grad_(flag)


def to_01(x: torch.Tensor, assume_range: str = "0_1") -> torch.Tensor:
    """
    Convert tensor to [0,1].
    assume_range: "0_1" | "-1_1" | "auto"
    """
    if assume_range == "0_1":
        return x
    if assume_range == "-1_1":
        return (x + 1.0) * 0.5
    if assume_range == "auto":
        with torch.no_grad():
            xmin = float(x.min())
            xmax = float(x.max())
        if xmin < -0.2 and xmax <= 1.2:
            return (x + 1.0) * 0.5
        return x
    raise ValueError(f"Unknown assume_range: {assume_range}")


def _safe_get_logits(out: Union[torch.Tensor, Dict, List, Tuple]) -> torch.Tensor:
    """
    Teacher forward in your code returns Tensor logits directly.
    This function also supports tuple/list/dict returns (robust).
    """
    if torch.is_tensor(out):
        return out
    if isinstance(out, (tuple, list)):
        return _safe_get_logits(out[0])
    if isinstance(out, dict):
        for k in ["logits", "out", "pred"]:
            if k in out:
                return _safe_get_logits(out[k])
    raise TypeError("Unsupported teacher output type.")


# -------------------------
# Loss components
# -------------------------
class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        if weight is None:
            return loss.mean()
        return (loss * weight).sum() / weight.sum().clamp_min(1.0)


class DiceLoss(nn.Module):
    """Soft Dice loss for prob maps in [0,1], with optional spatial weight mask."""
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor, weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        if weight is None:
            weight = torch.ones_like(pred)

        pred = pred * weight
        target = target * weight

        inter = (pred * target).sum(dim=(2, 3))
        denom = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2.0 * inter + self.eps) / (denom + self.eps)
        return (1.0 - dice).mean()


class MSSSIMLoss(nn.Module):
    """
    1 - MS-SSIM. Prefer pytorch-msssim; fallback to kornia ssim; else weak fallback.
    """
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred01: torch.Tensor, gt01: torch.Tensor) -> torch.Tensor:
        x = pred01.float()
        y = gt01.float()
        try:
            from pytorch_msssim import ms_ssim
            val = ms_ssim(x, y, data_range=self.data_range, size_average=True)
            return 1.0 - val
        except Exception:
            try:
                import kornia
                val = kornia.metrics.ssim(x, y, window_size=11, max_val=self.data_range).mean()
                return 1.0 - val
            except Exception:
                blur_x = F.avg_pool2d(x, 11, stride=1, padding=5)
                blur_y = F.avg_pool2d(y, 11, stride=1, padding=5)
                return (x - y).abs().mean() + 0.2 * (blur_x - blur_y).abs().mean()


def hann2d(h: int, w: int, device, dtype) -> torch.Tensor:
    wh = torch.hann_window(h, periodic=True, device=device, dtype=dtype)
    ww = torch.hann_window(w, periodic=True, device=device, dtype=dtype)
    return torch.outer(wh, ww).view(1, 1, h, w)


def radial_mask(h: int, w: int, low: float, high: float, device, dtype) -> torch.Tensor:
    yy = torch.linspace(-1.0, 1.0, steps=h, device=device, dtype=dtype).view(h, 1)
    xx = torch.linspace(-1.0, 1.0, steps=w, device=device, dtype=dtype).view(1, w)
    rr = torch.sqrt(xx * xx + yy * yy)
    rr = rr / rr.max().clamp_min(1e-6)
    m = ((rr >= low) & (rr <= high)).to(dtype=dtype)
    return m.view(1, 1, h, w)


class FocalFrequencyLoss(nn.Module):
    """
    FFL-style:
      - Hann window to reduce patch boundary spectral leakage
      - optional mid-high band focus for vessel edges
      - adaptive focusing weights (clamped)
    """
    def __init__(
        self,
        alpha: float = 1.0,
        clamp_weight: float = 10.0,
        use_hann: bool = True,
        use_band: bool = True,
        band_low: float = 0.25,
        band_high: float = 1.0,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.alpha = alpha
        self.clamp_weight = clamp_weight
        self.use_hann = use_hann
        self.use_band = use_band
        self.band_low = band_low
        self.band_high = band_high
        self.eps = eps

    def forward(self, pred01: torch.Tensor, gt01: torch.Tensor) -> torch.Tensor:
        x = pred01.float()
        y = gt01.float()
        b, c, h, w = x.shape
        device = x.device

        if self.use_hann:
            win = hann2d(h, w, device, x.dtype)
            x = x * win
            y = y * win

        fx = torch.fft.fft2(x, norm="ortho")
        fy = torch.fft.fft2(y, norm="ortho")
        diff = torch.abs(fx - fy)

        if self.use_band:
            m = radial_mask(h, w, self.band_low, self.band_high, device, diff.dtype)
            diff = diff * m

        mean_diff = diff.mean(dim=(2, 3), keepdim=True).clamp_min(self.eps)
        wmap = (diff / mean_diff).pow(self.alpha).clamp(max=self.clamp_weight)
        loss = (wmap * diff).mean()
        return loss.to(pred01.dtype)


# -------------------------
# Teacher feature hooks
# -------------------------
class FeatureHookBank:
    def __init__(self):
        self.outputs: Dict[str, torch.Tensor] = {}
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

    def clear(self):
        self.outputs.clear()

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles.clear()

    def register(self, model: nn.Module, layer_names: List[str]):
        name_to_module = dict(model.named_modules())
        missing = [n for n in layer_names if n not in name_to_module]
        if missing:
            raise ValueError(
                f"Teacher missing layers: {missing}\n"
                f"Tip: print([n for n,_ in teacher.named_modules()]) and choose valid names."
            )
        for n in layer_names:
            m = name_to_module[n]
            self.handles.append(m.register_forward_hook(self._make_hook(n)))

    def _make_hook(self, name: str):
        def hook(module, inp, out):
            if torch.is_tensor(out):
                self.outputs[name] = out
        return hook


# -------------------------
# Vessel teacher loss (adapted to your model.py)
# -------------------------
@dataclass
class VesselTeacherConfig:
    # Your predict_one preprocess is /255 only -> mean/std should be None unless you *know* teacher training used normalize.
    mean: Optional[List[float]] = None
    std: Optional[List[float]] = None

    # confidence mask: M = sigmoid(logits_gt)^gamma
    mask_gamma: float = 2.0
    hard_thresh: Optional[float] = None  # usually None (soft mask safer)

    # Feature layers for your UNet:
    # model.py contains e1/e2/e3/e4 (encoder_block) with .conv, and bottleneck b. :contentReference[oaicite:7]{index=7}
    feature_layers: Optional[List[str]] = None  # default: ["e1.conv","e2.conv","e3.conv"]

    # numeric stability under AMP
    force_teacher_fp32: bool = True


class VesselTeacherGuidance(nn.Module):
    """
    Teacher-guided vessel constraints (teacher outputs logits).

    HARD POINT #1 satisfied:
      - pred branch teacher forward is NOT inside torch.no_grad(),
        so gradients flow to pred image -> student network.
      - teacher parameters are frozen (requires_grad False).
    """
    def __init__(
        self,
        teacher: nn.Module,
        cfg: VesselTeacherConfig = VesselTeacherConfig(),
        w_bce: float = 1.0,
        w_dice: float = 1.0,
        w_feat: float = 0.2,   # keep small to reduce hallucination
        eps: float = 1e-6,
    ):
        super().__init__()
        self.teacher = teacher.eval()
        set_requires_grad(self.teacher, False)

        self.cfg = cfg
        if self.cfg.feature_layers is None:
            self.cfg.feature_layers = ["e1.conv", "e2.conv", "e3.conv"]

        self.w_bce = w_bce
        self.w_dice = w_dice
        self.w_feat = w_feat
        self.eps = eps

        self.dice = DiceLoss(eps=eps)
        self.hooks = FeatureHookBank()
        if self.cfg.feature_layers:
            self.hooks.register(self.teacher, self.cfg.feature_layers)

    def _norm(self, x01: torch.Tensor) -> torch.Tensor:
        if self.cfg.mean is None or self.cfg.std is None:
            return x01
        mean = torch.tensor(self.cfg.mean, device=x01.device, dtype=x01.dtype).view(1, -1, 1, 1)
        std = torch.tensor(self.cfg.std, device=x01.device, dtype=x01.dtype).view(1, -1, 1, 1)
        return (x01 - mean) / (std + 1e-12)

    def _teacher_forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.cfg.force_teacher_fp32:
            x = x.float()
        out = self.teacher(x)
        return _safe_get_logits(out)

    def _mask_from_prob(self, p_gt: torch.Tensor) -> torch.Tensor:
        m = p_gt.clamp(0, 1).pow(self.cfg.mask_gamma)
        if self.cfg.hard_thresh is not None:
            m = m * (p_gt >= self.cfg.hard_thresh).to(m.dtype)
        return m

    def forward(self, pred01: torch.Tensor, gt01: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        pred01/gt01: [B,3,H,W] in [0,1]
        """
        items: Dict[str, torch.Tensor] = {}

        # Teacher input domain: matches your predict_one preprocess (/255 only). :contentReference[oaicite:8]{index=8}
        pred_in = self._norm(pred01)
        gt_in = self._norm(gt01.detach())

        # ---- pred branch (WITH grad) ----
        self.hooks.clear()
        logits_pred = self._teacher_forward(pred_in)
        feats_pred = dict(self.hooks.outputs)

        # ---- gt branch (no grad) ----
        with torch.no_grad():
            self.hooks.clear()
            logits_gt = self._teacher_forward(gt_in)
            feats_gt = dict(self.hooks.outputs)

        if logits_pred.dim() == 3:
            logits_pred = logits_pred.unsqueeze(1)
        if logits_gt.dim() == 3:
            logits_gt = logits_gt.unsqueeze(1)

        p_gt = torch.sigmoid(logits_gt)          # detached target
        p_pred = torch.sigmoid(logits_pred)

        m = self._mask_from_prob(p_gt)           # [B,1,H,W]

        # vseg: masked BCEWithLogits + masked Dice
        bce_map = F.binary_cross_entropy_with_logits(logits_pred, p_gt, reduction="none")
        v_bce = (bce_map * m).sum() / (m.sum().clamp_min(self.eps))
        v_dice = self.dice(p_pred, p_gt, weight=m)

        items["vseg_bce"] = v_bce
        items["vseg_dice"] = v_dice
        vseg = self.w_bce * v_bce + self.w_dice * v_dice
        items["vseg_total"] = vseg

        # vfeat: masked L1 feature matching on shallow/mid layers
        vfeat_raw = pred01.new_tensor(0.0)
        if self.cfg.feature_layers:
            for name in self.cfg.feature_layers:
                fp = feats_pred.get(name, None)
                fg = feats_gt.get(name, None)
                if fp is None or fg is None or fp.dim() != 4:
                    continue
                m_l = F.interpolate(m, size=fp.shape[-2:], mode="bilinear", align_corners=False)
                vfeat_raw = vfeat_raw + ((fp - fg).abs() * m_l).sum() / (m_l.sum().clamp_min(self.eps))

        items["vfeat_raw"] = vfeat_raw
        vfeat = self.w_feat * vfeat_raw
        items["vfeat_total"] = vfeat

        total = vseg + vfeat
        items["vessel_total"] = total
        return total.to(pred01.dtype), items


# -------------------------
# Composite + ramp schedule
# -------------------------
@dataclass
class CompositeWeights:
    w_charb: float = 1.0
    w_msssim: float = 0.1
    w_ffl: float = 0.05
    w_vessel: float = 0.02


@dataclass
class RampSchedule:
    """
    progress in [0,1]:
      - vessel loss starts later
      - feature matching starts latest (implemented by scaling teacher.w_feat)
    """
    vessel_start: float = 0.20
    vessel_full: float = 0.80
    feat_start: float = 0.80
    feat_full: float = 1.00

    @staticmethod
    def _ramp(progress: float, start: float, full: float) -> float:
        if progress <= start:
            return 0.0
        if progress >= full:
            return 1.0
        return float((progress - start) / max(full - start, 1e-8))

    def vessel_factor(self, progress: float) -> float:
        return self._ramp(progress, self.vessel_start, self.vessel_full)

    def feat_factor(self, progress: float) -> float:
        return self._ramp(progress, self.feat_start, self.feat_full)


class FundusCompositeLoss(nn.Module):
    """
    Main entry:
      loss, stats = criterion(pred, gt, progress=progress)
    """
    def __init__(
        self,
        *,
        assume_range: str = "0_1",
        weights: CompositeWeights = CompositeWeights(),
        schedule: RampSchedule = RampSchedule(),
        # Optional: ROI mask for pixel loss (fundus black background). Default off.
        use_roi_mask_for_pixel: bool = False,
        roi_black_thr_255: int = 10,

        vessel_teacher: Optional[nn.Module] = None,
        vessel_cfg: VesselTeacherConfig = VesselTeacherConfig(),
        vessel_w_bce: float = 1.0,
        vessel_w_dice: float = 1.0,
        vessel_w_feat: float = 0.2,
    ):
        super().__init__()
        self.assume_range = assume_range
        self.weights = weights
        self.schedule = schedule

        self.use_roi_mask_for_pixel = use_roi_mask_for_pixel
        self.roi_black_thr = roi_black_thr_255 / 255.0

        self.charb = CharbonnierLoss(eps=1e-3)
        self.msssim = MSSSIMLoss(data_range=1.0)
        self.ffl = FocalFrequencyLoss(alpha=1.0, clamp_weight=10.0, use_hann=True, use_band=True, band_low=0.25, band_high=1.0)

        self.vessel = None
        if vessel_teacher is not None:
            self.vessel = VesselTeacherGuidance(
                teacher=vessel_teacher,
                cfg=vessel_cfg,
                w_bce=vessel_w_bce,
                w_dice=vessel_w_dice,
                w_feat=vessel_w_feat,
            )

    def _roi_mask(self, gt01: torch.Tensor) -> torch.Tensor:
        # gt01: [B,3,H,W]
        thr = self.roi_black_thr
        black = (gt01[:, 0:1] < thr) & (gt01[:, 1:2] < thr) & (gt01[:, 2:3] < thr)
        roi = (~black).to(gt01.dtype)  # [B,1,H,W]
        return roi

    def forward(
        self,
        pred: torch.Tensor,
        gt: torch.Tensor,
        *,
        progress: Optional[float] = None,  # float in [0,1], recommended
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        pred01 = to_01(pred, self.assume_range).clamp(0, 1)
        gt01 = to_01(gt, self.assume_range).clamp(0, 1)

        stats: Dict[str, torch.Tensor] = {}

        # Optional ROI for pixel fidelity
        roi = None
        if self.use_roi_mask_for_pixel:
            roi = self._roi_mask(gt01)

        l_charb = self.charb(pred01, gt01, weight=roi)
        l_msssim = self.msssim(pred01, gt01) if self.weights.w_msssim > 0 else pred01.new_tensor(0.0)
        l_ffl = self.ffl(pred01, gt01) if self.weights.w_ffl > 0 else pred01.new_tensor(0.0)

        stats["charb"] = l_charb
        stats["msssim"] = l_msssim
        stats["ffl"] = l_ffl

        # Vessel loss with ramp
        l_vessel = pred01.new_tensor(0.0)
        vessel_factor = 1.0
        feat_factor = 1.0
        if progress is not None:
            vessel_factor = self.schedule.vessel_factor(progress)
            feat_factor = self.schedule.feat_factor(progress)

        stats["vessel_factor"] = pred01.new_tensor(float(vessel_factor))
        stats["feat_factor"] = pred01.new_tensor(float(feat_factor))

        if self.vessel is not None and self.weights.w_vessel > 0:
            # scale feature matching only at late stage
            orig_w_feat = self.vessel.w_feat
            self.vessel.w_feat = orig_w_feat * feat_factor

            l_vessel, vdict = self.vessel(pred01, gt01)
            for k, v in vdict.items():
                stats[f"v_{k}"] = v

            self.vessel.w_feat = orig_w_feat

        stats["vessel"] = l_vessel

        total = (
            self.weights.w_charb * l_charb
            + self.weights.w_msssim * l_msssim
            + self.weights.w_ffl * l_ffl
            + (self.weights.w_vessel * vessel_factor) * l_vessel
        )
        stats["total"] = total
        return total, stats


# -------------------------
# Optional: robust teacher loader (handles signature mismatch)
# -------------------------
def build_vessel_teacher_from_files(
    *,
    model_py_path: str,
    checkpoint_path: str,
    device: torch.device,
    strict: bool = True,
    in_channels: int = 3,
    out_channels: int = 1,
) -> nn.Module:
    """
    HARD POINT #2 satisfied:
      - tries build_unet(in_channels=..., out_channels=...) first
      - falls back to build_unet() if TypeError (matches your uploaded model.py) :contentReference[oaicite:9]{index=9}
    """
    import importlib.util
    import os

    if not os.path.exists(model_py_path):
        raise FileNotFoundError(f"model.py not found: {model_py_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")

    spec = importlib.util.spec_from_file_location("vessel_teacher_mod", model_py_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    build_unet = getattr(mod, "build_unet")

    try:
        teacher = build_unet(in_channels=in_channels, out_channels=out_channels)
    except TypeError:
        teacher = build_unet()

    state = torch.load(checkpoint_path, map_location="cpu")
    state_dict = state["model"] if isinstance(state, dict) and "model" in state else state
    teacher.load_state_dict(state_dict, strict=strict)

    teacher.eval()
    set_requires_grad(teacher, False)
    teacher.to(device)
    return teacher
