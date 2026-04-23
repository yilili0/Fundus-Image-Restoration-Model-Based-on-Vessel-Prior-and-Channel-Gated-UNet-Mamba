"""
fundus_pipeline.py

目标：
1) 2560x2560 原图 -> 在线随机 256x256 patch（Input/GT 成对同坐标裁剪）
2) 黑背景拒绝采样（黑像素比例 > 40% 丢弃，最多 10 次，fallback 返回“最不黑”的一次）
3) 同步几何增强（train 用：hflip/vflip/rot90；val 默认关闭增强）
4) 原图级别划分 train/val（默认 100/20≈83/17），避免 patch 泄漏
5) 提供 3 个测试/验收函数（你之前已跑通）

使用方式（不需要命令行参数）：
- 只需要修改 main() 顶部的硬编码配置
"""

import os
import glob
import math
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
from PIL import Image

import torch
from torch.utils.data import IterableDataset, DataLoader, get_worker_info


# ----------------------------
# Utils
# ----------------------------
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def is_image_file(p: str) -> bool:
    return p.lower().endswith(IMG_EXTS)


def list_images(folder: str) -> List[str]:
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(folder, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(folder, f"*{ext.upper()}")))
    paths = sorted(list(set(paths)))
    return paths


def pil_to_np_rgb(img: Image.Image) -> np.ndarray:
    """Return uint8 RGB HxWx3"""
    if img.mode != "RGB":
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)


def np_to_torch_float01(arr_rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 -> 3xHxW float32 in [0,1]"""
    t = torch.from_numpy(arr_rgb).permute(2, 0, 1).contiguous()
    return t.float().div(255.0)


def black_ratio_rgb(arr_rgb: np.ndarray, thr: int = 10) -> float:
    """
    黑像素定义：RGB 三通道都 < thr 才算黑（更符合“背景黑圈”）。
    返回比例 [0,1]。
    """
    assert arr_rgb.ndim == 3 and arr_rgb.shape[2] == 3
    mask = (arr_rgb[:, :, 0] < thr) & (arr_rgb[:, :, 1] < thr) & (arr_rgb[:, :, 2] < thr)
    return float(mask.mean())


def apply_sync_aug(
    in_rgb: np.ndarray,
    gt_rgb: np.ndarray,
    rng: random.Random,
    p_hflip: float = 0.5,
    p_vflip: float = 0.5,
    p_rot90: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    同步几何增强（必须对齐）：
      - hflip (p=0.5)
      - vflip (p=0.5)
      - rot90 (p=0.5): 若触发，从 {0,1,2,3} 选 k 表示 90*k 旋转
    返回增强后的 in/gt 以及增强参数记录（便于 debug/验收）
    """
    assert in_rgb.shape == gt_rgb.shape, "Input/GT shape mismatch before aug"
    params = {"hflip": False, "vflip": False, "rot90": 0}

    if rng.random() < p_hflip:
        in_rgb = np.ascontiguousarray(in_rgb[:, ::-1, :])
        gt_rgb = np.ascontiguousarray(gt_rgb[:, ::-1, :])
        params["hflip"] = True

    if rng.random() < p_vflip:
        in_rgb = np.ascontiguousarray(in_rgb[::-1, :, :])
        gt_rgb = np.ascontiguousarray(gt_rgb[::-1, :, :])
        params["vflip"] = True

    if rng.random() < p_rot90:
        k = rng.randint(0, 3)  # 0/1/2/3 -> 0/90/180/270
        if k != 0:
            in_rgb = np.ascontiguousarray(np.rot90(in_rgb, k=k, axes=(0, 1)))
            gt_rgb = np.ascontiguousarray(np.rot90(gt_rgb, k=k, axes=(0, 1)))
        params["rot90"] = k

    assert in_rgb.shape == gt_rgb.shape, "Input/GT shape mismatch after aug"
    return in_rgb, gt_rgb, params


# ----------------------------
# Split
# ----------------------------
def make_train_val_split(
    data_root: str,
    out_dir: str,
    val_ratio: float = 20 / 120,  # 默认 100/20
    seed: int = 1234,
) -> Tuple[str, str]:
    """
    基于文件名（basename）做原图级别划分，确保 input/gt 严格配对。
    输出两个txt：train.txt / val.txt，每行一个 basename，例如 0001.png
    """
    input_dir = os.path.join(data_root, "input")
    gt_dir = os.path.join(data_root, "gt")
    os.makedirs(out_dir, exist_ok=True)

    input_paths = [p for p in list_images(input_dir) if is_image_file(p)]
    gt_paths = [p for p in list_images(gt_dir) if is_image_file(p)]
    if len(input_paths) == 0:
        raise FileNotFoundError(f"No input images found in {input_dir}")
    if len(gt_paths) == 0:
        raise FileNotFoundError(f"No gt images found in {gt_dir}")

    gt_map = {os.path.basename(p): p for p in gt_paths}

    names = []
    missing = []
    for ip in input_paths:
        name = os.path.basename(ip)
        if name not in gt_map:
            missing.append(name)
        else:
            names.append(name)

    if missing:
        raise FileNotFoundError(f"Missing GT files for some inputs. Example: {missing[:5]}")

    names = sorted(list(set(names)))
    if len(names) < 2:
        raise RuntimeError("Not enough paired images to split.")

    rng = random.Random(seed)
    rng.shuffle(names)

    n_total = len(names)
    n_val = max(1, int(round(n_total * val_ratio)))
    n_train = n_total - n_val
    if n_train < 1:
        raise RuntimeError("Split produced empty train set. Reduce val_ratio.")

    val_names = sorted(names[:n_val])
    train_names = sorted(names[n_val:])

    train_txt = os.path.join(out_dir, "train.txt")
    val_txt = os.path.join(out_dir, "val.txt")

    with open(train_txt, "w", encoding="utf-8") as f:
        for n in train_names:
            f.write(n + "\n")

    with open(val_txt, "w", encoding="utf-8") as f:
        for n in val_names:
            f.write(n + "\n")

    print(f"[Split] total={n_total} train={len(train_names)} val={len(val_names)}")
    print(f"[Split] saved: {train_txt}")
    print(f"[Split] saved: {val_txt}")
    return train_txt, val_txt


# ----------------------------
# Dataset Config
# ----------------------------
@dataclass
class PipelineConfig:
    patch_size: int = 256
    black_thr: int = 10
    black_ratio_max: float = 0.40
    max_tries: int = 10
    force_resize_to_2560: bool = False  # 若数据并非严格2560，可开启强制resize


# ----------------------------
# Datasets
# ----------------------------
class FundusPairInfinitePatchDataset(IterableDataset):
    """
    无限样本流（IterableDataset）：
      每次随机选一对 (input, gt) 原图，然后在其中随机切 256x256 patch，
      进行黑背景拒绝采样，并做同步几何增强（可开关）。
    """

    def __init__(
        self,
        data_root: str,
        cfg: PipelineConfig,
        seed: int = 1234,
        list_file: Optional[str] = None,  # train.txt / val.txt
        augment: bool = True,
        aug_p_hflip: float = 0.5,
        aug_p_vflip: float = 0.5,
        aug_p_rot90: float = 0.5,
    ):
        super().__init__()
        self.data_root = data_root
        self.cfg = cfg
        self.seed = int(seed)

        self.augment = bool(augment)
        self.aug_p_hflip = float(aug_p_hflip)
        self.aug_p_vflip = float(aug_p_vflip)
        self.aug_p_rot90 = float(aug_p_rot90)

        self.input_dir = os.path.join(data_root, "input")
        self.gt_dir = os.path.join(data_root, "gt")
        if not os.path.isdir(self.input_dir) or not os.path.isdir(self.gt_dir):
            raise FileNotFoundError(
                f"Expected folders:\n  {self.input_dir}\n  {self.gt_dir}\n"
                f"Please organize as data_root/input and data_root/gt."
            )

        self.input_paths = list_images(self.input_dir)
        if len(self.input_paths) == 0:
            raise FileNotFoundError(f"No images found in {self.input_dir}")

        allowed = None
        if list_file is not None:
            with open(list_file, "r", encoding="utf-8") as f:
                allowed = set(line.strip() for line in f if line.strip())
            if len(allowed) == 0:
                raise RuntimeError(f"Empty list_file: {list_file}")

        gt_paths = list_images(self.gt_dir)
        gt_map = {os.path.basename(p): p for p in gt_paths}

        pairs = []
        missing = []
        for ip in self.input_paths:
            name = os.path.basename(ip)
            if allowed is not None and name not in allowed:
                continue
            if name not in gt_map:
                missing.append(name)
            else:
                pairs.append((ip, gt_map[name]))

        if missing:
            raise FileNotFoundError(
                f"Missing GT files for {len(missing)} inputs. Example: {missing[:5]}\n"
                f"Input and GT filenames must match exactly."
            )

        self.pairs = pairs
        if len(self.pairs) < 1:
            raise RuntimeError("No valid input/gt pairs found for this split.")

    def _load_pair_rgb(self, input_path: str, gt_path: str) -> Tuple[np.ndarray, np.ndarray]:
        in_img = Image.open(input_path)
        gt_img = Image.open(gt_path)

        if self.cfg.force_resize_to_2560:
            in_img = in_img.convert("RGB").resize((2560, 2560), Image.BICUBIC)
            gt_img = gt_img.convert("RGB").resize((2560, 2560), Image.BICUBIC)

        in_rgb = pil_to_np_rgb(in_img)
        gt_rgb = pil_to_np_rgb(gt_img)

        if in_rgb.shape != gt_rgb.shape:
            raise ValueError(
                f"Input/GT size mismatch:\n  input={input_path} {in_rgb.shape}\n  gt={gt_path} {gt_rgb.shape}"
            )
        if in_rgb.shape[0] < self.cfg.patch_size or in_rgb.shape[1] < self.cfg.patch_size:
            raise ValueError(
                f"Image smaller than patch_size={self.cfg.patch_size}:\n  {input_path} shape={in_rgb.shape}"
            )
        return in_rgb, gt_rgb

    def _sample_patch(self, in_rgb: np.ndarray, gt_rgb: np.ndarray, rng: random.Random):
        H, W, _ = in_rgb.shape
        ps = self.cfg.patch_size

        # 记录最优（最少黑背景）的候选，避免10次都失败导致返回垃圾patch
        best = None
        best_ratio = 1.0
        best_xy = (0, 0)

        for _ in range(self.cfg.max_tries):
            x = rng.randint(0, W - ps)
            y = rng.randint(0, H - ps)

            in_patch = in_rgb[y:y + ps, x:x + ps, :]
            gt_patch = gt_rgb[y:y + ps, x:x + ps, :]

            br = black_ratio_rgb(in_patch, thr=self.cfg.black_thr)

            if br < best_ratio:
                best_ratio = br
                best = (in_patch, gt_patch)
                best_xy = (x, y)

            if br <= self.cfg.black_ratio_max:
                return in_patch, gt_patch, best_xy, br, False  # not fallback

        # fallback：返回“10次里最不黑”的那个
        assert best is not None
        in_patch, gt_patch = best
        return in_patch, gt_patch, best_xy, best_ratio, True

    def __iter__(self):
        worker = get_worker_info()
        if worker is None:
            worker_id = 0
        else:
            worker_id = worker.id

        # 每个 worker 独立 RNG，避免重复
        rng = random.Random(self.seed + 1000 * worker_id)

        while True:
            ip, gp = self.pairs[rng.randint(0, len(self.pairs) - 1)]
            in_rgb, gt_rgb = self._load_pair_rgb(ip, gp)

            # 成对切 patch（同坐标） + 背景拒绝采样
            in_patch, gt_patch, (x, y), br, fallback = self._sample_patch(in_rgb, gt_rgb, rng)

            # 同步增强（val 默认关闭）
            if self.augment:
                in_patch, gt_patch, aug_params = apply_sync_aug(
                    in_patch, gt_patch, rng,
                    p_hflip=self.aug_p_hflip,
                    p_vflip=self.aug_p_vflip,
                    p_rot90=self.aug_p_rot90,
                )
            else:
                aug_params = {"hflip": False, "vflip": False, "rot90": 0}

            in_t = np_to_torch_float01(in_patch)
            gt_t = np_to_torch_float01(gt_patch)

            assert in_t.shape == gt_t.shape == (3, self.cfg.patch_size, self.cfg.patch_size)

            meta = {
                "input_path": ip,
                "gt_path": gp,
                "xy": (int(x), int(y)),
                "black_ratio": float(br),
                "fallback": bool(fallback),
                "aug": aug_params,
            }
            yield in_t, gt_t, meta


class FundusPairValFixedSamples(IterableDataset):
    """
    Val 推荐用“固定数量样本 + 关闭增强 + 固定 seed”：
    - 解决无限流 val 指标波动问题
    - 每次评估可复现（审稿更喜欢）
    """

    def __init__(
        self,
        data_root: str,
        cfg: PipelineConfig,
        list_file: str,
        seed: int = 2026,
        num_samples: int = 2000,  # 每次验证评估多少个patch（你可按算力/时间调）
    ):
        super().__init__()
        self.inner = FundusPairInfinitePatchDataset(
            data_root=data_root,
            cfg=cfg,
            seed=seed,
            list_file=list_file,
            augment=False,  # val 关闭增强
        )
        self.num_samples = int(num_samples)

    def __iter__(self):
        it = iter(self.inner)
        for _ in range(self.num_samples):
            yield next(it)


# ----------------------------
# Visualization / Acceptance
# ----------------------------
def save_pair_image(in_t: torch.Tensor, gt_t: torch.Tensor, out_path: str):
    """保存一对 (input, gt) 为左右拼接图，便于肉眼验收对齐。"""
    in_np = (in_t.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    gt_np = (gt_t.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
    pair = np.concatenate([in_np, gt_np], axis=1)
    Image.fromarray(pair).save(out_path)


def save_batch_grid(batch_in: torch.Tensor, batch_gt: torch.Tensor, out_path: str, max_n: int = 16):
    """保存一个 batch 的网格图（每个样本左右拼接），最多 max_n 个样本。"""
    B = min(batch_in.shape[0], max_n)
    pairs = []
    for i in range(B):
        in_np = (batch_in[i].clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        gt_np = (batch_gt[i].clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
        pairs.append(np.concatenate([in_np, gt_np], axis=1))

    cols = 4
    rows = int(math.ceil(B / cols))
    H, W2, _ = pairs[0].shape
    grid = np.zeros((rows * H, cols * W2, 3), dtype=np.uint8)

    for idx, im in enumerate(pairs):
        r = idx // cols
        c = idx % cols
        grid[r * H:(r + 1) * H, c * W2:(c + 1) * W2, :] = im

    Image.fromarray(grid).save(out_path)


def acceptance_check_100_batches(
    dataset: IterableDataset,
    out_dir: str,
    batch_size: int,
    num_workers: int,
    cfg: PipelineConfig,
):
    """
    验收标准 1：
      连续读取 100 个 batch，保存部分 batch 网格图，并检查“极端黑图”比例。
    """
    os.makedirs(out_dir, exist_ok=True)
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(num_workers > 0))

    bad90 = 0
    bad99 = 0
    total = 0
    fallback_cnt = 0

    for bi, (xin, xgt, meta) in enumerate(dl):
        thr = cfg.black_thr / 255.0
        black_mask = (xin[:, 0] < thr) & (xin[:, 1] < thr) & (xin[:, 2] < thr)
        br = black_mask.float().mean(dim=(1, 2))

        bad90 += int((br > 0.90).sum().item())
        bad99 += int((br > 0.99).sum().item())
        total += int(br.numel())

        fb = meta["fallback"]
        if torch.is_tensor(fb):
            fallback_cnt += int(fb.sum().item())
        else:
            fallback_cnt += sum(bool(x) for x in fb)

        if bi < 20:
            save_batch_grid(xin, xgt, os.path.join(out_dir, f"batch_{bi:03d}.png"), max_n=16)

        if bi + 1 >= 100:
            break

    print("=== Acceptance Check: 100 batches ===")
    print(f"Total samples checked: {total}")
    print(f">90% black patches:   {bad90}")
    print(f">99% black patches:   {bad99}")
    print(f"Fallback used count:  {fallback_cnt}")
    print(f"Saved batch grids to: {out_dir}")

    assert bad99 == 0, f"Found {bad99} patches with >99% black (near full-black)!"
    assert bad90 == 0, f"Found {bad90} patches with >90% black!"
    print("PASS: No 'all black' or '90% black' patches detected in 100 batches.")


def visualize_10_pairs_alignment(dataset: IterableDataset, out_dir: str):
    """验收标准 2：可视化 10 对样本，确认 Input/GT 对齐。"""
    os.makedirs(out_dir, exist_ok=True)
    dl = DataLoader(dataset, batch_size=1, num_workers=0)

    for i, (xin, xgt, meta) in enumerate(dl):
        save_pair_image(xin[0], xgt[0], os.path.join(out_dir, f"pair_{i:02d}.png"))

        m = {k: meta[k][0] if isinstance(meta[k], (list, tuple)) else meta[k] for k in meta}
        print(f"[{i:02d}] black_ratio={m.get('black_ratio')} fallback={m.get('fallback')} xy={m.get('xy')} aug={m.get('aug')}")

        if i + 1 >= 10:
            break

    print(f"Saved 10 aligned pairs to: {out_dir}")


def demo_loader(dataset: IterableDataset, batch_size: int = 8, num_workers: int = 0, steps: int = 3):
    dl = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, pin_memory=(num_workers > 0))
    for i, (xin, xgt, meta) in enumerate(dl):
        # 取第一个样本的 aug 信息（兼容 dict-of-list 的情况）
        aug0 = meta.get("aug", None)
        if isinstance(aug0, dict):
            aug0 = {k: (v[0].item() if torch.is_tensor(v) else v[0]) for k, v in aug0.items()}
        else:
            # 如果你的环境里碰巧变成 list[dict]，也能兼容
            aug0 = aug0[0] if isinstance(aug0, (list, tuple)) and len(aug0) > 0 else aug0

        br0 = meta["black_ratio"][0].item() if torch.is_tensor(meta["black_ratio"]) else meta["black_ratio"][0]
        fb0 = meta["fallback"][0].item() if torch.is_tensor(meta["fallback"]) else meta["fallback"][0]

        print(f"batch {i}: xin={tuple(xin.shape)} xgt={tuple(xgt.shape)} "
              f"br(ex0)={br0} fallback(ex0)={fb0} aug(ex0)={aug0}")

        if i + 1 >= steps:
            break



# ----------------------------
# Main (Hard-coded config only)
# ----------------------------
def main():
    # ===============================
    # ✅ 只改这里：你的硬编码配置
    # ===============================
    DATA_ROOT = "C:\\Users\\29638\\Desktop\\Real_Fundus\\Real_Fundus"  # 需要包含: DATA_ROOT/input 和 DATA_ROOT/gt
    OUT_DIR = "C:\\Users\\29638\\Desktop\\Real_Fundus\\Real_Fundus"

    # split 比例（专家建议：100/20 = 0.1667）
    VAL_RATIO = 20 / 120
    SPLIT_SEED = 1234

    # patch/pipeline
    PATCH_SIZE = 256
    BLACK_THR = 10
    BLACK_RATIO_MAX = 0.40
    MAX_TRIES = 10
    FORCE_RESIZE_TO_2560 = False  # 若你的图不是严格2560*2560，可开 True（input/gt 都会 resize）

    # train loader
    TRAIN_SEED = 1234
    TRAIN_BATCH_SIZE = 8
    TRAIN_NUM_WORKERS = 0

    # val（固定样本数，推荐不增强）
    VAL_SEED = 2026
    VAL_NUM_SAMPLES = 2000
    VAL_BATCH_SIZE = 8
    VAL_NUM_WORKERS = 0

    # 是否运行测试（测试顺序demo_loader → vis10 → check100）
    RUN_TEST_CHECK100 = True
    RUN_TEST_VIS10 = False
    RUN_TEST_DEMO = False

    # ===============================
    # ✅ 不用改下面
    # ===============================
    os.makedirs(OUT_DIR, exist_ok=True)
    split_dir = os.path.join(OUT_DIR, "splits")
    train_txt, val_txt = make_train_val_split(
        data_root=DATA_ROOT,
        out_dir=split_dir,
        val_ratio=VAL_RATIO,
        seed=SPLIT_SEED,
    )

    cfg = PipelineConfig(
        patch_size=PATCH_SIZE,
        black_thr=BLACK_THR,
        black_ratio_max=BLACK_RATIO_MAX,
        max_tries=MAX_TRIES,
        force_resize_to_2560=FORCE_RESIZE_TO_2560,
    )

    # Train：无限流 + 增强
    train_ds = FundusPairInfinitePatchDataset(
        data_root=DATA_ROOT,
        cfg=cfg,
        seed=TRAIN_SEED,
        list_file=train_txt,
        augment=True,
        aug_p_hflip=0.5,
        aug_p_vflip=0.5,
        aug_p_rot90=0.5,
    )

    # Val：固定数量样本 + 关闭增强（稳定可复现）
    val_ds = FundusPairValFixedSamples(
        data_root=DATA_ROOT,
        cfg=cfg,
        list_file=val_txt,
        seed=VAL_SEED,
        num_samples=VAL_NUM_SAMPLES,
    )

    print(f"[Info] Train pairs: {len(train_ds.pairs)}")
    print(f"[Info] Val pairs:   {len(val_ds.inner.pairs)}")
    print(f"[Info] Val samples per eval: {VAL_NUM_SAMPLES}")

    # 训练时就用这两个 dataloader：
    train_loader = DataLoader(
        train_ds,
        batch_size=TRAIN_BATCH_SIZE,
        num_workers=TRAIN_NUM_WORKERS,
        pin_memory=(TRAIN_NUM_WORKERS > 0),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=VAL_BATCH_SIZE,
        num_workers=VAL_NUM_WORKERS,
        pin_memory=(VAL_NUM_WORKERS > 0),
    )

    # （可选）跑测试
    if RUN_TEST_CHECK100:
        acceptance_check_100_batches(
            dataset=train_ds,  # 用 train_ds 测黑背景更严格（因为有增强也应通过）
            out_dir=os.path.join(OUT_DIR, "check100_batches_train"),
            batch_size=TRAIN_BATCH_SIZE,
            num_workers=TRAIN_NUM_WORKERS,
            cfg=cfg,
        )

    if RUN_TEST_VIS10:
        visualize_10_pairs_alignment(
            dataset=train_ds,
            out_dir=os.path.join(OUT_DIR, "vis10_pairs_train"),
        )

    if RUN_TEST_DEMO:
        demo_loader(train_ds, batch_size=TRAIN_BATCH_SIZE, num_workers=TRAIN_NUM_WORKERS, steps=3)

    # 如果你只是把这个文件作为“模块”import 给训练脚本用，
    # 训练脚本里直接用 train_loader / val_loader 的构建方式即可。
    #
    # 这里给一个最小提示（不真的训练）：
    print("[Ready] Data pipeline OK. Use train_loader/val_loader in your training script.")


# if __name__ == "__main__":
#     main()
