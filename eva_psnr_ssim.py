import os
import csv
from glob import glob

import cv2
import numpy as np
from tqdm import tqdm

# =========================
# 全局设置（防止OpenCV崩溃）
# =========================
cv2.setNumThreads(0)

# =========================
# 读取图像（强制安全）
# =========================
def read_image(path):   
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"❌ 无法读取图像: {path}")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 强制 uint8
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)

    return img

# =========================
# PSNR（float32安全版）
# =========================
def calculate_psnr(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float("inf")

    return 20 * np.log10(255.0 / np.sqrt(mse))

# =========================
# SSIM（完全稳定版）
# =========================
def calculate_ssim(img1, img2):
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    try:
        mu1 = cv2.GaussianBlur(img1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(img2, (11, 11), 1.5)

        sigma1 = cv2.GaussianBlur(img1 * img1, (11, 11), 1.5) - mu1 * mu1
        sigma2 = cv2.GaussianBlur(img2 * img2, (11, 11), 1.5) - mu2 * mu2
        sigma12 = cv2.GaussianBlur(img1 * img2, (11, 11), 1.5) - mu1 * mu2

        ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1 * mu1 + mu2 * mu2 + C1) *
                    (sigma1 + sigma2 + C2))

        return float(ssim_map.mean())

    except Exception as e:
        print(f"🚨 SSIM异常: {e}")
        return np.nan

# =========================
# 主评估函数
# =========================
def evaluate(gt_dir, pred_dirs, save_csv="result.csv", crop_border=0):
    print("🚀 开始评估（稳定版）")
    print(f"GT: {gt_dir}")
    print(f"Models: {list(pred_dirs.keys())}")
    print("-" * 50)

    gt_paths = sorted(glob(os.path.join(gt_dir, "*")))
    assert len(gt_paths) > 0, "GT为空！"

    results = {k: {"psnr": [], "ssim": []} for k in pred_dirs}
    bad_cases = []

    for gt_path in tqdm(gt_paths, desc="Processing", ncols=100):
        name = os.path.basename(gt_path)

        try:
            gt_img = read_image(gt_path)
        except Exception as e:
            print(e)
            continue

        print(f"\n📌 {name}")

        for model_name, pred_dir in pred_dirs.items():
            pred_path = os.path.join(pred_dir, name)

            if not os.path.exists(pred_path):
                print(f"⚠️ [{model_name}] 缺失")
                results[model_name]["psnr"].append(np.nan)
                results[model_name]["ssim"].append(np.nan)
                bad_cases.append((name, model_name, "missing"))
                continue

            try:
                pred_img = read_image(pred_path)
            except Exception as e:
                print(f"⚠️ [{model_name}] 读取失败")
                results[model_name]["psnr"].append(np.nan)
                results[model_name]["ssim"].append(np.nan)
                bad_cases.append((name, model_name, "read_error"))
                continue

            if gt_img.shape != pred_img.shape:
                print(f"⚠️ [{model_name}] 尺寸不一致")
                results[model_name]["psnr"].append(np.nan)
                results[model_name]["ssim"].append(np.nan)
                bad_cases.append((name, model_name, "shape_mismatch"))
                continue

            # 裁边
            if crop_border > 0:
                gt_eval = gt_img[crop_border:-crop_border, crop_border:-crop_border]
                pred_eval = pred_img[crop_border:-crop_border, crop_border:-crop_border]
            else:
                gt_eval = gt_img
                pred_eval = pred_img

            try:
                psnr = calculate_psnr(gt_eval, pred_eval)
                ssim = calculate_ssim(gt_eval, pred_eval)
            except Exception as e:
                print(f"🚨 [{model_name}] 计算失败")
                psnr, ssim = np.nan, np.nan

            results[model_name]["psnr"].append(psnr)
            results[model_name]["ssim"].append(ssim)

            print(f"   [{model_name}] PSNR: {psnr:.3f} | SSIM: {ssim:.4f}")

            # 🔥 释放内存（关键）
            del pred_img, pred_eval

        del gt_img, gt_eval

    # =========================
    # 汇总
    # =========================
    print("\n" + "=" * 50)
    print("📊 最终结果")
    print("=" * 50)

    final = []

    for model_name, data in results.items():
        psnr_arr = np.array(data["psnr"])
        ssim_arr = np.array(data["ssim"])

        avg_psnr = np.nanmean(psnr_arr)
        std_psnr = np.nanstd(psnr_arr)

        avg_ssim = np.nanmean(ssim_arr)
        std_ssim = np.nanstd(ssim_arr)

        print(f"\n🏆 {model_name}")
        print(f" PSNR: {avg_psnr:.4f} ± {std_psnr:.4f}")
        print(f" SSIM: {avg_ssim:.4f} ± {std_ssim:.4f}")

        final.append([model_name, avg_psnr, std_psnr, avg_ssim, std_ssim])

    # 保存CSV
    with open(save_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Model", "PSNR_mean", "PSNR_std", "SSIM_mean", "SSIM_std"])
        writer.writerows(final)

    print(f"\n💾 保存到 {save_csv}")

    print(f"\n⚠️ 异常样本数: {len(bad_cases)}")
    print("✅ 完成（稳定运行）")

# =========================
# 入口
# =========================
if __name__ == "__main__":
    gt_dir = "/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/gt/"

    pred_dirs = {
        "Baseline(Input)": "/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/input/",
        "Ours": "/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/work_fundus_mamba/infer_out/",
        "LED": "/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/LED",
        "I_SECRET": "/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/I_SECRET",
        "PCENet": "/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/PCENet",
        "ArcNet": "/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/ArcNet/"
    }

    evaluate(gt_dir, pred_dirs, save_csv="result.csv", crop_border=0)