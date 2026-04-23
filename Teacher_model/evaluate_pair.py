import os
import csv
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from sklearn.metrics import precision_recall_curve, auc
from skimage.morphology import skeletonize


def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ==========================================
# 核心指标计算函数
# ==========================================
def calculate_dice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    pred_flat = pred_bin.flatten()
    gt_flat = gt_bin.flatten()
    intersection = np.sum(pred_flat * gt_flat)
    return float((2. * intersection + 1e-6) / (np.sum(pred_flat) + np.sum(gt_flat) + 1e-6))


def calculate_auc_pr(pred_prob: np.ndarray, gt_bin: np.ndarray) -> float:
    pred_flat = pred_prob.flatten()
    gt_flat = gt_bin.flatten()
    if np.sum(gt_flat) == 0:
        return 0.0
    precision, recall, _ = precision_recall_curve(gt_flat, pred_flat)
    return float(auc(recall, precision))


def calculate_cldice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    pred_bool = pred_bin > 0
    gt_bool = gt_bin > 0
    if np.sum(gt_bool) == 0 or np.sum(pred_bool) == 0:
        return 0.0
    pred_skel = skeletonize(pred_bool)
    gt_skel = skeletonize(gt_bool)

    t_prec_intersection = np.sum(pred_skel * gt_bool)
    t_prec = (t_prec_intersection + 1e-6) / (np.sum(pred_skel) + 1e-6)

    t_sens_intersection = np.sum(gt_skel * pred_bool)
    t_sens = (t_sens_intersection + 1e-6) / (np.sum(gt_skel) + 1e-6)

    return float((2. * t_prec * t_sens + 1e-6) / (t_prec + t_sens + 1e-6))


# ==========================================
# 直接读取分割图
# ==========================================
def read_mask(path: str) -> np.ndarray:
    """读取二值分割图，返回 0/1 mask"""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # 二值化，保证是 0/1
    _, mask = cv2.threshold(image, 127, 1, cv2.THRESH_BINARY)
    return mask.astype(np.uint8)


def read_prob_map(path: str) -> np.ndarray:
    """读取概率图（如果有），否则返回二值 mask"""
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")

    # 归一化到 [0, 1]
    prob = image.astype(np.float32) / 255.0
    return prob


# ==========================================
# 主流程
# ==========================================
def main(cfg_path: str, dir_gt_masks: str, dir_restored_masks: str):
    cfg = load_config(cfg_path)
    results_dir = "gt_input"
    create_dir(results_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    gt_files = sorted(glob(os.path.join(dir_gt_masks, "*.*")))
    gt_files = [f for f in gt_files if f.lower().endswith(valid_extensions)]

    total_dice, total_auc_pr, total_cldice = 0.0, 0.0, 0.0
    valid_count = 0
    csv_details = [["Filename", "Dice", "AUC-PR", "clDice"]]

    for path_gt in tqdm(gt_files, total=len(gt_files), desc="Evaluating"):
        name_with_ext = os.path.basename(path_gt)
        path_restored = os.path.join(dir_restored_masks, name_with_ext)

        if not os.path.exists(path_restored):
            print(f"\n[Warning] 未找到恢复图像对应的分割图：{name_with_ext}，跳过")
            continue

        # 直接读取分割图
        mask_gt = read_mask(path_gt)
        mask_restored = read_mask(path_restored)

        # 尝试读取概率图（如果文件名包含 prob 或 predict）
        prob_name = name_with_ext.replace(".png", "_prob.png").replace(".tif", "_prob.tif")
        prob_path = os.path.join(dir_restored_masks, prob_name)
        if os.path.exists(prob_path):
            prob_restored = read_prob_map(prob_path)
        else:
            # 如果没有概率图，使用二值 mask 作为概率图
            prob_restored = mask_restored.astype(np.float32)

        # 防爆改：如果 GT 是全黑的，打印警告并跳过
        if np.sum(mask_gt) == 0:
            print(f"\n[Warning] {name_with_ext} 的 GT 分割结果全黑，跳过该图。")
            continue

        dice = calculate_dice(mask_restored, mask_gt)
        auc_pr = calculate_auc_pr(prob_restored, mask_gt)
        cldice = calculate_cldice(mask_restored, mask_gt)

        total_dice += dice
        total_auc_pr += auc_pr
        total_cldice += cldice
        valid_count += 1

        csv_details.append([name_with_ext, f"{dice:.4f}", f"{auc_pr:.4f}", f"{cldice:.4f}"])

    if valid_count > 0:
        avg_dice = total_dice / valid_count
        avg_auc_pr = total_auc_pr / valid_count
        avg_cldice = total_cldice / valid_count

        print("\n" + "=" * 60)
        print("Downstream Segmentation Metrics (Direct Mask Evaluation)")
        print("=" * 60)
        print(f"{'Mean Relative Dice ↑':>25}: {avg_dice:.4f}")
        print(f"{'Mean AUC-PR ↑':>25}: {avg_auc_pr:.4f}")
        print(f"{'Mean clDice ↑':>25}: {avg_cldice:.4f}")
        print(f"(Based on {valid_count} valid images)")
        print("=" * 60)

        csv_file_path = os.path.join(results_dir, "downstream_metrics_report_LED.csv")
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["--- Summary ---"])
            writer.writerow(["Mean Dice", "Mean AUC-PR", "Mean clDice"])
            writer.writerow([f"{avg_dice:.4f}", f"{avg_auc_pr:.4f}", f"{avg_cldice:.4f}"])
            writer.writerow([])
            writer.writerow(["--- Detailed Image Metrics ---"])
            writer.writerows(csv_details)

        print(f"\n[Success] Report saved to: {csv_file_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    parser.add_argument("--dir_gt_masks", type=str,
                        default="/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/gt_mask",
                        help="GT 二值分割图目录")
    parser.add_argument("--dir_restored_masks", type=str,
                        default="/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/SCRNet_mask/",
                        help="恢复图像的二值分割图目录")
    args = parser.parse_args()
    main(args.config, args.dir_gt_masks, args.dir_restored_masks)
# python evaluate_pair.py \
#   --config config.yml \
#   --dir_gt_masks "/path/to/gt_masks" \
#   --dir_restored_masks "/path/to/restored_masks"