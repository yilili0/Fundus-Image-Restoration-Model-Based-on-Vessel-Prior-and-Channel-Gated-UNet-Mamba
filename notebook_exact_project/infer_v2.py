import os
import csv
from glob import glob

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm

from model import build_unet


def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def seeding(seed: int):
    import random
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def mask_parse(mask: np.ndarray):
    mask = np.expand_dims(mask, axis=-1)  # (H, W, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (H, W, 3)
    return mask


def read_tensor(path: str, size):
    """只负责读取并处理成模型需要的 Tensor，不返回用于可视化的图"""
    h, w = size
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    if (image.shape[0], image.shape[1]) != (h, w):
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR)
    x = np.transpose(image, (2, 0, 1))
    x = x / 255.0
    x = np.expand_dims(x, axis=0).astype(np.float32)
    x = torch.from_numpy(x)
    return x


def calculate_entropy(prob_map: np.ndarray) -> float:
    """
    计算概率图的信息熵 (二元交叉熵形式)。
    熵越低，说明模型对像素是前景还是背景的判断越果断、越确信。
    """
    epsilon = 1e-7
    # 限制概率在极小值和极大值之间，防止 log(0)
    p = np.clip(prob_map, epsilon, 1.0 - epsilon)
    entropy = - (p * np.log(p) + (1 - p) * np.log(1 - p))
    return float(np.mean(entropy))


def calculate_connected_components(binary_map: np.ndarray) -> int:
    """
    计算二值化图中的连通域数量。
    对于眼底血管来说，数量越少，说明血管断裂越少、连通性越好，或者是噪点被消除了。
    """
    binary_map_uint8 = (binary_map * 255).astype(np.uint8)
    num_labels, _ = cv2.connectedComponents(binary_map_uint8)
    # 减去 1 是为了排除背景(背景也算一个连通域)
    return max(0, num_labels - 1)


def main(cfg_path: str, dir_before: str, dir_after: str):
    cfg = load_config(cfg_path)
    seeding(int(cfg["seed"]))

    # 模型的输入尺寸
    h = int(cfg["input"]["height"])
    w = int(cfg["input"]["width"])
    size = (h, w)

    # Folders
    results_dir = cfg["paths"].get("results_dir", "results_visual_comparison")
    create_dir(results_dir)

    checkpoint_path = cfg["paths"]["checkpoint_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 获取优化前文件夹中的所有图像
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    before_files = sorted(glob(os.path.join(dir_before, "*.*")))
    before_files = [f for f in before_files if f.lower().endswith(valid_extensions)]

    if not before_files:
        print(f"Warning: No images found in {dir_before}")
        return

    # 初始化指标累计
    total_entropy_before = 0.0
    total_entropy_after = 0.0
    total_cc_before = 0.0
    total_cc_after = 0.0
    valid_count = 0

    for path_before in tqdm(before_files, total=len(before_files), desc="Inferencing & Comparing"):
        name_with_ext = os.path.basename(path_before)
        name = os.path.splitext(name_with_ext)[0]

        path_after = os.path.join(dir_after, name_with_ext)

        if not os.path.exists(path_after):
            print(f"\nSkipping {name_with_ext}: Missing in 'after' directory.")
            continue

        valid_count += 1

        # 1. 读取用于最终拼接的高清原图
        ori_img_before = cv2.imread(path_before, cv2.IMREAD_COLOR)
        ori_img_after = cv2.imread(path_after, cv2.IMREAD_COLOR)
        ori_h, ori_w = ori_img_before.shape[:2]

        # 2. 读取用于模型推理的 Tensor
        x_before = read_tensor(path_before, size).to(device)
        x_after = read_tensor(path_after, size).to(device)

        with torch.no_grad():
            # 推理获得概率图 (取值 0~1)
            pred_before_tensor = torch.sigmoid(model(x_before))
            pred_after_tensor = torch.sigmoid(model(x_after))

            pred_prob_before = np.squeeze(pred_before_tensor[0].cpu().numpy(), axis=0)
            pred_prob_after = np.squeeze(pred_after_tensor[0].cpu().numpy(), axis=0)

            # 计算定量指标 1：信息熵 (不确定性)
            total_entropy_before += calculate_entropy(pred_prob_before)
            total_entropy_after += calculate_entropy(pred_prob_after)

            # 二值化
            pred_np_before = (pred_prob_before > 0.5).astype(np.uint8)
            pred_np_after = (pred_prob_after > 0.5).astype(np.uint8)

            # 计算定量指标 2：连通域数量 (血管断裂度/噪点)
            total_cc_before += calculate_connected_components(pred_np_before)
            total_cc_after += calculate_connected_components(pred_np_after)

        # 3. 将预测结果放大回原图分辨率用于拼接
        pred_np_before_resized = cv2.resize(pred_np_before, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)
        pred_np_after_resized = cv2.resize(pred_np_after, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

        # 可视化准备
        pred_before_vis = mask_parse(pred_np_before_resized) * 255
        pred_after_vis = mask_parse(pred_np_after_resized) * 255

        # 分割线
        line = np.ones((ori_h, 15, 3), dtype=np.uint8) * 128

        # 拼接长图
        cat_images = np.concatenate([
            ori_img_before, line,
            pred_before_vis, line,
            pred_after_vis, line,
            ori_img_after
        ], axis=1)

        cv2.imwrite(os.path.join(results_dir, f"{name}_comparison_highres.png"), cat_images)

    # ==========================
    # 打印对比报告并保存 CSV
    # ==========================
    if valid_count > 0:
        avg_entropy_before = total_entropy_before / valid_count
        avg_entropy_after = total_entropy_after / valid_count
        avg_cc_before = total_cc_before / valid_count
        avg_cc_after = total_cc_after / valid_count

        print("\n" + "=" * 60)
        print("No-Reference Metrics Comparison (Before vs After)")
        print("=" * 60)

        # 终端打印
        print(
            f"{'Mean Entropy (Uncertainty) ↓':>30}: {avg_entropy_before:.4f}  ->  {avg_entropy_after:.4f}  ({avg_entropy_after - avg_entropy_before:+.4f})")
        print(
            f"{'Avg Connected Components ↓':>30}: {avg_cc_before:.1f}  ->  {avg_cc_after:.1f}  ({avg_cc_after - avg_cc_before:+.1f})")
        print("=" * 60)
        print("注：↓ 表示该指标越低越好 (熵值低代表置信度高，连通域少代表血管连续且噪点少)。")

        # 保存 CSV
        csv_data = [
            ["Metric", "Description", "Before", "After", "Difference"],
            ["Mean Entropy", "Lower means higher model confidence", f"{avg_entropy_before:.4f}",
             f"{avg_entropy_after:.4f}", f"{avg_entropy_after - avg_entropy_before:+.4f}"],
            ["Connected Components", "Lower means better vessel connectivity / less noise", f"{avg_cc_before:.1f}",
             f"{avg_cc_after:.1f}", f"{avg_cc_after - avg_cc_before:+.1f}"]
        ]

        csv_file_path = os.path.join(results_dir, "metrics_nomask_comparison.csv")
        with open(csv_file_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(csv_data)

        print(f"\n[Success] High-res images and CSV report saved to: {results_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="High-Res Visual Comparison & No-Mask Metrics")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the config file")
    parser.add_argument("--dir_before", type=str,
                        default="/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/input/",
                        help="Path to images BEFORE optimization")
    # parser.add_argument("--dir_after", type=str,
    #                     default="/shared_data/users/yili/Hybrid-Mamba-UNet/baseline_code/work_fundus_mamba/infer_out/",
    #                     help="Path to images AFTER optimization")
    parser.add_argument("--dir_after", type=str,
                        default="/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/gt/",
                        help="Path to images AFTER optimization")
    args = parser.parse_args()

    main(args.config, args.dir_before, args.dir_after)