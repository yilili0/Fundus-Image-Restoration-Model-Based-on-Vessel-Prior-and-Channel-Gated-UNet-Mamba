import os
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


def read_tensor(path: str, size):
    """读取并处理成模型需要的 Tensor"""
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


def main(cfg_path: str, input_dir: str, output_dir: str):
    # 1. 加载配置与初始化
    cfg = load_config(cfg_path)
    seeding(int(cfg.get("seed", 42)))

    h = int(cfg["input"]["height"])
    w = int(cfg["input"]["width"])
    size = (h, w)

    create_dir(output_dir)

    checkpoint_path = cfg["paths"]["checkpoint_path"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2. 加载模型
    model = build_unet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    # 3. 获取待推理的图像列表
    valid_extensions = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
    files = sorted(glob(os.path.join(input_dir, "*.*")))
    files = [f for f in files if f.lower().endswith(valid_extensions)]

    if not files:
        print(f"Warning: No images found in {input_dir}")
        return

    # 4. 开始推理
    for path in tqdm(files, total=len(files), desc="Inferencing"):
        name = os.path.splitext(os.path.basename(path))[0]

        # 读取原图获取尺寸，以便后续将掩码还原为原图分辨率
        ori_img = cv2.imread(path, cv2.IMREAD_COLOR)
        if ori_img is None:
            continue
        ori_h, ori_w = ori_img.shape[:2]

        # 准备输入 Tensor
        x = read_tensor(path, size).to(device)

        with torch.no_grad():
            # 推理获得概率图 (取值 0~1)
            pred_tensor = torch.sigmoid(model(x))
            pred_prob = np.squeeze(pred_tensor[0].cpu().numpy(), axis=0)

            # 二值化 (阈值设为 0.5)
            pred_np = (pred_prob > 0.5).astype(np.uint8)

        # 将预测结果放大回原图分辨率 (使用最近邻插值以保持二值属性)
        pred_np_resized = cv2.resize(pred_np, (ori_w, ori_h), interpolation=cv2.INTER_NEAREST)

        # 乘以 255，将 0/1 转换为 0/255，便于作为图像查看和保存
        pred_vis = pred_np_resized * 255

        # 保存推理结果
        save_path = os.path.join(output_dir, f"{name}_mask.png")
        cv2.imwrite(save_path, pred_vis)

    print(f"\n[Success] Inference completed. Masks are saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Standalone Inference Script")
    parser.add_argument("--config", type=str, default="config.yml", help="Path to the config file")
    parser.add_argument("--input_dir", type=str,
                        default="/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/LED/",
                        help="Directory containing input images")
    parser.add_argument("--output_dir", type=str,
                        default="/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/LED_mask",
                        help="Directory to save the predicted masks")
    args = parser.parse_args()

    main(args.config, args.input_dir, args.output_dir)