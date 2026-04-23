import os

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
# 严格限制底层计算库的线程数，防止 CPU 爆炸
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["cv2_NUM_THREADS"] = "0"  
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from tqdm import tqdm


from models.model import MambaRealSR11


# =========================================================================
# 核心推理逻辑：滑动窗口无缝拼接
# =========================================================================
@torch.no_grad()
def overlap_tile_inference(model, img_tensor, patch_size=256, overlap=64, device='cuda'):
    """
    使用滑动窗口策略进行大图推理，并使用汉明窗进行平滑融合，防止出现拼接缝隙。

    img_tensor: [1, C, H, W] 归一化到 [0, 1] 的 Tensor
    patch_size: 训练时的 patch_size (256)
    overlap: 相邻 patch 之间的重叠像素数
    """
    _, C, H, W = img_tensor.shape
    stride = patch_size - overlap

    # 初始化输出画布和权重累加画布
    out_tensor = torch.zeros((1, C, H, W), device=device)
    weight_acc = torch.zeros((1, 1, H, W), device=device)

    # 创建二维平滑窗口 (Hann Window)，中心权重高，边缘权重趋近于 0
    window_1d = torch.hann_window(patch_size, device=device)
    window_2d = window_1d.unsqueeze(0) * window_1d.unsqueeze(1)
    window_2d = window_2d.unsqueeze(0).unsqueeze(0)  # [1, 1, P, P]

    # 计算需要滑动的网格起始点
    h_starts = list(range(0, H - patch_size + stride, stride))
    w_starts = list(range(0, W - patch_size + stride, stride))

    # 确保最后一个 patch 能够紧贴右侧和底部边界
    if h_starts[-1] + patch_size < H:
        h_starts.append(H - patch_size)
    if w_starts[-1] + patch_size < W:
        w_starts.append(W - patch_size)

    # 滑动窗口进行局部推理
    for h in h_starts:
        for w in w_starts:
            # 裁剪局部 patch
            patch = img_tensor[:, :, h:h + patch_size, w:w + patch_size]

            # 模型前向传播 (使用 AMP 提速)
            with torch.cuda.amp.autocast():
                pred_patch = model(patch)

            # 将结果与权重相乘并累加到画布对应的位置
            out_tensor[:, :, h:h + patch_size, w:w + patch_size] += pred_patch * window_2d
            weight_acc[:, :, h:h + patch_size, w:w + patch_size] += window_2d

    # 防止除以 0 (理论上只要步长 < patch_size 且窗口覆盖全图就不会发生)
    weight_acc = torch.clamp(weight_acc, min=1e-8)

    # 归一化并限制像素范围
    final_output = out_tensor / weight_acc
    final_output = torch.clamp(final_output, 0.0, 1.0)

    return final_output


# =========================================================================
# 主函数
# =========================================================================
def main():
    # -----------------------------
    # 1. 基础配置+
    # -----------------------------
    CKPT_PATH = "./experiments/mamba_baseline_v1/checkpoints/best_model.pth"
    INPUT_IMAGE_PATH = "/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/input/00023.PNG"  # 替换为你的 2560x2560 输入图片路径
    OUTPUT_IMAGE_PATH = "./test_output023.png"

    PATCH_SIZE = 256
    OVERLAP = 64  # 重叠区域大小，建议设置在 32~128 之间，越大越平滑但推理越慢

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" [Infer] 推理任务启动，使用设备: {device}")

    # -----------------------------
    # 2. 加载模型与权重
    # -----------------------------
    model = MambaRealSR11(scale=1, dim=48).to(device)

    if os.path.isfile(CKPT_PATH):
        print(f" [Infer] 正在加载权重: {CKPT_PATH}")
        checkpoint = torch.load(CKPT_PATH, map_location=device)

        # 优先使用 EMA 权重，因为在训练代码中 EMA 的平滑历史表现通常更好
        if 'ema_state' in checkpoint:
            model.load_state_dict(checkpoint['ema_state'])
            print("✅ [Infer] 成功加载 EMA 权重")
        else:
            model.load_state_dict(checkpoint['model_state'])
            print("✅ [Infer] 成功加载普通 Model 权重")
    else:
        raise FileNotFoundError(f"❌ 找不到权重文件: {CKPT_PATH}")

    model.eval()

    # -----------------------------
    # 3. 图像读取与预处理
    # -----------------------------
    print(f" [Infer] 正在读取图像: {INPUT_IMAGE_PATH}")
    img = Image.open(INPUT_IMAGE_PATH).convert('RGB')

    # 转换为 Tensor，形状变为 [1, 3, H, W]，值域 [0, 1]
    input_tensor = ToTensor()(img).unsqueeze(0).to(device)

    print(f"📏 [Infer] 原始图像尺寸: {input_tensor.shape[2]}x{input_tensor.shape[3]}")

    # -----------------------------
    # 4. 执行大图推理
    # -----------------------------
    print(f"🧩 [Infer] 正在使用 Overlap-Tile 策略进行推理 (Patch: {PATCH_SIZE}, Overlap: {OVERLAP})...")
    output_tensor = overlap_tile_inference(
        model=model,
        img_tensor=input_tensor,
        patch_size=PATCH_SIZE,
        overlap=OVERLAP,
        device=device
    )

    # -----------------------------
    # 5. 后处理与保存
    # -----------------------------
    # 移除 Batch 维度，转换回 PIL Image
    output_img = ToPILImage()(output_tensor.squeeze(0).cpu())
    output_img.save(OUTPUT_IMAGE_PATH)
    print(f" [Infer] 推理完成！结果已保存至: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    main()
