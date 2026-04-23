import os
from glob import glob
import cv2
import numpy as np
from tqdm import tqdm


def create_dir(path: str):
    """创建目录（若不存在）"""
    if not os.path.exists(path):
        os.makedirs(path)


def read_mask(path: str, target_size=None):
    """
    读取二值掩码图并标准化处理
    :param path: 掩码路径
    :param target_size: 目标尺寸 (h, w)，None则保持原图尺寸
    :return: 标准化后的二值掩码（0/1）、原图尺寸
    """
    # 以灰度模式读取
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"无法读取掩码文件: {path}")

    # 保存原始尺寸
    ori_h, ori_w = mask.shape[:2]

    # 二值化（确保只有0/1，适配不同格式的mask）
    mask = (mask > 127).astype(np.uint8)  # 127为灰度阈值，兼容0/255或0/1格式

    # 调整尺寸（若指定）
    if target_size is not None:
        h, w = target_size
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    return mask, (ori_h, ori_w)


def get_matched_files(pred_dir: str, gt_dir: str):
    """
    匹配预测图和GT图的文件名（核心：按名称匹配，忽略后缀/前缀差异）
    :return: 匹配后的列表 [(pred_path, gt_path), ...]
    """
    # 修复核心：遍历扩展名，逐个glob并合并结果
    pred_ext = ('.png', '.jpg', '.jpeg', '.PNG', '.tiff')
    pred_files = []
    for ext in pred_ext:
        pred_files.extend(glob(os.path.join(pred_dir, f"*{ext}")))
    pred_files = sorted(pred_files)

    gt_files = []
    for ext in pred_ext:
        gt_files.extend(glob(os.path.join(gt_dir, f"*{ext}")))
    gt_files = sorted(gt_files)

    # 构建文件名映射（去除后缀/特殊字符，只保留核心名称）
    def clean_name(name):
        # 去除后缀、_mask/_pred等常见后缀
        name = os.path.splitext(os.path.basename(name))[0]
        for suffix in ['_mask', '_pred', '_predict', '_seg']:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        return name.lower()

    gt_name2path = {clean_name(p): p for p in gt_files}
    matched_pairs = []

    # 匹配预测图和GT图
    for pred_path in pred_files:
        clean_pred_name = clean_name(pred_path)
        if clean_pred_name in gt_name2path:
            matched_pairs.append((pred_path, gt_name2path[clean_pred_name]))
        else:
            print(f"警告：预测图 {pred_path} 未找到对应的GT图")

    if not matched_pairs:
        raise ValueError("未匹配到任何预测图和GT图，请检查文件名是否对应")

    return matched_pairs


def create_overlay_comparison(gt_mask, pred_mask):
    """
    生成视觉优化的对比叠加图：
    左侧=GT图（绿色）、中间=预测图（红色）、右侧=重叠对比图（GT绿+预测红）
    """
    # 统一尺寸（以GT尺寸为基准）
    h, w = gt_mask.shape[:2]
    if pred_mask.shape[:2] != (h, w):
        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    # 定义柔和配色（低饱和+低亮度，视觉更舒服）
    gt_color = (80, 200, 80)  # 柔和绿色（BGR）
    pred_color = (80, 80, 200)  # 柔和红色（BGR，实际是偏玫红，不刺眼）
    alpha = 0.4  # 透明度（平衡清晰和融合）
    blur_kernel = (3, 3)  # 修复：改为奇数核（3x3），符合OpenCV要求

    # --------------- 处理GT图（绿色）---------------
    gt_blurred = cv2.GaussianBlur(gt_mask.astype(np.float32), blur_kernel, 0)
    gt_blurred = np.clip(gt_blurred, 0, 1)
    gt_vis = np.zeros((h, w, 3), dtype=np.uint8)
    gt_vis[gt_blurred > 0.5] = gt_color  # GT区域填充绿色

    # --------------- 处理预测图（红色）---------------
    pred_blurred = cv2.GaussianBlur(pred_mask.astype(np.float32), blur_kernel, 0)
    pred_blurred = np.clip(pred_blurred, 0, 1)
    pred_vis = np.zeros((h, w, 3), dtype=np.uint8)
    pred_vis[pred_blurred > 0.5] = pred_color  # 预测区域填充红色

    # --------------- 生成重叠对比图 ---------------
    overlay_vis = np.zeros((h, w, 3), dtype=np.float32)
    # 叠加GT（绿色）
    overlay_vis[:, :, 0] += gt_blurred * gt_color[0] * alpha
    overlay_vis[:, :, 1] += gt_blurred * gt_color[1] * alpha
    overlay_vis[:, :, 2] += gt_blurred * gt_color[2] * alpha
    # 叠加预测（红色）
    overlay_vis[:, :, 0] += pred_blurred * pred_color[0] * alpha
    overlay_vis[:, :, 1] += pred_blurred * pred_color[1] * alpha
    overlay_vis[:, :, 2] += pred_blurred * pred_color[2] * alpha
    # 限制范围并转格式
    overlay_vis = np.clip(overlay_vis, 0, 255).astype(np.uint8)

    # --------------- 拼接三合一对比图 ---------------
    # 添加分隔线（灰色，细线条）
    separator = np.ones((h, 5, 3), dtype=np.uint8) * 128
    # 拼接：GT图 + 分隔线 + 预测图 + 分隔线 + 重叠图
    combined_img = np.hstack([gt_vis, separator, pred_vis, separator, overlay_vis])

    # 添加文字标注（左上角，白色文字+黑色描边，清晰不遮挡）
    def add_text(img, text, pos):
        # 黑色描边
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
        # 白色文字
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    add_text(combined_img, "GT (Green)", (20, 40))
    add_text(combined_img, "Pred (Red)", (w + 15, 40))
    add_text(combined_img, "Overlay", (2 * w + 25, 40))

    return combined_img


def main(pred_dir: str, gt_dir: str, output_dir: str):
    """
    主函数：读取预测图+GT图，生成重叠对比图
    :param pred_dir: 分割预测图目录
    :param gt_dir: GT图目录
    :param output_dir: 输出目录
    """
    # 1. 初始化
    create_dir(output_dir)
    print(f"开始处理：\n- 预测图目录：{pred_dir}\n- GT图目录：{gt_dir}\n- 输出目录：{output_dir}")

    # 2. 匹配预测图和GT图
    matched_pairs = get_matched_files(pred_dir, gt_dir)
    print(f"成功匹配到 {len(matched_pairs)} 组文件")

    # 3. 批量处理
    for pred_path, gt_path in tqdm(matched_pairs, desc="生成对比图"):
        # 获取核心文件名（用于保存）
        core_name = os.path.splitext(os.path.basename(pred_path))[0]
        core_name = core_name.replace('_mask', '').replace('_pred', '')

        # 读取并处理mask
        try:
            gt_mask, gt_size = read_mask(gt_path)
            pred_mask, _ = read_mask(pred_path, target_size=gt_size)
        except Exception as e:
            print(f"处理 {pred_path} 失败：{e}")
            continue

        # 生成优化后的对比叠加图
        combined_img = create_overlay_comparison(gt_mask, pred_mask)

        # 保存结果
        save_path = os.path.join(output_dir, f"{core_name}_comparison.png")
        cv2.imwrite(save_path, combined_img)

    print(f"\n✅ 处理完成！所有对比图已保存到：{output_dir}")


if __name__ == "__main__":
  
    PRED_DIR = "/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/ArcNet/"
    GT_DIR = "/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/gt/"
    OUTPUT_DIR = "/shared_data/users/yili/Hybrid-Mamba-UNet/data/res/ArcNet_overlay/"

    main(PRED_DIR, GT_DIR, OUTPUT_DIR)
