import os
import cv2
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
以下是一段适合放在代码开头的说明，用于向其他使用者介绍这段代码的作用：

---

### 代码说明：眼底图像对齐质量检测工具

本脚本用于检测眼底图像数据集中 `input`（退化图像）与 `gt`（真实标注图像）之间的对齐质量。通过相位相关性分析（Phase Correlation）计算图像间的全局位移，并评估对齐精度。

#### 主要功能：
1. **图像配对检查**  
   自动匹配 `input` 和 `gt` 目录中的同名图像文件，检查尺寸一致性。
2. **对齐质量评估**  
   利用 OpenCV 的 `phaseCorrelate` 方法计算每对图像的相对位移 [(dx, dy)](file://D:\codespace\Hybrid-Mamba-UNet\data\dataset.py#L188-L188) 及相似度峰值 `peak`。
3. **预处理优化**  
   支持提取绿色通道（突出血管结构）、应用高通滤波增强特征对比度，提升检测鲁棒性。
4. **结果统计与报告**  
   生成包含位移、峰值等指标的 CSV 报告，并提供整体统计摘要。
5. **可视化最差案例**  
   筛选出对齐误差最大或相似度最低的前 K 对图像，生成四宫格对比图（原始图像、差异图、棋盘叠加图、边缘叠加图），便于人工复核。

#### 输出内容：
- **CSV 报告**：保存在 `_alignment_check/alignment_report.csv`，记录每对图像的检测结果。
- **可视化图像**：保存在 `_alignment_check/vis_worst/` 目录下，命名格式为 `{文件名}_dx{dx值}_dy{dy值}_peak{peak值}.png`。

#### 使用建议：
- `shift_norm <= 2 px`：表示基本像素级对齐。
- `shift_norm > 5~10 px` 或 `peak` 偏低：可能存在未对齐、旋转、缩放或局部形变问题，需重点关注。

--- 

将上述说明添加到代码文件顶部，可以帮助使用者快速了解脚本的功能和使用方法。
'''


# =========================
# Config
# =========================
ROOT = r"C:\Users\29638\Desktop\Real_Fundus\Real_Fundus"   #数据路径
INPUT_DIR = os.path.join(ROOT, "input")
GT_DIR = os.path.join(ROOT, "gt")

OUT_DIR = os.path.join(ROOT, "_alignment_check")
VIS_DIR = os.path.join(OUT_DIR, "vis_worst")
os.makedirs(VIS_DIR, exist_ok=True)

REPORT_CSV = os.path.join(OUT_DIR, "alignment_report.csv")

SUPPORTED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# 可视化最差多少对（按 shift 大、peak 低综合排序）
SAVE_WORST_K = 20

# 预处理：可选取绿色通道（眼底血管更明显）
USE_GREEN_CHANNEL = True

# 计算 phase correlation 前做轻微高通：提升鲁棒性
APPLY_HIGHPASS = True

# =========================
# Utils
# =========================
def list_images(folder):
    files = []
    for fn in os.listdir(folder):
        ext = os.path.splitext(fn)[1].lower()
        if ext in SUPPORTED_EXT:
            files.append(fn)
    return sorted(files)

def basename_no_ext(fn):
    return os.path.splitext(fn)[0]

def read_image(path):
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to read image: {path}")
    return img

def to_gray_or_green(img_bgr):
    if USE_GREEN_CHANNEL:
        # BGR -> take G
        g = img_bgr[:, :, 1]
        return g
    else:
        return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

def normalize_for_corr(x):
    x = x.astype(np.float32)
    # 归一化到 [0,1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-6)
    return x

def highpass(x):
    # x float32 [0,1]
    # 用大核高斯模糊做低频，再减掉得到高频
    low = cv2.GaussianBlur(x, (0, 0), 7)
    hp = x - low
    # 标准化
    hp = (hp - hp.mean()) / (hp.std() + 1e-6)
    return hp.astype(np.float32)

def phase_corr_shift(a, b):
    """
    a, b: float32 images, same size
    return: (dx, dy, peak)
    OpenCV phaseCorrelate returns shift (x, y) meaning:
      if you shift 'a' by (x,y), it aligns to 'b' (approx).
    """
    # Hanning window improves peak sharpness
    win = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (shift_x, shift_y), peak = cv2.phaseCorrelate(a, b, win)
    return float(shift_x), float(shift_y), float(peak)

def make_checkerboard(a, b, tile=64):
    h, w = a.shape[:2]
    out = a.copy()
    yy, xx = np.indices((h, w))
    mask = ((xx // tile + yy // tile) % 2 == 0)
    out[mask] = b[mask]
    return out

def overlay_edges(img_gray_a, img_gray_b):
    # edges in red/green overlay
    ea = cv2.Canny(img_gray_a, 40, 120)
    eb = cv2.Canny(img_gray_b, 40, 120)
    overlay = np.zeros((img_gray_a.shape[0], img_gray_a.shape[1], 3), dtype=np.uint8)
    overlay[:, :, 1] = eb  # GT edges in green
    overlay[:, :, 2] = ea  # input edges in red
    return overlay

def save_vis_pair(input_bgr, gt_bgr, dx, dy, peak, out_path):
    inp = to_gray_or_green(input_bgr)
    gt = to_gray_or_green(gt_bgr)

    # 归一化用于可视化差分
    inp_n = normalize_for_corr(inp)
    gt_n = normalize_for_corr(gt)

    diff = np.abs(inp_n - gt_n)
    diff_u8 = np.clip(diff * 255.0, 0, 255).astype(np.uint8)

    # checkerboard
    cb = make_checkerboard(
        cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB),
        cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB),
        tile=80
    )

    # edge overlay (on normalized gray)
    inp_u8 = np.clip(inp_n * 255.0, 0, 255).astype(np.uint8)
    gt_u8 = np.clip(gt_n * 255.0, 0, 255).astype(np.uint8)
    edges = overlay_edges(inp_u8, gt_u8)

    # create a 2x2 panel
    fig = plt.figure(figsize=(12, 10), dpi=120)

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(cv2.cvtColor(input_bgr, cv2.COLOR_BGR2RGB))
    ax1.set_title("Input (degraded)")
    ax1.axis("off")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2RGB))
    ax2.set_title("GT")
    ax2.axis("off")

    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(diff_u8, cmap="gray")
    ax3.set_title("Abs Diff (after simple norm)")
    ax3.axis("off")

    ax4 = fig.add_subplot(2, 2, 4)
    # show checkerboard with edge overlay blended
    cb2 = cb.copy()
    # blend edges on top (convert edges to RGB)
    edges_rgb = cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)
    cb2 = (0.75 * cb2 + 0.25 * edges_rgb).astype(np.uint8)
    ax4.imshow(cb2)
    ax4.set_title(f"Checkerboard + Edges | dx={dx:.2f}, dy={dy:.2f}, peak={peak:.4f}")
    ax4.axis("off")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

# =========================
# Main
# =========================
def main():
    input_files = list_images(INPUT_DIR)
    gt_files = list_images(GT_DIR)

    input_map = {basename_no_ext(fn): fn for fn in input_files}
    gt_map = {basename_no_ext(fn): fn for fn in gt_files}

    common_keys = sorted(set(input_map.keys()) & set(gt_map.keys()))
    missing_in_gt = sorted(set(input_map.keys()) - set(gt_map.keys()))
    missing_in_input = sorted(set(gt_map.keys()) - set(input_map.keys()))

    print(f"[Found] input: {len(input_files)} | gt: {len(gt_files)}")
    print(f"[Paired] {len(common_keys)}")
    if missing_in_gt:
        print(f"[Warn] Missing in gt: {len(missing_in_gt)} (show up to 10): {missing_in_gt[:10]}")
    if missing_in_input:
        print(f"[Warn] Missing in input: {len(missing_in_input)} (show up to 10): {missing_in_input[:10]}")

    rows = []
    for key in tqdm(common_keys, desc="Checking alignment"):
        in_path = os.path.join(INPUT_DIR, input_map[key])
        gt_path = os.path.join(GT_DIR, gt_map[key])

        in_img = read_image(in_path)
        gt_img = read_image(gt_path)

        if in_img.shape[:2] != gt_img.shape[:2]:
            rows.append({
                "key": key,
                "input_file": input_map[key],
                "gt_file": gt_map[key],
                "status": "SIZE_MISMATCH",
                "h_in": in_img.shape[0],
                "w_in": in_img.shape[1],
                "h_gt": gt_img.shape[0],
                "w_gt": gt_img.shape[1],
                "dx": np.nan,
                "dy": np.nan,
                "shift_norm": np.nan,
                "peak": np.nan
            })
            continue

        a = to_gray_or_green(in_img)
        b = to_gray_or_green(gt_img)

        a = normalize_for_corr(a)
        b = normalize_for_corr(b)

        if APPLY_HIGHPASS:
            a2 = highpass(a)
            b2 = highpass(b)
        else:
            a2 = a.astype(np.float32)
            b2 = b.astype(np.float32)

        dx, dy, peak = phase_corr_shift(a2, b2)
        shift_norm = math.sqrt(dx * dx + dy * dy)

        rows.append({
            "key": key,
            "input_file": input_map[key],
            "gt_file": gt_map[key],
            "status": "OK",
            "h_in": in_img.shape[0],
            "w_in": in_img.shape[1],
            "h_gt": gt_img.shape[0],
            "w_gt": gt_img.shape[1],
            "dx": dx,
            "dy": dy,
            "shift_norm": shift_norm,
            "peak": peak
        })

    df = pd.DataFrame(rows)
    df.to_csv(REPORT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n[Saved] CSV report -> {REPORT_CSV}")

    # 对 OK 的样本进行“最差排序”：shift 大 + peak 低
    ok = df[df["status"] == "OK"].copy()
    if len(ok) == 0:
        print("[Error] No OK pairs to visualize.")
        return

    # 组合一个排序分数：shift_norm（越大越差） + (1-peak)（越大越差）
    # peak 通常在 0~1 附近，但也可能略超；这里做 clamp
    peak_clamped = np.clip(ok["peak"].to_numpy(), 0.0, 1.0)
    score = ok["shift_norm"].to_numpy() + (1.0 - peak_clamped) * 10.0
    ok["bad_score"] = score

    ok_sorted = ok.sort_values("bad_score", ascending=False).head(SAVE_WORST_K)

    print(f"[Visualize] Saving worst {len(ok_sorted)} pairs -> {VIS_DIR}")
    for _, r in ok_sorted.iterrows():
        key = r["key"]
        in_path = os.path.join(INPUT_DIR, r["input_file"])
        gt_path = os.path.join(GT_DIR, r["gt_file"])
        in_img = read_image(in_path)
        gt_img = read_image(gt_path)

        out_path = os.path.join(VIS_DIR, f"{key}_dx{r['dx']:.2f}_dy{r['dy']:.2f}_peak{r['peak']:.4f}.png")
        save_vis_pair(in_img, gt_img, r["dx"], r["dy"], r["peak"], out_path)

    # 打印一个简要统计，帮你快速判断“整体是否对齐”
    print("\n=== Summary (OK pairs) ===")
    print(f"Pairs: {len(ok)}")
    print(f"Median shift_norm: {ok['shift_norm'].median():.3f} px")
    print(f"Mean shift_norm:   {ok['shift_norm'].mean():.3f} px")
    print(f"Max shift_norm:    {ok['shift_norm'].max():.3f} px")
    print(f"Median peak:       {ok['peak'].median():.4f}")
    print(f"Mean peak:         {ok['peak'].mean():.4f}")
    print("\nRule of thumb:")
    print("- shift_norm <= 2 px for most pairs => 基本像素级对齐")
    print("- shift_norm 经常 > 5~10 px 或 peak 偏低 => 很可能没对齐/存在旋转缩放/局部形变")

if __name__ == "__main__":
    main()
