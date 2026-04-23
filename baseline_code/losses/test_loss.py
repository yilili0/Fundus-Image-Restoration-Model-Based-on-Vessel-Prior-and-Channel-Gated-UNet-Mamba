from baseline_code.losses.loss import FundusCompositeLoss, build_vessel_teacher_from_files, VesselTeacherConfig
import  torch
# Step 1: 准备测试数据
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size, channels, height, width = 2, 3, 256, 256
pred = torch.rand(batch_size, channels, height, width).to(device)
gt = torch.rand(batch_size, channels, height, width).to(device)

# Step 2: 加载教师模型（可选）
# 假设你有一个预训练的 UNet 模型文件和对应的 checkpoint
model_py_path = "/shared_data/users/yili/Hybrid-Mamba-UNet/notebook_exact_project/model.py"
checkpoint_path = "/shared_data/users/yili/Hybrid-Mamba-UNet/notebook_exact_project/files/checkpoint.pth"

try:
    vessel_teacher = build_vessel_teacher_from_files(
        model_py_path=model_py_path,
        checkpoint_path=checkpoint_path,
        device=device,
        in_channels=3,
        out_channels=1,
    )
except Exception as e:
    print(f"Failed to load teacher model: {e}")
    vessel_teacher = None

# Step 3: 初始化损失函数
# 使用 VesselTeacherConfig 创建配置对象
vessel_cfg = VesselTeacherConfig(
    mask_gamma=2.0,
    hard_thresh=None,
    feature_layers=["e1.conv", "e2.conv", "e3.conv"],  # 可选，默认值已在类中定义
)

loss_fn = FundusCompositeLoss(
    assume_range="0_1",
    weights={"w_charb": 1.0, "w_msssim": 0.1, "w_ffl": 0.05, "w_vessel": 0.02},
    schedule={"vessel_start": 0.2, "vessel_full": 0.8, "feat_start": 0.8, "feat_full": 1.0},
    vessel_teacher=vessel_teacher,
    vessel_cfg=vessel_cfg,  # 传入 VesselTeacherConfig 对象
    vessel_w_bce=1.0,
    vessel_w_dice=1.0,
    vessel_w_feat=0.2,
).to(device)

# Step 4: 运行前向传播
progress = 0.5  # 训练进度（0~1）
loss_value, stats = loss_fn(pred, gt, progress=progress)

# Step 5: 验证输出
print("Total Loss:", loss_value.item())
for key, value in stats.items():
    print(f"{key}: {value.item()}")
