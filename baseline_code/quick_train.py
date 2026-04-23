import os
# 严格限制底层计算库的线程数，防止 CPU 爆炸
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["cv2_NUM_THREADS"] = "0"  # 如果代码里用到了 OpenCV，关掉它的多线程
import time
import copy
import signal
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR


from models.model import MambaRealSR11
from data.dataset import (
    PipelineConfig,
    make_train_val_split,
    FundusPairInfinitePatchDataset,
    FundusPairValFixedSamples
)


# =========================================================================
# 0. 安全日志模块 (防卡死 & 实时落盘)
# =========================================================================
class SafeLogger(object):
    """
    """

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "a", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def __getattr__(self, attr):
        # 当第三方库请求未知的属性（如 isatty）时，透明地转发给真正的 terminal，防止崩溃
        return getattr(self.terminal, attr)


# =========================================================================
# 1. 训练策略组件 (EMA & Loss)
# =========================================================================
class ModelEMA:
    """
    指数移动平均 (Exponential Moving Average)。平滑历史权重，提升验证集表现。
    """

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        self.ema.eval()
        self.decay = decay
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model):
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema.parameters(), model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1 - self.decay)


class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps2 = eps * eps

    def forward(self, pred, target):
        diff = pred - target
        return torch.mean(torch.sqrt((diff * diff) + self.eps2))


def calculate_psnr(pred, gt):
    mse = torch.mean((pred - gt) ** 2)
    if mse == 0:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


# =========================================================================
# 2. 核心训练逻辑
# =========================================================================
def main():
    # -----------------------------
    # 基础配置与安全日志激活
    # -----------------------------
    DATA_ROOT = "/shared_data/users/yili/Hybrid-Mamba-UNet/data/Real_Fundus/"
    OUT_DIR = "./experiments/mamba_baseline_v1"
    CKPT_DIR = os.path.join(OUT_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)
    RESUME_CKPT = os.path.join(CKPT_DIR, "latest.pth")  # 或者 "emergency_save.pth"
    # 激活日志双写（控制台 + 文件）
    log_path = os.path.join(OUT_DIR, "train_detail.log")
    sys.stdout = SafeLogger(log_path)
    sys.stderr = sys.stdout  # 捕获所有报错信息入库

    # 训练超参数
    BATCH_SIZE = 8
    NUM_WORKERS = 4
    LEARNING_RATE = 2e-4
    EPOCHS = 100
    ITERS_PER_EPOCH = 500
    SAVE_FREQ = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("=" * 60)
    print(f" [Init] 正式训练任务启动")
    print(f" [Init] 输出目录: {OUT_DIR}")
    print(
        f"  [Init] 运算设备: {device} | 显卡: {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'N/A'}")
    print("=" * 60)

    # -----------------------------
    # 准备数据流 (加入防卡死机制)
    # -----------------------------
    split_dir = os.path.join(OUT_DIR, "splits")
    train_txt, val_txt = make_train_val_split(data_root=DATA_ROOT, out_dir=split_dir, val_ratio=20 / 120)

    cfg = PipelineConfig(patch_size=256)

    train_ds = FundusPairInfinitePatchDataset(data_root=DATA_ROOT, cfg=cfg, seed=1234, list_file=train_txt,
                                              augment=True)
    val_ds = FundusPairValFixedSamples(data_root=DATA_ROOT, cfg=cfg, list_file=val_txt, seed=2026, num_samples=200)

    # 安全配置：timeout 与 persistent_workers 彻底杜绝 DataLoader 死锁
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=True, prefetch_factor=2, persistent_workers=True, timeout=120
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
        pin_memory=True, prefetch_factor=2, persistent_workers=True, timeout=120
    )

    # -----------------------------
    # 模型、优化器与调度器
    # -----------------------------
    model = MambaRealSR11(scale=1, dim=48).to(device)
    ema_model = ModelEMA(model, decay=0.999)

    criterion = CharbonnierLoss(eps=1e-3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
    # -----------------------------
    # 模型、优化器与调度器
    # -----------------------------
    model = MambaRealSR11(scale=1, dim=48).to(device)
    ema_model = ModelEMA(model, decay=0.999)

    criterion = CharbonnierLoss(eps=1e-3).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999), weight_decay=1e-4)
    scaler = GradScaler()
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # -----------------------------
    # 断点续训加载逻辑
    # -----------------------------
    start_epoch = 0
    if RESUME_CKPT is not None and os.path.isfile(RESUME_CKPT):
        print(f" [Resume] 发现断点文件，正在加载: {RESUME_CKPT}")
        checkpoint = torch.load(RESUME_CKPT, map_location=device)

        # 恢复状态
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state'])
        ema_model.ema.load_state_dict(checkpoint['ema_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])

        print(f" [Resume] 成功恢复到 Epoch {start_epoch}，将从 Epoch {start_epoch + 1} 继续训练。")
    elif RESUME_CKPT is not None:
        print(f" [Resume] 未找到断点文件 {RESUME_CKPT}，将从头开始训练。")
    # -----------------------------
    # 存档辅助函数
    # -----------------------------
    def save_checkpoint(epoch, is_best=False, is_emergency=False):
        state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'ema_state': ema_model.ema.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict()
        }
        if is_emergency:
            path = os.path.join(CKPT_DIR, "emergency_save.pth")
            print(f"\n [Emergency] 紧急保存模型至: {path}")
        elif is_best:
            path = os.path.join(CKPT_DIR, "best_model.pth")
        else:
            path = os.path.join(CKPT_DIR, f"epoch_{epoch:03d}.pth")

        torch.save(state, path)
        if not is_emergency:
            torch.save(state, os.path.join(CKPT_DIR, "latest.pth"))

    # -----------------------------
    # 安全中断机制
    # -----------------------------
    graceful_stop = False

    def signal_handler(sig, frame):
        nonlocal graceful_stop
        print("\n [Emergency] 接收到中断信号！将在当前 Iteration 结束后安全退出并保存...")
        graceful_stop = True

    signal.signal(signal.SIGINT, signal_handler)

    # -----------------------------
    # 训练主循环 (受异常保护)
    # -----------------------------
    train_iter = iter(train_loader)
    best_psnr = 0.0
    current_epoch = 0

    try:
        for current_epoch in range(start_epoch + 1, EPOCHS + 1):
            model.train()
            epoch_loss = 0.0
            start_time = time.time()

            print(f"\n" + "-" * 60)
            print(f" [Epoch {current_epoch}/{EPOCHS}] | LR: {scheduler.get_last_lr()[0]:.2e}")

            for i in range(ITERS_PER_EPOCH):
                if graceful_stop:
                    break

                iter_start_time = time.time()
                try:
                    xin, xgt, _ = next(train_iter)
                except StopIteration:
                    train_iter = iter(train_loader)
                    xin, xgt, _ = next(train_iter)

                xin, xgt = xin.to(device), xgt.to(device)
                optimizer.zero_grad()

                with autocast():
                    preds = model(xin)
                    loss = criterion(preds, xgt)

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                scaler.step(optimizer)
                scaler.update()

                ema_model.update(model)
                epoch_loss += loss.item()

                if (i + 1) % 50 == 0:
                    iter_time = time.time() - iter_start_time
                    current_lr = optimizer.param_groups[0]['lr']

                    if torch.cuda.is_available():
                        mem_alloc = torch.cuda.memory_allocated() / 1024 ** 2
                        mem_reserved = torch.cuda.memory_reserved() / 1024 ** 2
                        mem_info = f"{mem_alloc:.0f}MB/{mem_reserved:.0f}MB"
                    else:
                        mem_info = "N/A"

                    print(f"  [Iter {i + 1:03d}/{ITERS_PER_EPOCH}] "
                          f"Loss: {loss.item():.5f} | "
                          f"LR: {current_lr:.2e} | "
                          f"Time: {iter_time:.3f}s/it | "
                          f"GPU Mem: {mem_info}")

            if graceful_stop:
                print(" 正在执行紧急保存机制...")
                save_checkpoint(current_epoch, is_emergency=True)
                break

            scheduler.step()
            print(
                f" [Epoch {current_epoch} 总结] 耗时: {time.time() - start_time:.2f}s | Avg Loss: {epoch_loss / ITERS_PER_EPOCH:.5f}")

            # -----------------------------
            # 验证循环 (使用 EMA)
            # -----------------------------
            ema_model.ema.eval()
            val_psnr = 0.0
            val_steps = 0

            with torch.no_grad():
                for xin, xgt, _ in val_loader:
                    xin, xgt = xin.to(device), xgt.to(device)
                    with autocast():
                        preds = ema_model.ema(xin)
                    preds_clamped = torch.clamp(preds, 0.0, 1.0)
                    val_psnr += calculate_psnr(preds_clamped, xgt)
                    val_steps += 1

            avg_val_psnr = val_psnr / val_steps
            print(f" [验证集评估] EMA Val PSNR: {avg_val_psnr:.2f} dB")

            is_best = avg_val_psnr > best_psnr
            if is_best:
                best_psnr = avg_val_psnr
                save_checkpoint(current_epoch, is_best=True)
                print(f" [存档] 发现最佳模型！已保存至 best_model.pth (PSNR: {best_psnr:.2f})")

            if current_epoch % SAVE_FREQ == 0:
                save_checkpoint(current_epoch)
                print(f" [存档] 已保存第 {current_epoch} 轮常规检查点。")

    except Exception as e:
        print(f"\n [Fatal Error] 训练中途发生严重错误: {e}")
        print(" 正在尝试抢救模型权重...")
        save_checkpoint(max(1, current_epoch), is_emergency=True)
        raise e

    finally:
        print("\n [Cleanup] 正在清理 Dataloader 线程，释放显存...")
        del train_iter
        del train_loader
        del val_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(" [Cleanup] 清理完成，安全退出。")


if __name__ == "__main__":
    main()
