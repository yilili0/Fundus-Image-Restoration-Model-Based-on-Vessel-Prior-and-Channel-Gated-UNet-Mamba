import os
import time
import random
from glob import glob
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

from dataset import DriveDataset
from model import build_unet, DiceBCELoss


# ----------------------------
# Utils (same as notebook)
# ----------------------------
def seeding(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def create_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


def epoch_time(start_time: float, end_time: float):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    epoch_loss = 0.0
    model.train()

    for x, y in loader:
        x = x.to(device, dtype=torch.float32)
        y = y.to(device, dtype=torch.float32)

        optimizer.zero_grad()
        y_pred = model(x)
        loss = loss_fn(y_pred, y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(loader)


def evaluate(model, loader, loss_fn, device):
    epoch_loss = 0.0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32)

            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            epoch_loss += loss.item()

    return epoch_loss / len(loader)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main(cfg_path: str = "config.yml"):
    cfg = load_config(cfg_path)

    # Seeding (same as notebook)
    seeding(int(cfg["seed"]))

    # Folders (same as notebook)
    create_dir(cfg["paths"]["checkpoint_dir"])
    checkpoint_path = cfg["paths"]["checkpoint_path"]

    # Collect data paths (same as notebook logic: sorted(glob()))
    train_x = sorted(glob(cfg["data"]["train_images_glob"]))
    train_y = sorted(glob(cfg["data"]["train_masks_glob"]))
    valid_x = sorted(glob(cfg["data"]["valid_images_glob"]))
    valid_y = sorted(glob(cfg["data"]["valid_masks_glob"]))

    print(f"Dataset Size:\\nTrain: {len(train_x)} - Valid: {len(valid_x)}")

    # Hyperparams
    h = int(cfg["input"]["height"])
    w = int(cfg["input"]["width"])
    size: Tuple[int, int] = (h, w)

    batch_size = int(cfg["train"]["batch_size"])
    lr = float(cfg["train"]["lr"])
    num_epochs = int(cfg["train"]["num_epochs"])
    num_workers = int(cfg["train"]["num_workers"])

    # Datasets / loaders
    train_dataset = DriveDataset(train_x, train_y, size=size)
    valid_dataset = DriveDataset(valid_x, valid_y, size=size)

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=bool(cfg["train"].get("pin_memory", True)) if num_workers > 0 else False,
    )

    valid_loader = DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=bool(cfg["train"].get("pin_memory", True)) if num_workers > 0 else False,
    )

    # Device / model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_unet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Notebook creates a ReduceLROnPlateau scheduler but does not step it in the loop.
    # To keep the original behavior, we create it and leave it unused by default.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, "min", patience=int(cfg["train"].get("lr_patience", 5)), verbose=True
    )
    use_scheduler = bool(cfg["train"].get("use_scheduler", False))

    loss_fn = DiceBCELoss()

    best_valid_loss = float("inf")

    for epoch in range(num_epochs):
        start_time = time.time()

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        valid_loss = evaluate(model, valid_loader, loss_fn, device)

        if use_scheduler:
            scheduler.step(valid_loss)

        # Save best (same as notebook)
        if valid_loss < best_valid_loss:
            print(
                f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. "
                f"Saving checkpoint: {checkpoint_path}"
            )
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), checkpoint_path)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(
            f"Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\\n"
            f"\\tTrain Loss: {train_loss:.3f}\\n"
            f"\\tVal. Loss: {valid_loss:.3f}\\n"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    args = parser.parse_args()

    main(args.config)
