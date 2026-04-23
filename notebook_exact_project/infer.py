import os
import time
from glob import glob
from operator import add

import cv2
import numpy as np
import torch
import yaml
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score

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


def calculate_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    # Ground truth
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    # Prediction
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc]


def mask_parse(mask: np.ndarray):
    mask = np.expand_dims(mask, axis=-1)            # (H, W, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  # (H, W, 3)
    return mask


def read_image(path: str, size):
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
    return image, x


def read_mask(path: str, size):
    h, w = size
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Could not read mask: {path}")
    if (mask.shape[0], mask.shape[1]) != (h, w):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    y = np.expand_dims(mask, axis=0)
    y = y / 255.0
    y = np.expand_dims(y, axis=0).astype(np.float32)
    y = torch.from_numpy(y)
    return mask, y


def main(cfg_path: str = "config.yml"):
    cfg = load_config(cfg_path)
    seeding(int(cfg["seed"]))

    h = int(cfg["input"]["height"])
    w = int(cfg["input"]["width"])
    size = (h, w)

    # Folders
    results_dir = cfg["paths"].get("results_dir", "results")
    create_dir(results_dir)

    # Load dataset (same as notebook: sorted(glob()))
    test_x = sorted(glob(cfg["data"]["test_images_glob"]))
    test_y = sorted(glob(cfg["data"]["test_masks_glob"]))

    checkpoint_path = cfg["paths"]["checkpoint_path"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet().to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0]
    time_taken = []

    for x_path, y_path in tqdm(list(zip(test_x, test_y)), total=len(test_x)):
        name = os.path.splitext(os.path.basename(x_path))[0]

        image, x = read_image(x_path, size)
        _, y = read_mask(y_path, size)

        x = x.to(device)
        y = y.to(device)

        with torch.no_grad():
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)

            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))

            pred_np = pred_y[0].cpu().numpy()      # (1, H, W)
            pred_np = np.squeeze(pred_np, axis=0)  # (H, W)
            pred_np = pred_np > 0.5
            pred_np = np.array(pred_np, dtype=np.uint8)

        # Save visualization (same as notebook)
        mask = cv2.imread(y_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Could not read mask: {y_path}")
        if (mask.shape[0], mask.shape[1]) != size:
            mask = cv2.resize(mask, (size[1], size[0]), interpolation=cv2.INTER_NEAREST)

        ori_mask = mask_parse(mask)
        pred_vis = mask_parse(pred_np)
        line = np.ones((size[0], 10, 3)) * 128

        cat_images = np.concatenate([image, line, ori_mask, line, pred_vis * 255], axis=1)
        cv2.imwrite(os.path.join(results_dir, f"{name}.png"), cat_images)

    # Report
    denom = max(len(test_x), 1)
    jaccard = metrics_score[0] / denom
    f1 = metrics_score[1] / denom
    recall = metrics_score[2] / denom
    precision = metrics_score[3] / denom
    acc = metrics_score[4] / denom

    print(
        f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - "
        f"Precision: {precision:1.4f} - Acc: {acc:1.4f}"
    )

    fps = 1.0 / float(np.mean(time_taken)) if len(time_taken) else 0.0
    print("FPS:", fps)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config.yml")
    args = parser.parse_args()
    main(args.config)
