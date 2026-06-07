"""predict_augmented.py - Inference for Task 3 (augmented model).

Loads the robust augmented model and runs inference on data/predict/,
applying the calibrated threshold. Writes artifacts/task03/predictions.csv.

Usage: python predict_augmented.py --timeout_seconds 600
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK03_DIR = os.path.join(ARTIFACTS_DIR, "task03")
PREDICT_DIR = os.path.join(DATA_DIR, "predict")

TARGET_SIZE = (64, 64)
K = 32
BATCH_SIZE = 64

torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


def build_cnn(k=K):
    return nn.Sequential(
        nn.Conv2d(3, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(k, 2*k, 3, padding=1), nn.BatchNorm2d(2*k), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(2*k, 4*k, 3, padding=1), nn.BatchNorm2d(4*k), nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(4*k, 4*k, 3, padding=1), nn.BatchNorm2d(4*k), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(), nn.Dropout(0.5), nn.Linear(4*k, 2),
    )


def load_parquet_dir(directory):
    frames = []
    for f in sorted(os.listdir(directory)):
        if f.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(directory, f)))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def img_to_tensor(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(TARGET_SIZE, Image.BICUBIC)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    t0 = time.time()
    os.makedirs(TASK03_DIR, exist_ok=True)

    # Load augmented model
    for name in ["best_model_augmented.pt", "last_model_augmented.pt"]:
        path = os.path.join(ARTIFACTS_DIR, name)
        if os.path.exists(path):
            break
    else:
        print("ERROR: No augmented model found. Run train_augmented.py first.")
        sys.exit(1)

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_cnn(K)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = ckpt.get("threshold", 0.5)
    print(f"Loaded {path}  threshold={threshold:.4f}")

    # Load predict data
    df = load_parquet_dir(PREDICT_DIR)
    if df.empty:
        print("ERROR: No predict data found.")
        sys.exit(1)
    print(f"Predict samples: {len(df)}")

    # Inference
    row_ids = df["row_id"].values
    preds = []
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Predicting"):
        batch = df.iloc[start:start + BATCH_SIZE]
        tensors = [img_to_tensor(row["image"]) for _, row in batch.iterrows()]
        with torch.no_grad():
            probs = torch.softmax(model(torch.stack(tensors)), dim=1)[:, 1]
        preds.extend((probs.numpy() >= threshold).astype(int))

    # Write CSV
    out = os.path.join(TASK03_DIR, "predictions.csv")
    pd.DataFrame({"row_id": row_ids, "predicted_label": preds}).to_csv(
        out, index=False)
    print(f"Saved {out}  (0s={preds.count(0)}, 1s={preds.count(1)})")
    print(f"[predict_augmented.py] Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()




def augment_batch(X):
    """Heavy per-image augmentation for robustness to compression/blur/jitter."""
    n, c, h, w = X.shape

    # Only augment ~80% of images
    mask = torch.rand(n) < 0.8
    aug = X[mask].clone()
    n_aug = aug.shape[0]
    if n_aug == 0:
        return X

    # --- Per-image pixel transforms ---

    # Gaussian noise (sigma 0.05-0.12, 60% of images)
    m = (torch.rand(n_aug, 1, 1, 1) < 0.6).float()
    sigma = 0.05 + torch.rand(n_aug, 1, 1, 1) * 0.07
    aug = aug + torch.randn_like(aug) * sigma * m

    # Brightness shift +/-0.20 (50% of images)
    m = (torch.rand(n_aug, 1, 1, 1) < 0.5).float()
    aug = aug + (torch.rand(n_aug, 1, 1, 1) - 0.5) * 0.4 * m

    # Contrast scaling 0.6-1.4 (50% of images)
    m = (torch.rand(n_aug, 1, 1, 1) < 0.5).float()
    factor = 0.6 + torch.rand(n_aug, 1, 1, 1) * 0.8
    mean = aug.mean(dim=(2, 3), keepdim=True)
    aug = aug + (mean + (aug - mean) * factor - aug) * m

    # Per-channel color shift +/-0.08 (40% of images)
    m = (torch.rand(n_aug, 1, 1, 1) < 0.4).float()
    aug = aug + (torch.rand(n_aug, c, 1, 1) - 0.5) * 0.16 * m

    # --- Spatial transforms (batch-level) ---

    # Downscale + upscale (30% chance) — simulates compression
    if torch.rand(1).item() < 0.3:
        scale = np.random.choice([0.4, 0.5, 0.6, 0.75])
        small = (int(h * scale), int(w * scale))
        aug = F.interpolate(aug, size=small, mode='bilinear', align_corners=False)
        aug = F.interpolate(aug, size=(h, w), mode='bilinear', align_corners=False)

    # Gaussian blur (30% chance)
    if torch.rand(1).item() < 0.3:
        ks = np.random.choice([3, 5])
        sig = np.random.uniform(0.5, 1.5)
        ax = torch.arange(ks, dtype=torch.float32) - ks // 2
        k1d = torch.exp(-ax**2 / (2 * sig**2))
        k1d = k1d / k1d.sum()
        k2d = (k1d[:, None] @ k1d[None, :]).expand(c, 1, ks, ks)
        pad = ks // 2
        aug = F.conv2d(F.pad(aug, [pad]*4, mode='reflect'), k2d, groups=c)

    # Horizontal flip (per-image, 50%)
    flip = torch.rand(n_aug) < 0.5
    aug[flip] = aug[flip].flip(-1)

    # Cutout / random erasing (40% of images)
    cut = torch.rand(n_aug) < 0.4
    n_cut = int(cut.sum())
    if n_cut > 0:
        eh = torch.randint(int(h*0.1), int(h*0.3)+1, (n_cut,))
        ew = torch.randint(int(w*0.1), int(w*0.3)+1, (n_cut,))
        y0 = (torch.rand(n_cut) * (h - eh.float())).long()
        x0 = (torch.rand(n_cut) * (w - ew.float())).long()
        idx = torch.where(cut)[0]
        for j in range(n_cut):
            i = idx[j]
            aug[i, :, y0[j]:y0[j]+eh[j], x0[j]:x0[j]+ew[j]] = torch.rand(c, 1, 1)

    X[mask] = aug.clamp(0.0, 1.0)
    return X