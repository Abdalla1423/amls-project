"""predict.py - Inference for Task 2.

Loads the best trained model and runs inference on data/predict/,
applying the calibrated threshold. Writes artifacts/task02/predictions.csv.

Usage: python predict.py --timeout_seconds 600
"""

import argparse
import io
import os
import signal
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
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
PREDICT_DIR = os.path.join(DATA_DIR, "predict")

TARGET_SIZE = (64, 64)
K = 32
BATCH_SIZE = 64

torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


def timeout_handler(signum, frame):
    print("[predict.py] Timeout – exiting.")
    sys.exit(0)


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
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(4*k, 2),
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

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, args.timeout_seconds - 30))

    t0 = time.time()
    os.makedirs(TASK02_DIR, exist_ok=True)

    # Load model
    for name in ["best_model.pt", "last_model.pt"]:
        path = os.path.join(ARTIFACTS_DIR, name)
        if os.path.exists(path):
            break
    else:
        print("ERROR: No model found. Run train.py first.")
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
    out = os.path.join(TASK02_DIR, "predictions.csv")
    pd.DataFrame({"row_id": row_ids, "predicted_label": preds}).to_csv(
        out, index=False)
    print(f"Saved {out}  (0s={preds.count(0)}, 1s={preds.count(1)})")
    print(f"[predict.py] Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
