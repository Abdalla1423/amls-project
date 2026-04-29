"""
Task 1.2 - Prediction / Inference

Loads the best trained model and the predict/ data split,
runs inference, and writes artifacts/task02/predictions.csv.

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
PREDICT_DIR = os.path.join(DATA_DIR, "predict")

TARGET_SIZE = (128, 128)
K = 32
BATCH_SIZE = 64

torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


def timeout_handler(signum, frame):
    print("[predict.py] Timeout reached – saving progress and exiting.")
    sys.exit(0)


def build_cnn(k: int = K) -> nn.Module:
    """Build the same CNN architecture used in training."""
    features = nn.Sequential(
        nn.Conv2d(3, k, kernel_size=3, padding=1),
        nn.BatchNorm2d(k),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(k, 2 * k, kernel_size=3, padding=1),
        nn.BatchNorm2d(2 * k),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(2 * k, 4 * k, kernel_size=3, padding=1),
        nn.BatchNorm2d(4 * k),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(4 * k, 2),
    )
    return nn.Sequential(features, classifier)


def load_parquet_dir(directory: str) -> pd.DataFrame:
    """Load all parquet files from a directory."""
    frames = []
    if not os.path.isdir(directory):
        print(f"Warning: directory {directory} not found.")
        return pd.DataFrame()
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(directory, fname)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def image_bytes_to_tensor(image_bytes: bytes,
                          target_size: tuple[int, int]) -> torch.Tensor:
    """Convert raw image bytes to a normalized tensor (C, H, W)."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(target_size, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr.transpose(2, 0, 1))  # (3, H, W)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, args.timeout_seconds - 30))

    start_time = time.time()
    os.makedirs(TASK02_DIR, exist_ok=True)

    # Load model
    model_path = os.path.join(ARTIFACTS_DIR, "best_model.pt")
    if not os.path.exists(model_path):
        model_path = os.path.join(ARTIFACTS_DIR, "last_model.pt")
    if not os.path.exists(model_path):
        print("ERROR: No trained model found. Run train.py first.")
        sys.exit(1)

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    model = build_cnn(K)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    threshold = checkpoint.get("threshold", 0.5)
    print(f"Loaded model from {model_path} (threshold={threshold:.4f})")

    # Load predict data
    print("Loading predict data...")
    predict_df = load_parquet_dir(PREDICT_DIR)
    if predict_df.empty:
        print("ERROR: No predict data found.")
        sys.exit(1)
    print(f"Loaded {len(predict_df)} samples for prediction.")

    # Run inference
    row_ids = predict_df["row_id"].values
    predictions = []

    # Process in batches
    for start_idx in tqdm(range(0, len(predict_df), BATCH_SIZE),
                          desc="Predicting"):
        end_idx = min(start_idx + BATCH_SIZE, len(predict_df))
        batch_df = predict_df.iloc[start_idx:end_idx]

        tensors = []
        for _, row in batch_df.iterrows():
            t = image_bytes_to_tensor(row["image"], TARGET_SIZE)
            tensors.append(t)

        X_batch = torch.stack(tensors)
        with torch.no_grad():
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1].numpy()

        batch_preds = (probs >= threshold).astype(int)
        predictions.extend(batch_preds)

    # Write predictions
    result_df = pd.DataFrame({
        "row_id": row_ids,
        "predicted_label": predictions,
    })
    out_path = os.path.join(TASK02_DIR, "predictions.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nSaved predictions to {out_path}")
    print(f"  Label distribution: {np.bincount(predictions)}")
    print(f"[predict.py] Completed in {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()
