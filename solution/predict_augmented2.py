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

    # Load the trained model

    for model_name in ["best_model_augmented.pt", "last_model_augmented.pt"]:
        model_path = os.path.join(ARTIFACTS_DIR, model_name)
        if os.path.exists(model_path):
            break
    else:        
        print("ERROR: No model found. Run train.py first.")
        sys.exit(1)

    
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model = build_cnn(K)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = ckpt.get("threshold", 0.5)
    print(f"Loaded {model_path}  threshold={threshold:.4f}")

    # Load predict data
    df = load_parquet_dir(PREDICT_DIR)
    if df.empty:
        print("ERROR: No predict data found.")
        sys.exit(1)
    
    # Predict
    print("Predicting...")
    row_ids = df["row_id"].values
    predictions = []
    for start in tqdm(range(0, len(df), BATCH_SIZE), desc="Predicting"):
        end = min(start + BATCH_SIZE, len(df))
        batch = df.iloc[start:end]
        img_tensors = torch.stack([img_to_tensor(img) for img in batch["image"].values])
        with torch.no_grad():
            logits = model(img_tensors)
            probs = torch.softmax(logits, dim=1)[:, 1]  # AI class probabilities
            predictions.extend((probs.numpy() >= threshold).astype(int))

    # Save predictions    df["label"] = np.array(predictions, dtype=np.int8)
    out_path = os.path.join(TASK03_DIR, "predictions.csv")
    pd.DataFrame({"row_id": row_ids, "predicted_label": predictions}).to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}  ({len(predictions)} samples)")
    print(f"Total time: {time.time() - t0:.2f} seconds")

if __name__ == "__main__":
    main()