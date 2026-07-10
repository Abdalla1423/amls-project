"""predict.py - Inference for Task 2.

Outputs: artifacts/task02/predictions.csv (columns: row_id, predicted_label)

Usage: python predict.py --timeout_seconds 600
"""

import argparse
import csv
import glob
import io
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from PIL import Image

from common import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEVICE,
    ImageDataset,
    load_checkpoint,
    make_loader,
)

TIME_OUT = 600
IMAGE_SIZE = 64

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
PREDICT_DIR = os.path.join(DATA_DIR, "predict")
PREDICTIONS = os.path.join(TASK02_DIR, "predictions.csv")
CHECKPOINT_PATH = os.path.join(TASK02_DIR, "best_model.pt")


def preprocess_single_image(img_bytes, target_h=IMAGE_SIZE, target_w=IMAGE_SIZE):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((target_h, target_w), Image.BICUBIC)
    img_arr = np.array(img)
    return np.transpose(img_arr, (2, 0, 1))


def extract_features(df):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

    processed = []
    for img_bytes in df["image"].values:
        img = preprocess_single_image(img_bytes).astype(np.float32) / 255.0
        img = (img - mean) / std
        processed.append(img)
    return np.stack(processed)


def load_predict_data(input_path, batch_size=500):
    """Read parquet(s); return X and row_ids (from column if present, else running index)."""
    print(f"Processing {input_path}...")
    file_paths = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    if not file_paths:
        print(f"No parquet files found in {input_path}")
        return None, None

    X_parts, row_id_parts = [], []
    offset = 0
    for fp in file_paths:
        print(f"Reading {fp} in batches...")
        parquet_file = pq.ParquetFile(fp)
        schema_names = set(parquet_file.schema_arrow.names)
        cols = ["image"] + (["row_id"] if "row_id" in schema_names else [])

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=cols):
            df = batch.to_pandas()
            X_parts.append(extract_features(df))
            if "row_id" in df.columns:
                row_id_parts.append(df["row_id"].to_numpy())
            else:
                row_id_parts.append(np.arange(offset, offset + len(df)))
                offset += len(df)
            del df, batch

    X = np.concatenate(X_parts, axis=0)
    row_ids = np.concatenate(row_id_parts, axis=0)
    return X, row_ids


def save_predictions(row_ids, preds, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "predicted_label"])
        for rid, pred in zip(row_ids, preds):
            writer.writerow([rid, int(pred)])


def run_inference(checkpoint_path, predict_dir, output_path, timeout_seconds):
    start_time = time.time()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("\n=== Loading model ===")
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: {checkpoint_path} not found. Run training first.")
        sys.exit(1)
    model, threshold, _ = load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint (threshold={threshold:.4f})")

    print("\n=== Loading data ===")
    X, row_ids = load_predict_data(predict_dir)
    if X is None:
        sys.exit(1)

    y_dummy = np.zeros(len(X), dtype=np.int64)
    loader = make_loader(X, y_dummy, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_probs = []
    with torch.no_grad():
        for batch, _ in loader:
            if time.time() - start_time > timeout_seconds:
                print(f"\nERROR: Timeout after {len(all_probs) * BATCH_SIZE} samples. "
                      "Refusing to write truncated predictions.csv.")
                sys.exit(2)
            batch = batch.to(DEVICE)
            probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)

    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= threshold).astype(np.int64)
    print(f"Predicted {len(preds)} rows. Saving to {output_path}")
    save_predictions(row_ids, preds, output_path)
    print(f"\n[predict.py] Done in {time.time() - start_time:.1f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args = parser.parse_args()
    run_inference(CHECKPOINT_PATH, PREDICT_DIR, PREDICTIONS, args.timeout_seconds)


if __name__ == "__main__":
    main()
