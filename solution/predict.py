"""predict.py - Inference for Task 2.

Outputs: artifacts/task02/predictions.csv (columns: row_id, predicted_label)

Usage: python predict.py --timeout_seconds 600
"""

import argparse
import os
import sys
import csv
import glob
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from PIL import Image
import io
import time

import psutil

import torch
from torch.utils.data import DataLoader

from model import Given_CNN, ImageDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
PREDICT_DIR = os.path.join(DATA_DIR, "predict") 
PREDICTIONS = os.path.join(TASK02_DIR, "predictions.csv")
CHECKPOINT_PATH = os.path.join(TASK02_DIR, "best_model.pt")

TIME_OUT = 600

BATCH_SIZE = 64
IMAGE_SIZE = 64
DEVICE = "cpu"

def print_ram_usage(msg=""):
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024**3)
    print(f"{msg} RAM used: {ram_gb:.2f} GB")

# Loads the best model trained in task02
def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = Given_CNN()
    model.load_state_dict(ckpt["state_dict"])

    return model

# Save predictions to CSV file at artifacts/task02/predictions.csv
def save_predictions(all_preds, save_path=PREDICTIONS):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "predicted_label"])
        for id, pred in enumerate(all_preds):
            writer.writerow([id, pred])

# Preprocess a single image by center cropping and padding to 64x64, then convert to CHW format
def preprocess_single_image(img_bytes, target_h=IMAGE_SIZE, target_w=IMAGE_SIZE):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    img = img.resize((target_h, target_w), Image.BICUBIC)
    img_arr = np.array(img)
    return np.transpose(img_arr, (2, 0, 1))

# Extract features and labels from the dataframe
def extract_features_and_labels(df):
    processed_images = []

    mean = np.array([0.485, 0.456, 0.406],
                    dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225],
                   dtype=np.float32).reshape(3, 1, 1)
    
    for img_bytes in df["image"].values:
        processed_img = preprocess_single_image(img_bytes)
        processed_img = processed_img.astype(np.float32) / 255.0
        processed_img = (processed_img - mean) / std
        processed_images.append(processed_img.astype(np.float16))

    X = np.stack(processed_images)
    return X

def process_parquet(input_path, batch_size=500):

    print(f"Processing {input_path}...")
    
    file_paths = sorted(glob.glob(os.path.join(input_path, "*.parquet")))

    if not file_paths:
        print(f"No parquet files found in {input_path}")
        return
    
    X_parts = []

    for fp in file_paths:
        print(f"Reading {fp} in batches...")
        parquet_file = pq.ParquetFile(fp)
        
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["image", "source_class"]):
            df = batch.to_pandas()

            X_part = extract_features_and_labels(df)
            X_parts.append(X_part)

            del df
            del batch

    X = np.concatenate(X_parts, axis=0)
    y = np.zeros(len(X), dtype=np.int64)

    return X, y


def make_loader(X, y, batch_size=BATCH_SIZE):
    dataset = ImageDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, 
                        num_workers=2, persistent_workers=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args = parser.parse_args()

    start_time = time.time()

    os.makedirs(TASK02_DIR, exist_ok=True)

    # 1. Load trained model from ARTIFACTS_DIR
    print("\n=== Loading model ===")
    print_ram_usage("Start")
    model = load_model()

    # 2. Load predict data from PREDICT_DIR (parquet with columns: row_id, image)
    print("\n=== Loading data ===")
    X, y = process_parquet(PREDICT_DIR)
    data_loader = make_loader(X, y)

    print_ram_usage("After loading")

    model.eval()

    all_preds = []
    with torch.no_grad():
        for batch, _ in data_loader:

            if time.time() - start_time > args.timeout_seconds:
                print("\nTimeout reached, stopping predicting.")
                break

            batch = batch.to(DEVICE)
            predictions = model(batch)
            preds = predictions.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())

    # 3. Write artifacts/task02/predictions.csv (columns: row_id, predicted_label)
    output_file = os.path.join(ARTIFACTS_DIR, "task02/predictions.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    print(f"Predicted total of {len(y)} unlabeled data. Predictions are saved under {output_file} ")
    
    save_predictions(all_preds, save_path=output_file)

    f"\n[predict.py] Done in {time.time() - start_time:.1f}s\n"

if __name__ == "__main__":
    main()
