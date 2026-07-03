"""prepare.py - Data preparation for training.

Usage: python prepare.py --timeout_seconds 600
"""

import argparse
import os
import sys
import glob
import time 
import numpy as np
from PIL import Image
import pyarrow.parquet as pq
import io

# Paths and global variables

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

TIME_OUT = 600
SEED = 42

IMAGE_SIZE = 64

# Set random seeds for reproducibility
def set_random_seeds(seed=SEED):
    np.random.seed(seed)

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
        processed_images.append(processed_img)

    X = np.stack(processed_images)
    y = df["source_class"].to_numpy()

    y = np.where(y > 0, 1, y).astype(np.int8)

    return X, y

def process_and_save_npy(input_path, output_path, batch_size=500):

    x_path = f"{output_path}_X.npy"
    y_path = f"{output_path}_y.npy"

    print(f"Processing {input_path} and saving to:\n  -> {x_path}\n  -> {y_path}...")
    
    file_paths = sorted(glob.glob(os.path.join(input_path, "*.parquet")))

    if not file_paths:
        print(f"No parquet files found in {input_path}")
        return
    
    X_parts = []
    y_parts = []

    for fp in file_paths:
        print(f"Reading {fp} in batches...")
        parquet_file = pq.ParquetFile(fp)
        
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["image", "source_class"]):
            df = batch.to_pandas()

            X_part, y_part = extract_features_and_labels(df)
            X_parts.append(X_part)
            y_parts.append(y_part)

            del df
            del batch

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    os.makedirs(os.path.dirname(x_path), exist_ok=True)  
    os.makedirs(os.path.dirname(y_path), exist_ok=True)  

    np.save(x_path, X)
    np.save(y_path, y)

    print(f"Successfully saved {len(X)} samples.")

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    set_random_seeds()

    datasets = [
        (os.path.join(ARTIFACTS_DIR, "task01"), os.path.join(ARTIFACTS_DIR, "task02/training_data")),
        (os.path.join(DATA_DIR, "calibration"), os.path.join(ARTIFACTS_DIR, "task02/calibration_data")),
        (os.path.join(DATA_DIR, "validation"), os.path.join(ARTIFACTS_DIR, "task02/validation_data")),
        (os.path.join(DATA_DIR, "calibration_augmented"), os.path.join(ARTIFACTS_DIR, "task03/calibration_augmented_data")),
        (os.path.join(DATA_DIR, "validation_augmented"), os.path.join(ARTIFACTS_DIR, "task03/validation_augmented_data"))
    ]

    for input_dir, output_p in datasets:
        if time.time() - start_time > args.timeout_seconds:
            print(f"\n[Timeout] Reached execution limit of {args.timeout_seconds} seconds.")
            sys.exit(1)

        process_and_save_npy(input_dir, output_p)

    print(f"\n[prepare.py] Done in {time.time() - start_time:.1f}s\n")

if __name__ == "__main__":
    main()
