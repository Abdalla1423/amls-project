"""prepare.py - Data preparation for training.

Note: Do NOT prepare data from data/predict/ (it may change after training).

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

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

TIME_OUT = 600
SEED = 42

IMAGE_SIZE = 64
CROP = False

# Set random seeds for reproducibility
def set_random_seeds(seed=SEED):
    np.random.seed(seed)

# Preprocess a single image by center cropping and padding to 64x64, then convert to CHW format
def preprocess_single_image(img_bytes, crop=CROP, target_h=IMAGE_SIZE, target_w=IMAGE_SIZE):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img_array = np.array(img)

    if not crop:
      img = img.resize((target_h, target_w), Image.BICUBIC)
      img_arr = np.array(img)
      return np.transpose(img_arr, (2, 0, 1))
    
    h, w, _ = img_array.shape

    pad_h = max(0, target_h - h)
    pad_w = max(0, target_w - w)

    if pad_h > 0 or pad_w > 0:
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        img_array = np.pad(img_array, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='reflect')
        h, w, _ = img_array.shape

    left = (w - target_w) // 2
    top = (h - target_h) // 2
    cropped = img_array[top:top+target_h, left:left+target_w]

    return np.transpose(cropped, (2, 0, 1))

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
    y = df["source_class"].to_numpy()

    y = np.where(y > 0, 1, y).astype(np.int8)

    return X, y

def process_and_save_npy(input_path, output_path, batch_size=500, balancing=True):

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

    if balancing:
        idx_0 = np.where(y == 0)[0]
        idx_1 = np.where(y == 1)[0]
        min_class_size = min(len(idx_0), len(idx_1))
        idx_0_sampled = np.random.choice(idx_0, size=min_class_size, replace=False)
        idx_1_sampled = np.random.choice(idx_1, size=min_class_size, replace=False)
        balanced_indices = np.concatenate([idx_0_sampled, idx_1_sampled])

        np.random.shuffle(balanced_indices)

        X = X[balanced_indices]
        y = y[balanced_indices]

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
        (os.path.join(ARTIFACTS_DIR, "task01"), os.path.join(ARTIFACTS_DIR, "task02/prepared_training_data"), False),
        (os.path.join(DATA_DIR, "calibration"), os.path.join(ARTIFACTS_DIR, "task02/prepared_calibration_data"), False),
        (os.path.join(DATA_DIR, "validation"), os.path.join(ARTIFACTS_DIR, "task02/prepared_validation_data"), False),
        (os.path.join(DATA_DIR, "calibration_augmented"), os.path.join(ARTIFACTS_DIR, "task03/prepared_calibration_augmented_data"), False),
        (os.path.join(DATA_DIR, "validation_augmented"), os.path.join(ARTIFACTS_DIR, "task03/prepared_validation_augmented_data"), False)
    ]

    for input_dir, output_p, with_balancing in datasets:
        if time.time() - start_time > args.timeout_seconds:
            print(f"\n[Timeout] Reached execution limit of {args.timeout_seconds} seconds.")
            sys.exit(1)

        process_and_save_npy(input_dir, output_p, balancing=with_balancing)

    print(f"\n [prepare.py] Total time invested in preparation: {time.time() - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
