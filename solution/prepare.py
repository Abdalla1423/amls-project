"""prepare.py - Data preparation for training.

Usage: python prepare.py --timeout_seconds 600
"""

import argparse
import glob
import io
import os
import random
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

TIME_OUT = 600
SEED = 42
IMAGE_SIZE = 64
BATCH_SIZE = 500

IMAGE_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
IMAGE_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)


def set_random_seeds(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def preprocess_single_image(img_bytes, target_h=IMAGE_SIZE, target_w=IMAGE_SIZE):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((target_h, target_w), Image.BICUBIC)
    img_arr = np.array(img)
    return np.transpose(img_arr, (2, 0, 1))


def extract_features_and_labels(df):
    processed_images = []
    labels = []
    for row in df.itertuples(index=False):
        try:
            processed_img = preprocess_single_image(row.image)
        except UnidentifiedImageError as e:
            print(f"Skipping unreadable image: {e}")
            continue

        processed_img = processed_img.astype(np.float32) / 255.0
        processed_img = (processed_img - IMAGE_MEAN) / IMAGE_STD
        processed_images.append(processed_img)
        labels.append(row.source_class)

    X = np.stack(processed_images)
    y = np.array(labels)
    y = np.where(y > 0, 1, y).astype(np.int8)
    return X, y


def process_and_save_npy(input_path, output_path, batch_size=BATCH_SIZE):
    x_path = f"{output_path}_X.npy"
    y_path = f"{output_path}_y.npy"

    print(f"Processing {input_path} and saving to:\n  -> {x_path}\n  -> {y_path}...")
    file_paths = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    if not file_paths:
        print(f"No parquet files found in {input_path}")
        return

    X_parts, y_parts = [], []
    for fp in file_paths:
        print(f"Reading {fp} in batches...")
        try:
            parquet_file = pq.ParquetFile(fp)
        except Exception as e:
            print(f"Could not open {fp}: {e}")
            continue

        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["image", "source_class"]):
            df = batch.to_pandas()
            X_part, y_part = extract_features_and_labels(df)
            X_parts.append(X_part)
            y_parts.append(y_part)
            del df, batch

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)

    os.makedirs(os.path.dirname(x_path), exist_ok=True)
    os.makedirs(os.path.dirname(y_path), exist_ok=True)
    np.save(x_path, X)
    np.save(y_path, y)
    print(f"Successfully saved {len(X)} samples.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args = parser.parse_args()

    start_time = time.time()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    set_random_seeds()

    datasets = [
        (os.path.join(ARTIFACTS_DIR, "task01"), os.path.join(ARTIFACTS_DIR, "task02/training_data")),
        (os.path.join(DATA_DIR, "calibration"), os.path.join(ARTIFACTS_DIR, "task02/calibration_data")),
        (os.path.join(DATA_DIR, "validation"), os.path.join(ARTIFACTS_DIR, "task02/validation_data")),
        (os.path.join(DATA_DIR, "calibration_augmented"), os.path.join(ARTIFACTS_DIR, "task03/calibration_augmented_data")),
        (os.path.join(DATA_DIR, "validation_augmented"), os.path.join(ARTIFACTS_DIR, "task03/validation_augmented_data")),
    ]

    for input_dir, output_p in datasets:
        if time.time() - start_time > args.timeout_seconds:
            print(f"\n[Timeout] Reached execution limit of {args.timeout_seconds} seconds.")
            return
        process_and_save_npy(input_dir, output_p)

    print(f"\n[prepare.py] Done in {time.time() - start_time:.1f}s\n")


if __name__ == "__main__":
    main()
