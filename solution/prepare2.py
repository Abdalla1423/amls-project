import argparse
import io
import os
import signal
import sys
import time

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

TARGET_SIZE = (64, 64)

def timeout_handler(signum, frame):
    print("[prepare.py] Timeout – exiting.")
    sys.exit(0)

def load_parquet_dir(directory):
    """Load all parquet files from a directory into one DataFrame."""
    frames = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(directory, fname)))
    return pd.concat(frames, ignore_index=True)

def images_to_array(df, target_size):
    """Convert image-bytes column to float32 array (N, 3, H, W) in [0, 1]."""
    arrays = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting"):
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        img = img.resize(target_size, Image.BICUBIC)
        img_arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arrays.append(img_arr.transpose(2, 0, 1))           # (3, H, W)
    return np.stack(arrays)

def clean_and_prepare_split(split):
    df_split = load_parquet_dir(os.path.join(DATA_DIR, split))
    if df_split.empty:
        print(f"  Warning: {split} split is empty, skipping.")
        return
    labels_split = (df_split["source_class"] > 0).astype(np.int8).values
    X_split = images_to_array(df_split, TARGET_SIZE)
    np.savez_compressed(os.path.join(ARTIFACTS_DIR, f"{split}.npz"), X=X_split, y=labels_split)
    print(f"  {split}: {X_split.shape}  (real={int((labels_split == 0).sum())}, ai={int((labels_split == 1).sum())})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout_seconds)
    
    start_time = time.time()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load cleaned training data
    print("Loading cleaned training data...")
    cleaned_path = os.path.join(ARTIFACTS_DIR, "cleaned_train.parquet")
    df_train = pd.read_parquet(cleaned_path)
    labels_train = df_train["label"].values.astype(np.int8)

    # Convert images to array
    print("Converting images to arrays...")
    X_train = images_to_array(df_train, TARGET_SIZE)
    np.savez_compressed(os.path.join(ARTIFACTS_DIR, "train.npz"), X=X_train, y=labels_train)
    print(f"  train: {X_train.shape}  (real={int((labels_train == 0).sum())}, ai={int((labels_train == 1).sum())})")

    # Convert validation and calibration splits
    for split in ["validation", "calibration", "validation_augmented", "calibration_augmented"]:
        print(f"Processing {split} split...")
        clean_and_prepare_split(split)

    print(f"\n[prepare.py] Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()