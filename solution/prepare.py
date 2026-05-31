"""prepare.py - Data preparation for training (Task 1.2).

Loads the cleaned training data and calibration/validation splits,
converts image bytes to numpy arrays (float32, channels-first),
and saves as .npz files for fast loading during training.

Note: Does NOT prepare data from data/predict/ (it may change after training).

Usage: python prepare.py --timeout_seconds 600
"""

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

TARGET_SIZE = (128, 128)


def timeout_handler(signum, frame):
    print("[prepare.py] Timeout – exiting.")
    sys.exit(0)


def load_parquet_dir(directory):
    """Load all parquet files from a directory into one DataFrame."""
    frames = []
    if not os.path.isdir(directory):
        print(f"  Warning: {directory} not found, skipping.")
        return pd.DataFrame()
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(directory, fname)))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def images_to_array(df, target_size):
    """Convert image-bytes column to float32 array (N, 3, H, W) in [0, 1]."""
    arrays = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="    Converting"):
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arrays.append(arr.transpose(2, 0, 1))           # (3, H, W)
    return np.stack(arrays)


def prepare_labeled_split(directory, name, target_size):
    """Prepare a labeled split (has source_class column) and save as .npz."""
    df = load_parquet_dir(directory)
    if df.empty:
        print(f"  {name}: empty, skipping.")
        return
    labels = (df["source_class"] > 0).astype(np.int8).values
    print(f"  {name}: {len(df)} samples  (real={int((labels == 0).sum())}, "
          f"ai={int((labels == 1).sum())})")
    X = images_to_array(df, target_size)
    out = os.path.join(ARTIFACTS_DIR, f"{name}.npz")
    np.savez_compressed(out, X=X, y=labels)
    print(f"    -> {out}  {X.shape}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, args.timeout_seconds - 30))

    start = time.time()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Cleaned training data (from clean.py output)
    cleaned_path = os.path.join(ARTIFACTS_DIR, "cleaned_train.parquet")
    print("Preparing cleaned training data...")
    df_train = pd.read_parquet(cleaned_path)
    labels_train = df_train["label"].values.astype(np.int8)
    X_train = images_to_array(df_train, TARGET_SIZE)
    np.savez_compressed(os.path.join(ARTIFACTS_DIR, "train.npz"),
                        X=X_train, y=labels_train)
    print(f"  train: {X_train.shape}  (real={int((labels_train == 0).sum())}, "
          f"ai={int((labels_train == 1).sum())})")

    # 2. Calibration and validation splits
    for name in ["calibration", "calibration_augmented",
                 "validation", "validation_augmented"]:
        prepare_labeled_split(os.path.join(DATA_DIR, name), name, TARGET_SIZE)

    print(f"\n[prepare.py] Done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
