"""
Task 1.2 - Data Preparation

Loads the cleaned training data and calibration/validation splits,
converts images to numpy arrays, and saves them as .npz files for
fast loading during training.

Note: Does NOT prepare data from data/predict/ (as it may change after training).

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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
CLEANED_PATH = os.path.join(ARTIFACTS_DIR, "cleaned_train.parquet")

TARGET_SIZE = (128, 128)


def timeout_handler(signum, frame):
    print("[prepare.py] Timeout reached – saving progress and exiting.")
    sys.exit(0)


def load_parquet_dir(directory: str) -> pd.DataFrame:
    """Load all parquet files from a directory into a single DataFrame."""
    frames = []
    if not os.path.isdir(directory):
        print(f"Warning: directory {directory} not found.")
        return pd.DataFrame()
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".parquet"):
            fpath = os.path.join(directory, fname)
            frames.append(pd.read_parquet(fpath))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def images_to_arrays(df: pd.DataFrame, target_size: tuple[int, int]) -> np.ndarray:
    """Convert image bytes column to a numpy array of shape (N, C, H, W), float32, normalized."""
    arrays = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Converting images"):
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        img = img.resize(target_size, Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0  # (H, W, 3)
        arr = arr.transpose(2, 0, 1)  # (3, H, W) — channels first
        arrays.append(arr)
    return np.stack(arrays, axis=0)


def prepare_split(df: pd.DataFrame, name: str, target_size: tuple[int, int]):
    """Prepare a labeled data split: convert to arrays and save as .npz."""
    if df.empty:
        print(f"  Skipping {name} (empty).")
        return

    # Binary labels
    labels = (df["source_class"] > 0).astype(np.int8).values

    print(f"  Preparing {name}: {len(df)} samples...")
    X = images_to_arrays(df, target_size)

    out_path = os.path.join(ARTIFACTS_DIR, f"{name}.npz")
    np.savez_compressed(out_path, X=X, y=labels)
    print(f"  Saved {out_path}  (X: {X.shape}, y: {labels.shape})")


def prepare_cleaned_train(target_size: tuple[int, int]):
    """Prepare the cleaned training data (already has 'label' column)."""
    print("Loading cleaned training data...")
    df = pd.read_parquet(CLEANED_PATH)
    if df.empty:
        print("ERROR: Cleaned training data is empty. Run clean.py first.")
        sys.exit(1)

    labels = df["label"].values.astype(np.int8)
    print(f"  Preparing cleaned train: {len(df)} samples...")
    X = images_to_arrays(df, target_size)

    out_path = os.path.join(ARTIFACTS_DIR, "train.npz")
    np.savez_compressed(out_path, X=X, y=labels)
    print(f"  Saved {out_path}  (X: {X.shape}, y: {labels.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, args.timeout_seconds - 30))

    start_time = time.time()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # 1. Prepare cleaned training data
    prepare_cleaned_train(TARGET_SIZE)

    # 2. Prepare calibration split
    cal_df = load_parquet_dir(os.path.join(DATA_DIR, "calibration"))
    prepare_split(cal_df, "calibration", TARGET_SIZE)

    # 3. Prepare calibration_augmented split
    cal_aug_df = load_parquet_dir(os.path.join(DATA_DIR, "calibration_augmented"))
    prepare_split(cal_aug_df, "calibration_augmented", TARGET_SIZE)

    # 4. Prepare validation split
    val_df = load_parquet_dir(os.path.join(DATA_DIR, "validation"))
    prepare_split(val_df, "validation", TARGET_SIZE)

    # 5. Prepare validation_augmented split
    val_aug_df = load_parquet_dir(os.path.join(DATA_DIR, "validation_augmented"))
    prepare_split(val_aug_df, "validation_augmented", TARGET_SIZE)

    elapsed = time.time() - start_time
    print(f"\n[prepare.py] Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
