"""
Task 1.1 - Dataset Exploration and Cleaning

Analyzes the training data (class distribution, image-size distribution,
descriptive statistics) and applies a deterministic cleaning pipeline.

Usage: python clean.py --timeout_seconds 600
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
TRAIN_DIR = os.path.join(DATA_DIR, "train")
CLEANED_PATH = os.path.join(ARTIFACTS_DIR, "cleaned_train.parquet")
STATS_PATH = os.path.join(ARTIFACTS_DIR, "data_stats.csv")

# ---------------------------------------------------------------------------
# Cleaning parameters
# ---------------------------------------------------------------------------
TARGET_SIZE = (128, 128)  # Resize all images to this (CPU-friendly)
MIN_IMAGE_BYTES = 100     # Discard images smaller than this (likely corrupt)


def timeout_handler(signum, frame):
    print("[clean.py] Timeout reached – saving progress and exiting.")
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


def bytes_to_image(image_bytes: bytes) -> Image.Image | None:
    """Safely decode image bytes into a PIL Image."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()
        # Re-open after verify (verify closes the file)
        img = Image.open(io.BytesIO(image_bytes))
        return img.convert("RGB")
    except Exception:
        return None


def explore_dataset(df: pd.DataFrame) -> dict:
    """Compute and print dataset statistics."""
    stats = {}

    # Class distribution (original multi-class)
    print("\n=== Class Distribution (source_class) ===")
    class_counts = df["source_class"].value_counts().sort_index()
    print(class_counts)
    stats["class_counts"] = class_counts.to_dict()

    # Binary class distribution
    binary_labels = (df["source_class"] > 0).astype(int)
    print("\n=== Binary Distribution (0=real, 1=ai_generated) ===")
    print(binary_labels.value_counts().sort_index())

    # Image size distribution (sample for speed)
    sample_n = min(1000, len(df))
    sample_df = df.sample(n=sample_n, random_state=42)
    widths, heights, sizes_bytes = [], [], []

    print(f"\nAnalyzing {sample_n} sample images for size statistics...")
    for _, row in tqdm(sample_df.iterrows(), total=sample_n, desc="Exploring"):
        img_bytes = row["image"]
        sizes_bytes.append(len(img_bytes))
        img = bytes_to_image(img_bytes)
        if img is not None:
            widths.append(img.width)
            heights.append(img.height)

    if widths:
        print(f"\n=== Image Dimensions (sample of {sample_n}) ===")
        print(f"  Width  — min: {min(widths)}, max: {max(widths)}, "
              f"mean: {np.mean(widths):.1f}, median: {np.median(widths):.1f}")
        print(f"  Height — min: {min(heights)}, max: {max(heights)}, "
              f"mean: {np.mean(heights):.1f}, median: {np.median(heights):.1f}")
        stats["width_mean"] = float(np.mean(widths))
        stats["height_mean"] = float(np.mean(heights))

    print(f"\n=== File Sizes (bytes, sample of {sample_n}) ===")
    print(f"  min: {min(sizes_bytes)}, max: {max(sizes_bytes)}, "
          f"mean: {np.mean(sizes_bytes):.0f}")

    # Check for duplicates (by image hash)
    print("\nChecking for exact duplicates (by image bytes hash)...")
    hashes = df["image"].apply(lambda b: hash(bytes(b)))
    n_dup = hashes.duplicated().sum()
    print(f"  Exact duplicate images: {n_dup}")
    stats["n_duplicates"] = int(n_dup)

    return stats


def clean_image(image_bytes: bytes, target_size: tuple[int, int]) -> bytes | None:
    """
    Deterministic cleaning pipeline for a single image:
      1. Skip if too small (likely corrupt).
      2. Decode to RGB.
      3. Resize to target_size using LANCZOS.
      4. Re-encode as PNG for lossless, deterministic storage.
    """
    if len(image_bytes) < MIN_IMAGE_BYTES:
        return None

    img = bytes_to_image(image_bytes)
    if img is None:
        return None

    # Resize deterministically
    img = img.resize(target_size, Image.LANCZOS)

    # Encode as PNG (deterministic, lossless)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the deterministic cleaning pipeline to the full dataset."""
    print(f"\n=== Cleaning {len(df)} images ===")
    print(f"  Target size: {TARGET_SIZE}")

    # Create binary label: 0 = real, 1 = ai_generated
    df = df.copy()
    df["label"] = (df["source_class"] > 0).astype(np.int8)

    cleaned_images = []
    kept_indices = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning"):
        result = clean_image(row["image"], TARGET_SIZE)
        if result is not None:
            cleaned_images.append(result)
            kept_indices.append(idx)

    cleaned_df = df.loc[kept_indices].copy()
    cleaned_df["image"] = cleaned_images

    # Remove exact duplicates (keep first)
    cleaned_df["_hash"] = cleaned_df["image"].apply(lambda b: hash(bytes(b)))
    n_before = len(cleaned_df)
    cleaned_df = cleaned_df.drop_duplicates(subset="_hash", keep="first")
    cleaned_df = cleaned_df.drop(columns=["_hash"])
    n_removed = n_before - len(cleaned_df)
    if n_removed > 0:
        print(f"  Removed {n_removed} duplicate images after cleaning.")

    # Keep only needed columns: image (cleaned) + label
    cleaned_df = cleaned_df[["image", "label"]].reset_index(drop=True)

    print(f"  Kept {len(cleaned_df)} / {len(df)} images after cleaning.")
    print(f"  Label distribution:\n{cleaned_df['label'].value_counts().sort_index()}")

    return cleaned_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    # Set up timeout
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, args.timeout_seconds - 30))  # 30s safety margin

    start_time = time.time()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_df = load_parquet_dir(TRAIN_DIR)
    if train_df.empty:
        print("ERROR: No training data found.")
        sys.exit(1)
    print(f"Loaded {len(train_df)} training samples.")

    # Explore
    stats = explore_dataset(train_df)

    # Save stats
    stats_df = pd.DataFrame([stats])
    stats_df.to_csv(STATS_PATH, index=False)
    print(f"\nSaved dataset statistics to {STATS_PATH}")

    # Clean
    cleaned_df = clean_dataset(train_df)

    # Save cleaned dataset
    cleaned_df.to_parquet(CLEANED_PATH, index=False)
    print(f"Saved cleaned dataset to {CLEANED_PATH}")

    elapsed = time.time() - start_time
    print(f"\n[clean.py] Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
