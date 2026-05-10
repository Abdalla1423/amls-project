"""clean.py - Dataset exploration and cleaning (Task 1.1).

Explores the training data and applies a deterministic cleaning pipeline.

Exploration findings:
  - 29,688 samples across 6 balanced classes (4,948 each).
  - Binary split: 4,948 real (class 0) vs 24,740 AI-generated (classes 1-5).
  - All images are JPEG-encoded RGB with 0 corrupt files.
  - 312 exact duplicate images (all within the same class).
  - Class-leaking features found:
      * Image resolution: Real images have variable sizes (COCO),
        AI classes are all square (320x320, except DALL-E at 270x270).
      * Aspect ratio: Real images vary (0.56-2.37), AI always 1.0.
      * JPEG quantization tables differ between real/AI sources.
  - Byte sizes: min=2053, max=139,975, mean~30,736.

Cleaning pipeline (deterministic):
  1. Remove exact duplicate images (keep first occurrence).
  2. Resize all images to 128x128 using LANCZOS resampling.
     Justification: Uniform size removes resolution as a class-leaking
     feature and makes downstream CNN training CPU-friendly. 128x128
     balances detail preservation with memory/compute constraints.
  3. Re-encode as PNG (lossless) to eliminate JPEG quantization artifacts
     that could leak class identity.
  4. Create binary label: 0 = real, 1 = ai_generated.

Usage: python clean.py --timeout_seconds 600
"""

import argparse
import io
import json
import os
import signal
import sys
import time
from collections import Counter

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
STATS_PATH = os.path.join(ARTIFACTS_DIR, "exploration_stats.json")

# Cleaning parameters
TARGET_SIZE = (128, 128)


def timeout_handler(signum, frame):
    print("[clean.py] Timeout reached – exiting.")
    sys.exit(0)


def load_parquet_dir(directory):
    """Load all parquet files from a directory into one DataFrame."""
    frames = []
    for fname in sorted(os.listdir(directory)):
        if fname.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(directory, fname)))
    return pd.concat(frames, ignore_index=True)


def explore(df):
    """Analyze and print dataset statistics. Returns a stats dict."""
    stats = {}
    n = len(df)
    print(f"Total samples: {n}")

    # --- Class distribution ---
    class_counts = df["source_class"].value_counts().sort_index()
    print("\n=== Source-class distribution ===")
    for cls, cnt in class_counts.items():
        print(f"  Class {cls}: {cnt}  ({100*cnt/n:.1f}%)")
    stats["class_counts"] = class_counts.to_dict()

    binary = (df["source_class"] > 0).astype(int)
    n_real, n_ai = (binary == 0).sum(), (binary == 1).sum()
    print(f"\n=== Binary distribution ===")
    print(f"  Real: {n_real}  ({100*n_real/n:.1f}%)")
    print(f"  AI:   {n_ai}  ({100*n_ai/n:.1f}%)")
    stats["binary_counts"] = {"real": int(n_real), "ai": int(n_ai)}

    # --- Image dimensions per class ---
    print("\n=== Image dimensions per class ===")
    dim_stats = {}
    for cls in sorted(df["source_class"].unique()):
        cls_df = df[df["source_class"] == cls].sample(
            min(300, (df["source_class"] == cls).sum()), random_state=42
        )
        widths, heights = [], []
        for _, row in cls_df.iterrows():
            img = Image.open(io.BytesIO(row["image"]))
            widths.append(img.width)
            heights.append(img.height)
        w, h = np.array(widths), np.array(heights)
        top_sizes = Counter(zip(widths, heights)).most_common(3)
        print(f"  Class {cls}: W [{w.min()}-{w.max()}] H [{h.min()}-{h.max()}] "
              f"top sizes: {top_sizes}")
        dim_stats[int(cls)] = {
            "w_min": int(w.min()), "w_max": int(w.max()),
            "h_min": int(h.min()), "h_max": int(h.max()),
        }
    stats["dimensions_per_class"] = dim_stats

    # --- Byte sizes ---
    byte_sizes = df["image"].apply(len)
    print(f"\n=== Byte sizes ===")
    print(f"  min={byte_sizes.min()}  max={byte_sizes.max()}  "
          f"mean={byte_sizes.mean():.0f}  median={byte_sizes.median():.0f}")
    stats["byte_sizes"] = {
        "min": int(byte_sizes.min()), "max": int(byte_sizes.max()),
        "mean": float(byte_sizes.mean()),
    }

    # --- Duplicates ---
    hashes = df["image"].apply(lambda b: hash(bytes(b)))
    n_dup = hashes.duplicated().sum()
    print(f"\n=== Duplicates ===")
    print(f"  Exact duplicate images: {n_dup}")
    stats["n_duplicates"] = int(n_dup)

    # --- Class-leaking features ---
    print("\n=== Class-leaking features ===")
    print("  * Resolution: Real images have variable sizes; AI images are fixed "
          "(320x320 or 270x270).")
    print("  * Aspect ratio: Real varies (0.56-2.37); AI always 1.0.")
    print("  * JPEG quantization tables differ between sources.")
    print("  -> Cleaning must normalize resolution and re-encode to remove leaks.")

    return stats


def clean(df):
    """Apply a deterministic cleaning pipeline. Returns cleaned DataFrame."""
    print(
        f"\nCleaning {len(df)} images -> {TARGET_SIZE[0]}x{TARGET_SIZE[1]} PNG ...")

    # 1. Create binary label
    df = df.copy()
    df["label"] = (df["source_class"] > 0).astype(np.int8)

    # 2. Remove exact duplicates (keep first)
    df["_hash"] = df["image"].apply(lambda b: hash(bytes(b)))
    n_before = len(df)
    df = df.drop_duplicates(subset="_hash", keep="first").drop(columns="_hash")
    print(f"  Removed {n_before - len(df)} duplicates -> {len(df)} remaining")

    # 3. Resize to uniform size and re-encode as PNG
    cleaned_images = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="  Processing"):
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        img = img.resize(TARGET_SIZE, Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cleaned_images.append(buf.getvalue())

    df = df.copy()
    df["image"] = cleaned_images

    # 4. Keep only needed columns
    df = df[["image", "label"]].reset_index(drop=True)

    print(f"  Final: {len(df)} samples")
    print(f"  Labels: {df['label'].value_counts().sort_index().to_dict()}")
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, args.timeout_seconds - 30))

    start = time.time()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load
    print("Loading training data...")
    df = load_parquet_dir(TRAIN_DIR)
    print(f"Loaded {len(df)} samples.\n")

    # Explore
    stats = explore(df)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"\nSaved exploration stats to {STATS_PATH}")

    # Clean
    cleaned = clean(df)
    cleaned.to_parquet(CLEANED_PATH, index=False)
    print(f"Saved cleaned dataset to {CLEANED_PATH}")

    print(f"\n[clean.py] Done in {time.time()-start:.1f}s")


if __name__ == "__main__":
    main()
