import argparse
import io
import json
import os
import signal
import sys
import time
from collections import Counter

import pandas as pd
from PIL import Image
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
CLEANED_PATH = os.path.join(ARTIFACTS_DIR, "cleaned_train.parquet")
STATS_PATH = os.path.join(ARTIFACTS_DIR, "exploration_stats.json")

TARGET_SIZE = (64, 64)

def timeout_handler(signum, frame):
    print("[clean.py] Timeout reached – exiting.")
    sys.exit(0)

def load_data(data_dir):
    frames = []
    for frame in sorted(os.listdir(data_dir)):
        if frame.endswith(".parquet"):
            frames.append(pd.read_parquet(os.path.join(data_dir, frame)))
    return pd.concat(frames, ignore_index=True)

def explore(df):
    n = len(df)

    # Class distribution
    class_cnts = df["source_class"].value_counts().sort_index()
    print("\n=== Source-class distribution ===")
    for imgcls, cnt in class_cnts.items():
        print(f" Class {imgcls}: {cnt}  ({100*cnt/n:.1f}%)")
    stats = {"class_counts": class_cnts.to_dict()}

    binary_stats = (df["source_class"] > 0).astype(int)
    n_real, n_ai = (binary_stats == 0).sum(), (binary_stats == 1).sum()
    print(f"\n=== Binary distribution ===")
    print(f" Real: {n_real}  ({100*n_real/n:.1f}%)")
    print(f" AI: {n_ai}  ({100*n_ai/n:.1f}%)")
    stats["binary_counts"] = {"real": int(n_real), "ai": int(n_ai)}

    # Image dimensions per class
    print("\n=== Image dimensions per class ===")
    dim_stats = {}
    for cls in sorted(df["source_class"].unique()):
        cls_imgs = df[df["source_class"] == cls]
        widths = []
        heights = []
        sizes = []

        for _, row in cls_imgs.iterrows():
            img = Image.open(io.BytesIO(row["image"]))
            sizes.append((img.width, img.height))
            widths.append(img.width)
            heights.append(img.height)
        print(f" Class {cls}: width {min(widths)}-{max(widths)}, height {min(heights)}-{max(heights)}, sizes {Counter(sizes).most_common(3)}")
        dim_stats[int(cls)] = {
            "w_min": int(min(widths)), "w_max": int(max(widths)),
            "h_min": int(min(heights)), "h_max": int(max(heights)),
        }
    stats["dimension_stats"] = dim_stats

    # Byte sizes
    print("\n=== Byte sizes per class ===")
    byte_stats = {}
    for cls in sorted(df["source_class"].unique()):
        cls_bytes = df[df["source_class"] == cls]["image"].apply(len)
        print(f" Class {cls}: size {min(cls_bytes)}-{max(cls_bytes)} bytes, mean {cls_bytes.mean():.1f} bytes")
        byte_stats[int(cls)] = {
            "size_min": int(min(cls_bytes)), "size_max": int(max(cls_bytes)), "size_mean": float(cls_bytes.mean())
        }
    stats["byte_stats"] = byte_stats

    # Duplicate images
    print("\n=== Duplicate images ===")
    img_hashes = df["image"].apply(lambda x: hash(bytes(x)))
    n_dups = img_hashes.duplicated().sum()
    print(f" Found {n_dups} duplicate images.")
    stats["duplicate_count"] = int(n_dups)

    # TODO: delete before submission
    # Class leaking features
    print("\n=== Class-leaking features ===")
    print("  * Resolution: Real images have variable sizes, AI images have fixed sizes are fixed (320x320 or 270x270).")
    print("  * Aspect ratio: Real images have variable aspect ratios, AI images are always square (1:1).")
    print("  * Byte size: Real images are larger (~51KB mean), AI images smaller (~18-32KB mean)")

    return stats


def clean(df):

    df = df.copy()

    # Creat binary label
    df["label"] = (df["source_class"] > 0).astype(int)

    # Remove duplicates
    img_hashes = df["image"].apply(lambda x: hash(bytes(x)))
    n_before = len(df)
    df = df[~img_hashes.duplicated()].reset_index(drop=True)
    print(f"  Removed {n_before - len(df)} duplicates -> {len(df)} remaining")

    # Resize to uniform sizes and re-encode as PNG
    cleaned_images = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Cleaning images"):
        img = Image.open(io.BytesIO(row["image"])).convert("RGB")
        img = img.resize(TARGET_SIZE, resample=Image.BICUBIC)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        cleaned_images.append(buf.getvalue())

    df["image"] = cleaned_images

    # Keep only relevant columns
    df = df[["image", "label"]].reset_index(drop=True)

    print(f"  Cleaned {len(df)} images to {TARGET_SIZE[0]}x{TARGET_SIZE[1]} PNG format.")

    return df
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()

    # TODO: Delete before submission
    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(args.timeout_seconds)

    start_time = time.time()
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # Load data
    print("Loading data...")
    df = load_data(TRAIN_DIR)
    print(f"Loaded {len(df)} samples.")

    # Explore data
    stats = explore(df)
    with open(STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)

    # Clean data
    cleaned_df = clean(df)
    cleaned_df.to_parquet(CLEANED_PATH, index=False)
    print(f"Saved cleaned data to {CLEANED_PATH}")

    print(f"\n[clean.py] Done in {time.time() - start_time:.1f}s")

if __name__ == "__main__":
    main()