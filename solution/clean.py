"""clean.py - Dataset exploration and cleaning.

Usage: python clean.py --timeout_seconds 600
"""

import argparse
import os
import sys
import pandas as pd
from PIL import Image
import io
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import imagehash
import pyarrow as pa
import pyarrow.parquet as pq

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TRAIN_DIR = os.path.join(DATA_DIR, "train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    clean_data([os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR)], os.path.join(ARTIFACTS_DIR, "task01/training_dataset.parquet"))


def clean_data(files, save_path):
    def process_image(image_bytes, size=(224,224)):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Hash before resize
        img_hash = imagehash.dhash(img)

        # Resize
        img = img.resize(size, Image.Resampling.LANCZOS)

        buffer = io.BytesIO()
        img.save(buffer, format="JPEG")

        return img_hash, buffer.getvalue()

    writer = None
    seen_hashes = set()

    for f in files:
        df = pd.read_parquet(f)

        processed_rows = []

        for row in df.itertuples(index=False):
            img_hash, resized_img = process_image(row.image)

            if img_hash in seen_hashes:
                continue

            seen_hashes.add(img_hash)

            row_dict = row._asdict()
            row_dict["image"] = resized_img
            processed_rows.append(row_dict)

        if processed_rows:
            df_filtered = pd.DataFrame(processed_rows)
            table = pa.Table.from_pandas(df_filtered)

            if writer is None:
                writer = pq.ParquetWriter(save_path, table.schema)

            writer.write_table(table)

    if writer:
        writer.close()

def explore_data(path):
    files = os.listdir(path)
    class_shapes = defaultdict(list)
    class_formats = defaultdict(list)

    for f in files:
        df = pd.read_parquet(os.path.join(path, f))

        for _, row in df.iterrows():
            img = Image.open(io.BytesIO(row["image"]))
            label = row["source_class"]

            class_shapes[label].append(img.size)
            class_formats[label].append(img.mode)

    classes = sorted(class_shapes.keys())

    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 5), sharey=True)

    if len(classes) == 1:
        axes = [axes]

    for ax, label in zip(axes, classes):
        widths = [s[0] for s in class_shapes[label]]
        heights = [s[1] for s in class_shapes[label]]

        ax.scatter(widths, heights, alpha=0.4)
        ax.set_title(f"Class {label}")
        ax.set_xlabel("Width")
        ax.set_ylabel("Height")

    plt.suptitle("Resolution Distribution by Class")
    plt.tight_layout()
    plt.show()


    fig, axes = plt.subplots(1, len(classes), figsize=(6 * len(classes), 4), sharey=True)

    if len(classes) == 1:
        axes = [axes]

    for ax, label in zip(axes, classes):
        aspect_ratios = [w / h for w, h in class_shapes[label]]

        ax.hist(aspect_ratios, bins=30)
        ax.set_title(f"Class {label}")
        ax.set_xlabel("Aspect Ratio")
        ax.set_ylabel("Count")

    plt.suptitle("Aspect Ratio Distribution by Class")
    plt.tight_layout()
    plt.show()
    for label in sorted(class_shapes.keys()):
        widths = [s[0] for s in class_shapes[label]]
        heights = [s[1] for s in class_shapes[label]]

        print(f"\nClass {label}")
        print(f"Count: {len(widths)}")
        print(f"Avg Width: {sum(widths)/len(widths):.2f}")
        print(f"Avg Height: {sum(heights)/len(heights):.2f}")
        print(f"Formats: {Counter(class_formats[label])}")


def plot_random_images(path):
    fig, axes = plt.subplots(3, 3, figsize=(10,10))

    df = pd.read_parquet(path)
    samples = df.sample(9)

    for ax, (_, row) in zip(axes.flatten(), samples.iterrows()):
        img = Image.open(io.BytesIO(row["image"]))
        ax.imshow(img)
        ax.set_title(f"Class: {row['source_class']}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()

def detect_duplicates(path, files):
    hash_map = {}
    for f in files:
        df = pd.read_parquet(os.path.join(path, f))
        for idx, row in df.iterrows():
            img = Image.open(io.BytesIO(row["image"])) 
            img_hash = str(imagehash.dhash(img)) 
            image_id = f"{f}_{idx}" 
            if img_hash in hash_map: 
                hash_map[img_hash].append(image_id) 
            else: 
                hash_map[img_hash] = [image_id]

    duplicates = {k: v for k, v in hash_map.items() if len(v) > 1}
    print(f"Found {len(duplicates)} groups of duplicate images.")

    def get_image_from_id(image_id, base_path=TRAIN_DIR):
        filename, idx = image_id.rsplit('_', 1)
        df = pd.read_parquet(os.path.join(base_path, filename))
        row = df.iloc[int(idx)]
        return (Image.open(io.BytesIO(row["image"])), row["source_class"])

    # Plotting
    fig, axes = plt.subplots(5, 2, figsize=(10, 20))
    plt.subplots_adjust(hspace=0.4)

    for i, (img_hash, ids) in enumerate(duplicates.items()):
        if i == 5:
            break
        img1,label1 = get_image_from_id(ids[0])
        img2,label2 = get_image_from_id(ids[1])

        axes[i, 0].imshow(img1)
        axes[i, 0].set_title(f"Hash: {img_hash}\nID: {ids[0]}\nLabel: {label1}")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(img2)
        axes[i, 1].set_title(f"Duplicate Match\nID: {ids[1]}\nLabel: {label2}")
        axes[i, 1].axis('off')

    plt.show()

if __name__ == "__main__":
    main()

    # plot_random_images(os.path.join(ARTIFACTS_DIR, "task01/training_dataset.parquet"))

    # explore_data(TRAIN_DIR)

    # detect_duplicates(TRAIN_DIR, os.listdir(TRAIN_DIR))
