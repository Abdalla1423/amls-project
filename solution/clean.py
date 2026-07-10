"""clean.py - Dataset exploration and cleaning.

Usage: python clean.py --timeout_seconds 600
"""

import argparse
import os
import time
import pandas as pd
from PIL import Image, UnidentifiedImageError
import io
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import imagehash
import pyarrow as pa
import pyarrow.parquet as pq

# Paths and global variables
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
TASK_DIR = os.path.join(ARTIFACTS_DIR, "task01")

LOG_FILE = os.path.join(ARTIFACTS_DIR, "task01/data_exploration_and_cleaning.txt")
CLEANED_PARQUET = os.path.join(TASK_DIR, "training_dataset.parquet")

TIME_OUT = 600

# Log gathered information to artifacts/task01/data_exploration_and_cleaning.txt
def write_log(total_seen, total_saved, total_duplicates, total_file_errors,
              label_conflicts, class_sizes, class_shapes, class_formats):
    class_0_sizes = class_sizes.get(0, [])

    class_0_count = len(class_0_sizes)
    class_0_average = (sum(class_0_sizes) / class_0_count / 1024) if class_0_count > 0 else 0.0

    class_1_sizes = []
    for label, sizes in class_sizes.items():
        if int(label) > 0:
            class_1_sizes.extend(sizes)
    class_1_count = len(class_1_sizes)
    class_1_average = (sum(class_1_sizes) / class_1_count / 1024) if class_1_count > 0 else 0.0

    total = class_0_count + class_1_count

    with open(LOG_FILE, mode="w") as file:
        file.write("\n=== FINAL SUMMARY ===\n")
        file.write(f"Total samples processed: {total_seen:,}\n")
        file.write(f"Total samples saved:     {total_saved:,}\n")
        file.write(f"Total file errors:       {total_file_errors:,}\n")
        file.write(f"Total duplicates:        {total_duplicates:,}\n")
        file.write(f"Duplicate label conflicts: {label_conflicts:,}\n")

        file.write("\n=== TARGET BINARY CLASS DISTRIBUTION ===\n")
        if total > 0:
            file.write(f"Class 0 (REAL) - Count: {class_0_count} ({class_0_count/total * 100:.1f}%) | Average Size: {class_0_average:.2f} KB\n")
            file.write(f"Class 1 (AI)   - Count: {class_1_count} ({class_1_count/total * 100:.1f}%) | Average Size: {class_1_average:.2f} KB\n")
        else:
            file.write("No samples saved (empty run or timeout before any writes).\n")

        file.write("\n=== CLASS DETAIL & PROPERTY ANALYSIS ===\n")
        for label in sorted(class_shapes.keys()):
            widths = [s[0] for s in class_shapes[label]]
            heights = [s[1] for s in class_shapes[label]]
            sizes = class_sizes[label]
            count = len(widths)
            if count == 0:
                continue

            file.write(f"\nClass {label}\n")
            file.write(f"   Count: {count:,}\n")
            file.write(f"   Avg Size: {sum(sizes)/count/1024:.2f} KB\n")
            file.write(f"   Avg Width: {sum(widths)/count:.2f}\n")
            file.write(f"   Avg Height: {sum(heights)/count:.2f}\n")
            file.write(f"   Formats: {dict(Counter(class_formats[label]))}\n")

# Clean data by removing duplicates, and correcting labels, then save to a single parquet file for training
def clean_data(files, save_path, timeout_seconds):
    start_time = time.time()

    class_shapes = defaultdict(list)
    class_formats = defaultdict(list)
    class_sizes = defaultdict(list)

    def process_image(image_bytes):
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_hash = imagehash.dhash(img)
        return img_hash, img

    writer = None
    hash_to_label = {}
    timeout_triggered = False

    total_seen = 0
    total_saved = 0
    total_duplicates = 0
    total_file_errors = 0
    label_conflicts = 0

    for f in files:
        print(f"Cleaning file {f}")
        if timeout_triggered:
            break

        try:
            df = pd.read_parquet(f)
        except Exception as e:
            print(f" Skipping file {f}, due to {e}")
            continue

        file_seen = 0
        file_saved = 0
        file_duplicates = 0
        file_errors = 0

        processed_rows = []

        for row in df.itertuples(index=False):
            if time.time() - start_time > timeout_seconds:
                print(f"\n[Timeout] Reached execution limit of {timeout_seconds} seconds.")
                timeout_triggered = True
                break

            total_seen += 1
            file_seen += 1

            label = row.source_class
            raw_bytes = row.image

            try:
                img_hash, img = process_image(raw_bytes)
            except(UnidentifiedImageError) as e:
                file_errors += 1
                print(f"Corrupt / Unreadable image skipping: {e}")
                continue

            if img_hash in hash_to_label:
                total_duplicates += 1
                file_duplicates += 1
                if hash_to_label[img_hash] != label:
                    label_conflicts += 1
                continue

            hash_to_label[img_hash] = label

            class_shapes[label].append(img.size)  # (width, height)
            class_formats[label].append(img.mode)
            class_sizes[label].append(len(raw_bytes))

            processed_rows.append(({
                "image": raw_bytes,
                "source_class": label,
                }))

            total_saved += 1
            file_saved += 1

        if processed_rows:
            df_filtered = pd.DataFrame(processed_rows)
            table = pa.Table.from_pandas(df_filtered)

            if writer is None:
                writer = pq.ParquetWriter(save_path, table.schema)

            writer.write_table(table)
        print(
            f"File summary -> "
            f"Seen: {file_seen:,}, "
            f"Saved: {file_saved:,}, "
            f"Duplicates removed: {file_duplicates:,}, "
            f"Total file errors: {file_errors}"
        )

    if writer:
        writer.close()

    write_log(total_seen, total_saved, total_duplicates, total_file_errors,
              label_conflicts, class_sizes, class_shapes, class_formats)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args = parser.parse_args()

    start_time = time.time()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training data directory not found: {TRAIN_DIR}")

    os.makedirs(os.path.dirname(CLEANED_PARQUET), exist_ok=True)

    files = sorted(os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith(".parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {TRAIN_DIR}")
    
    clean_data(files, CLEANED_PARQUET, timeout_seconds=args.timeout_seconds)
    
    print(f"\n[clean.py] Done in {time.time() - start_time:.1f}s\n")


if __name__ == "__main__":
    main()

# Additional function for data exploration 
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

# Additional function to plot random samples from the dataset, for visual inspection of image quality and label correctness
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

# Additional function to detect duplicate images using perceptual hashing, and visualize some examples of duplicates side by side
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
