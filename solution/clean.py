"""clean.py - Dataset exploration and cleaning.

Usage: python clean.py --timeout_seconds 600
"""

import argparse
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TRAIN_DIR = os.path.join(DATA_DIR, "train")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # TODO: Load training data from TRAIN_DIR (parquet files with columns: image, source_class)
    # TODO: Explore class distribution, image-size distribution, descriptive statistics
    # TODO: Build a deterministic cleaning pipeline
    # TODO: Save cleaned data to ARTIFACTS_DIR


if __name__ == "__main__":
    main()
