"""predict_augmented.py - Inference for Task 3 (augmented model).

Outputs: artifacts/task03/predictions.csv (columns: row_id, predicted_label)

Usage: python predict_augmented.py --timeout_seconds 600
"""

import argparse
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK03_DIR = os.path.join(ARTIFACTS_DIR, "task03")
PREDICT_DIR = os.path.join(DATA_DIR, "predict")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(TASK03_DIR, exist_ok=True)

    # TODO: Load augmented model from ARTIFACTS_DIR
    # TODO: Load predict data from PREDICT_DIR (parquet with columns: row_id, image)
    # TODO: Run inference and apply calibrated threshold
    # TODO: Write artifacts/task03/predictions.csv (columns: row_id, predicted_label)


if __name__ == "__main__":
    main()
