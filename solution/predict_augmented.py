"""predict_augmented.py - Inference for Task 3 (augmented model).

Outputs: artifacts/task03/predictions.csv (columns: row_id, predicted_label)

Usage: python predict_augmented.py --timeout_seconds 600
"""

import argparse
import os

from common import ARTIFACTS_DIR
from predict import run_inference

TIME_OUT = 600

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
TASK03_DIR = os.path.join(ARTIFACTS_DIR, "task03")
PREDICT_DIR = os.path.join(DATA_DIR, "predict")
PREDICTIONS = os.path.join(TASK03_DIR, "predictions.csv")
CHECKPOINT_PATH = os.path.join(TASK03_DIR, "best_model.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args = parser.parse_args()
    run_inference(CHECKPOINT_PATH, PREDICT_DIR, PREDICTIONS, args.timeout_seconds)


if __name__ == "__main__":
    main()
