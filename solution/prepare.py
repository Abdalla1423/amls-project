"""prepare.py - Data preparation for training.

Note: Do NOT prepare data from data/predict/ (it may change after training).

Usage: python prepare.py --timeout_seconds 600
"""

import argparse
import os
import sys

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # TODO: Load cleaned training data from ARTIFACTS_DIR
    # TODO: Load calibration, calibration_augmented, validation, validation_augmented from DATA_DIR
    # TODO: Convert image bytes to arrays/tensors suitable for training
    # TODO: Save prepared data to ARTIFACTS_DIR


if __name__ == "__main__":
    main()
