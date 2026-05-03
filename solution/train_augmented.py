"""train_augmented.py - Augmented/robust model training (Task 3).

Usage: python train_augmented.py --timeout_seconds 1800
"""

import argparse
import os
import sys

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK03_DIR = os.path.join(ARTIFACTS_DIR, "task03")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()

    os.makedirs(TASK03_DIR, exist_ok=True)

    # TODO: Optionally load Task 2 model checkpoint from ARTIFACTS_DIR as starting point
    # TODO: Apply data augmentation (blur, JPEG compression, scaling, noise, etc.)
    # TODO: Train robust model with time-budget awareness
    # TODO: Calibrate threshold on calibration_augmented data (FPR <= 20%)
    # TODO: Evaluate on both validation and validation_augmented splits
    # TODO: Save best augmented model checkpoint to ARTIFACTS_DIR


if __name__ == "__main__":
    main()
