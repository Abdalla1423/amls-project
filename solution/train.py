"""train.py - Model training (Task 2).

Usage: python train.py --timeout_seconds 1800
"""

import argparse
import os
import sys

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()

    os.makedirs(TASK02_DIR, exist_ok=True)

    # TODO: Load prepared training data from ARTIFACTS_DIR
    # TODO: Build model (e.g. CNN, see Appendix B in the exercise PDF)
    # TODO: Train with time-budget awareness (check args.timeout_seconds)
    # TODO: Calibrate decision threshold on calibration data (FPR <= 20%)
    # TODO: Evaluate on validation and validation_augmented splits
    # TODO: Save best model checkpoint to ARTIFACTS_DIR


if __name__ == "__main__":
    main()
