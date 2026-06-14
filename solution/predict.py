"""predict.py - Inference for Task 2.

Outputs: artifacts/task02/predictions.csv (columns: row_id, predicted_label)

Usage: python predict.py --timeout_seconds 600
"""

import argparse
import os
import sys
import csv
import glob
import pandas as pd

import torch
from torch.utils.data import DataLoader

from model import Given_CNN, ImageDataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
PREDICT_DIR = os.path.join(DATA_DIR, "predict") 
PREDICTIONS = os.path.join(TASK02_DIR, "predictions.csv")
CHECKPOINT_PATH = os.path.join(TASK02_DIR, "best_model.pth")
LOG_FILE = os.path.join(TASK02_DIR, "prediction_report.txt")

TIME_OUT = 600

BATCH_SIZE = 64
DEVICE = "cpu"

def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = Given_CNN()
    model.load_state_dict(ckpt["state_dict"])

    return model

# Save predictions to CSV file at artifacts/task02/predictions.csv
def save_predictions(all_preds, save_path=PREDICTIONS):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, mode="w") as file:
        writer = csv.writer(file)
        writer.writerow(["row_id", "predicted_label"])
        for id, pred in enumerate(all_preds):
            writer.writerow([id, pred])

# Prepare prediction dataloader from data/predict/ 
def prepare_prediction_dataloader(predict_path=PREDICT_DIR):
    file_paths = sorted(glob.glob(os.path.join(predict_path, "*.parquet")))
    if file_paths:
        df_list = [pd.read_parquet(fp) for fp in file_paths]
        df = pd.concat(df_list, ignore_index=True)
    else:
        raise FileNotFoundError(f"No parquet files found in {predict_path}")
    
    dataset = ImageDataset(df)
    data_loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return data_loader

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args = parser.parse_args()

    os.makedirs(TASK02_DIR, exist_ok=True)

    # 1. Load trained model from ARTIFACTS_DIR
    model = load_model()

    # 2. Load predict data from PREDICT_DIR (parquet with columns: row_id, image)
    data_loader = prepare_prediction_dataloader()
    
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch, labels in data_loader:
            batch = batch.to(DEVICE)
            predictions = model(batch)
            preds = predictions.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    # 3. Write artifacts/task02/predictions.csv (columns: row_id, predicted_label)
    output_file = os.path.join(ARTIFACTS_DIR, "task02/predictions.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    save_predictions(all_preds, save_path=output_file)


if __name__ == "__main__":
    main()
