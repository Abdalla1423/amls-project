"""train_augmented.py - Augmented/robust model training (Task 3).

Usage: python train_augmented.py --timeout_seconds 1800 [--finetune]
"""

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

from common import (
    ARTIFACTS_DIR,
    BATCH_SIZE,
    DEVICE,
    MAX_FPR,
    Given_CNN,
    calibrate_threshold,
    compute_class_weights,
    configure_runtime,
    evaluate,
    init_weights,
    load_data_split,
    make_loader,
    set_seed,
    train_one_epoch,
)

TIME_OUT = 1800
NUM_EPOCHS = 100
LR = 0.005
WD = 0.0001

TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
TASK03_DIR = os.path.join(ARTIFACTS_DIR, "task03")
LOG_FILE = os.path.join(TASK03_DIR, "augmented_training_log.txt")
FINETUNE_CHECKPOINT_PATH = os.path.join(TASK02_DIR, "best_model.pt")
BEST_MODEL_PATH = os.path.join(TASK03_DIR, "best_model.pt")


class AddGaussianNoise:
    def __init__(self, mean=0.0, std_max=0.02):
        self.mean = mean
        self.std_max = std_max

    def __call__(self, tensor):
        std = random.uniform(0.0, self.std_max)
        if std == 0:
            return tensor
        return tensor + torch.randn(tensor.size()) * std + self.mean


class AugmentedImageDataset(Dataset):
    """Applies augmentation only when is_training=True; always renormalizes."""

    _mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    _std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    _train_transforms = T.Compose([
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomApply(
            [T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)],
            p=0.3,
        ),
        T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.1),
        T.RandomApply([AddGaussianNoise(mean=0.0, std_max=0.015)], p=0.1),
    ])

    def __init__(self, images, labels, is_training):
        self.images = images
        self.labels = labels
        self.is_training = is_training
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)

        if self.is_training:
            img = (img * self._std) + self._mean
            img = torch.clamp(img, 0.0, 1.0)
            img = self._train_transforms(img)
            img = torch.clamp(img, 0.0, 1.0)
            img = self.normalize(img)

        return img, label


def make_augmented_loader(X, y, batch_size=BATCH_SIZE, shuffle=True, is_training=False):
    g = torch.Generator()
    g.manual_seed(42)
    dataset = AugmentedImageDataset(X, y, is_training=is_training)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=g if shuffle else None,
    )


def load_finetune_model():
    if not os.path.exists(FINETUNE_CHECKPOINT_PATH):
        print(f"ERROR: {FINETUNE_CHECKPOINT_PATH} not found. "
              "Run train.py first to produce task02/best_model.pt.")
        sys.exit(1)
    ckpt = torch.load(FINETUNE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = Given_CNN()
    model.load_state_dict(ckpt["state_dict"])
    return model


def initialize_model_and_optimizer(finetune, lr=LR, wd=WD, num_epochs=NUM_EPOCHS):
    if finetune:
        model = load_finetune_model()
        model.to(DEVICE)
    else:
        model = Given_CNN()
        model.to(DEVICE)
        model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    return model, optimizer, scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    parser.add_argument("--finetune", action="store_true", help="Fine-tune from task02 best_model.pt")
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + args.timeout_seconds - 120

    set_seed()
    configure_runtime()
    os.makedirs(TASK03_DIR, exist_ok=True)

    print("\n=== Loading data ===")
    train_data = load_data_split("task02/training_data")
    cal_data = load_data_split("task03/calibration_augmented_data")
    val_data = load_data_split("task03/validation_augmented_data")

    if train_data is None:
        print("ERROR: training_data_*.npy not found. Run prepare.py first.")
        sys.exit(1)

    X_tr, y_tr = train_data
    print(f"Train: {X_tr.shape}  real={int((y_tr == 0).sum())}  ai={int((y_tr == 1).sum())}")

    model, optimizer, scheduler = initialize_model_and_optimizer(finetune=args.finetune)

    weights = compute_class_weights(y_tr, balanced=False)
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    # Augmentation on the training loader; identity + normalization skip on val/cal
    # so metrics are directly comparable to Task 2.
    train_loader = make_augmented_loader(X_tr, y_tr, batch_size=BATCH_SIZE, shuffle=True, is_training=True)
    cal_loader = make_loader(cal_data[0], cal_data[1], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = make_loader(val_data[0], val_data[1], batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    best_recall, best_thr = 0.0, 0.5

    for epoch in range(NUM_EPOCHS):
        e_start = time.time()

        if time.time() > deadline:
            print("\nTimeout reached, stopping training.")
            break

        avg_loss = train_one_epoch(model, optimizer, loss_fn, train_loader, DEVICE)
        scheduler.step()

        with open(LOG_FILE, mode="a") as file:
            file.write(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}\n")

        thr = calibrate_threshold(model, cal_loader)

        if val_data:
            metrics = evaluate(model, val_loader, thr, "validation")
            if metrics["recall_ai"] > best_recall and metrics["fpr"] <= MAX_FPR:
                best_recall = metrics["recall_ai"]
                best_thr = thr
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "threshold": best_thr,
                        "epoch": epoch + 1,
                        "recall_ai": best_recall,
                        "fpr": metrics["fpr"],
                    },
                    BEST_MODEL_PATH,
                )
                with open(LOG_FILE, mode="a") as file:
                    file.write(
                        f"New best model saved at epoch {epoch+1} with recall_ai={best_recall:.4f} "
                        f"and FPR={metrics['fpr']:.4f} at threshold={best_thr:.4f}\n"
                    )

        print(f"Epoch completed in {time.time() - e_start:.1f} seconds")

    with open(LOG_FILE, mode="a") as file:
        file.write(f"\n[train_augmented.py] Done in {time.time() - start_time:.1f}s\n")
        file.write(f"Best recall_ai={best_recall:.2f} at threshold={best_thr:.2f}\n")


if __name__ == "__main__":
    main()
