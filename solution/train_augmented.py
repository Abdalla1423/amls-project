"""
Task 1.3 - Data Augmentation and Robust Training

Applies augmentation techniques (blur, compression, scaling, noise)
to make the model robust to real-world image modifications.
Builds upon the Task 2 model checkpoint.

Usage: python train_augmented.py --timeout_seconds 1800
"""

import argparse
import io
import os
import random
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image, ImageFilter
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK03_DIR = os.path.join(ARTIFACTS_DIR, "task03")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 5e-4
NUM_EPOCHS = 40
K = 32
MAX_FPR = 0.20
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


def timeout_handler(signum, frame):
    print("[train_augmented.py] Timeout – exiting with saved checkpoint.")
    sys.exit(0)


def build_cnn(k: int = K) -> nn.Module:
    """Build the same CNN architecture used in training."""
    features = nn.Sequential(
        nn.Conv2d(3, k, kernel_size=3, padding=1),
        nn.BatchNorm2d(k),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(k, 2 * k, kernel_size=3, padding=1),
        nn.BatchNorm2d(2 * k),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2),
        nn.Conv2d(2 * k, 4 * k, kernel_size=3, padding=1),
        nn.BatchNorm2d(4 * k),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )
    classifier = nn.Sequential(
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(4 * k, 2),
    )
    return nn.Sequential(features, classifier)


def load_split(name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """Load a prepared .npz split."""
    path = os.path.join(ARTIFACTS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        print(f"Warning: {path} not found.")
        return None
    data = np.load(path)
    return data["X"], data["y"]


# ---------------------------------------------------------------------------
# Augmentation
# ---------------------------------------------------------------------------
class AugmentedDataset(Dataset):
    """Dataset that applies random augmentations during training."""

    def __init__(self, X: np.ndarray, y: np.ndarray, augment: bool = True):
        self.X = X  # (N, 3, H, W) float32 [0, 1]
        self.y = y
        self.augment = augment

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = self.X[idx].copy()  # (3, H, W)
        label = self.y[idx]

        if self.augment and random.random() < 0.7:
            x = self._apply_augmentation(x)

        return torch.from_numpy(x), int(label)

    def _apply_augmentation(self, x: np.ndarray) -> np.ndarray:
        """Apply one or more random augmentations."""
        # Convert to PIL for augmentation
        img_arr = (x.transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(img_arr, "RGB")

        augmentations = [
            self._blur,
            self._jpeg_compress,
            self._scale,
            self._add_noise,
            self._horizontal_flip,
            self._brightness,
        ]

        # Apply 1-3 random augmentations
        n_augs = random.randint(1, 3)
        chosen = random.sample(augmentations, n_augs)
        for aug_fn in chosen:
            img = aug_fn(img)

        # Convert back to numpy (C, H, W) float32
        arr = np.array(img, dtype=np.float32) / 255.0
        return arr.transpose(2, 0, 1)

    @staticmethod
    def _blur(img: Image.Image) -> Image.Image:
        radius = random.uniform(0.5, 2.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))

    @staticmethod
    def _jpeg_compress(img: Image.Image) -> Image.Image:
        quality = random.randint(30, 85)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")

    @staticmethod
    def _scale(img: Image.Image) -> Image.Image:
        w, h = img.size
        factor = random.uniform(0.5, 0.9)
        new_w, new_h = int(w * factor), int(h * factor)
        img = img.resize((new_w, new_h), Image.LANCZOS)
        return img.resize((w, h), Image.LANCZOS)  # Scale back up

    @staticmethod
    def _add_noise(img: Image.Image) -> Image.Image:
        arr = np.array(img, dtype=np.float32)
        noise = np.random.normal(0, random.uniform(5, 20), arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")

    @staticmethod
    def _horizontal_flip(img: Image.Image) -> Image.Image:
        if random.random() < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    @staticmethod
    def _brightness(img: Image.Image) -> Image.Image:
        factor = random.uniform(0.7, 1.3)
        arr = np.array(img, dtype=np.float32) * factor
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr, "RGB")


def calibrate_threshold(model: nn.Module, cal_X: np.ndarray,
                        cal_y: np.ndarray, max_fpr: float = MAX_FPR) -> float:
    """Find threshold so FPR on real images ≤ max_fpr."""
    model.eval()
    loader = DataLoader(
        AugmentedDataset(cal_X, cal_y, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = torch.softmax(model(X_batch), dim=1)[:, 1]
            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    real_probs = probs[labels == 0]

    if len(real_probs) == 0:
        return 0.5
    return float(np.quantile(real_probs, 1 - max_fpr))


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray,
             threshold: float, split_name: str):
    """Evaluate and print metrics."""
    model.eval()
    loader = DataLoader(
        AugmentedDataset(X, y, augment=False),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = torch.softmax(model(X_batch), dim=1)[:, 1]
            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall_ai = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    print(f"  [{split_name}] threshold={threshold:.4f} "
          f"recall_ai={recall_ai:.4f} fpr={fpr:.4f} "
          f"(TN={tn} FP={fp} FN={fn} TP={tp})")
    return {"recall_ai": recall_ai, "fpr": fpr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, args.timeout_seconds - 60))

    start_time = time.time()
    os.makedirs(TASK03_DIR, exist_ok=True)

    # Load data
    train_data = load_split("train")
    cal_data = load_split("calibration_augmented")
    val_data = load_split("validation")
    val_aug_data = load_split("validation_augmented")

    if train_data is None:
        print("ERROR: training data not found. Run prepare.py first.")
        sys.exit(1)

    train_X, train_y = train_data
    print(f"Training data: {train_X.shape}, labels: {np.bincount(train_y)}")

    # Build model — try to load Task 2 checkpoint
    model = build_cnn(K)
    base_model_path = os.path.join(ARTIFACTS_DIR, "best_model.pt")
    if os.path.exists(base_model_path):
        checkpoint = torch.load(base_model_path, map_location="cpu",
                                weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Loaded Task 2 checkpoint from {base_model_path}")
    else:
        print("No Task 2 checkpoint found — training from scratch.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=1e-4)

    # Class weights
    n_real = (train_y == 0).sum()
    n_ai = (train_y == 1).sum()
    weight = torch.tensor([1.0, n_real / max(n_ai, 1)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    # Augmented training dataset
    train_ds = AugmentedDataset(train_X, train_y, augment=True)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=0)

    best_recall_aug = 0.0
    best_threshold = 0.5

    for epoch in range(NUM_EPOCHS):
        elapsed = time.time() - start_time
        if elapsed > args.timeout_seconds - 120:
            print(f"\n[train_augmented.py] Approaching timeout at epoch {epoch+1}.")
            break

        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        print(f"\nEpoch {epoch+1}/{NUM_EPOCHS} — loss: {avg_loss:.4f} "
              f"({time.time()-start_time:.0f}s elapsed)")

        # Evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            if cal_data is not None:
                threshold = calibrate_threshold(model, cal_data[0], cal_data[1])
            else:
                threshold = 0.5

            metrics_val = None
            if val_data is not None:
                metrics_val = evaluate(model, val_data[0], val_data[1],
                                       threshold, "validation")

            metrics_aug = None
            if val_aug_data is not None:
                metrics_aug = evaluate(model, val_aug_data[0], val_aug_data[1],
                                       threshold, "validation_augmented")

            # Save best model based on augmented validation recall
            save_recall = (metrics_aug["recall_ai"] if metrics_aug
                           else (metrics_val["recall_ai"] if metrics_val else 0))
            save_fpr = (metrics_aug["fpr"] if metrics_aug
                        else (metrics_val["fpr"] if metrics_val else 1))

            if save_fpr <= MAX_FPR and save_recall > best_recall_aug:
                best_recall_aug = save_recall
                best_threshold = threshold
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "threshold": best_threshold,
                    "epoch": epoch + 1,
                    "recall_ai_aug": best_recall_aug,
                }, os.path.join(ARTIFACTS_DIR, "best_model_augmented.pt"))
                print(f"  ** New best augmented model saved "
                      f"(recall_ai_aug={best_recall_aug:.4f})")

    # Save final checkpoint
    if cal_data is not None:
        final_threshold = calibrate_threshold(model, cal_data[0], cal_data[1])
    else:
        final_threshold = 0.5

    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold": final_threshold,
        "epoch": NUM_EPOCHS,
    }, os.path.join(ARTIFACTS_DIR, "last_model_augmented.pt"))

    print(f"\n[train_augmented.py] Best augmented recall_ai: {best_recall_aug:.4f}")
    print(f"[train_augmented.py] Completed in {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()
