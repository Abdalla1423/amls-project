"""
Task 1.2 - Model Training

Trains a CNN for binary classification (real vs ai_generated).
Uses the cleaned training data and calibrates the decision threshold
on the calibration split to maintain FPR ≤ 20% on real images.
Periodically saves the best model checkpoint.

Usage: python train.py --timeout_seconds 1800
"""

import argparse
import os
import signal
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import recall_score, confusion_matrix

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50
K = 32  # Base channel width
MAX_FPR = 0.20  # Maximum false-positive rate on real images

# Reproducibility
SEED = 42

# CPU thread control
torch.manual_seed(SEED)
np.random.seed(SEED)
torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


def timeout_handler(signum, frame):
    print("[train.py] Timeout reached – exiting with saved checkpoint.")
    sys.exit(0)


def build_cnn(k: int = K) -> nn.Module:
    """Build the reference CNN architecture for binary classification."""
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


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int,
                shuffle: bool = True) -> DataLoader:
    """Create a PyTorch DataLoader from numpy arrays."""
    X_t = torch.from_numpy(X)
    y_t = torch.from_numpy(y).long()
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=False)


def calibrate_threshold(model: nn.Module, cal_X: np.ndarray,
                        cal_y: np.ndarray, max_fpr: float = MAX_FPR) -> float:
    """
    Find the decision threshold on calibration data such that
    the false-positive rate on real images (class 0) is ≤ max_fpr.
    """
    model.eval()
    loader = make_loader(cal_X, cal_y, BATCH_SIZE, shuffle=False)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]  # P(ai_generated)
            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)

    # Get probabilities for real images (class 0)
    real_probs = probs[labels == 0]

    # Threshold = the (1 - max_fpr) quantile of real-image probabilities
    # This ensures at most max_fpr fraction of real images are above threshold
    if len(real_probs) == 0:
        return 0.5
    threshold = np.quantile(real_probs, 1 - max_fpr)
    return float(threshold)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray,
             threshold: float, split_name: str = "validation"):
    """Evaluate the model and print metrics."""
    model.eval()
    loader = make_loader(X, y, BATCH_SIZE, shuffle=False)

    all_probs = []
    all_labels = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            logits = model(X_batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
            all_probs.extend(probs.numpy())
            all_labels.extend(y_batch.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= threshold).astype(int)

    # Metrics
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall_ai = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    accuracy = (tp + tn) / len(labels)

    print(f"\n  [{split_name}] threshold={threshold:.4f}")
    print(f"    Accuracy:   {accuracy:.4f}")
    print(f"    Recall AI:  {recall_ai:.4f}")
    print(f"    FPR (real):  {fpr:.4f}  (max allowed: {MAX_FPR})")
    print(f"    Confusion:  TN={tn} FP={fp} FN={fn} TP={tp}")

    return {"accuracy": accuracy, "recall_ai": recall_ai, "fpr": fpr}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()

    if hasattr(signal, "SIGALRM"):
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(max(1, args.timeout_seconds - 60))  # 60s safety margin

    start_time = time.time()
    os.makedirs(TASK02_DIR, exist_ok=True)

    # Load data
    train_data = load_split("train")
    cal_data = load_split("calibration")
    val_data = load_split("validation")
    val_aug_data = load_split("validation_augmented")

    if train_data is None:
        print("ERROR: training data not found. Run prepare.py first.")
        sys.exit(1)

    train_X, train_y = train_data
    print(f"Training data: {train_X.shape}, labels: {np.bincount(train_y)}")

    # Build model
    model = build_cnn(K)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,
                                  weight_decay=1e-4)

    # Class weights for imbalanced data
    n_real = (train_y == 0).sum()
    n_ai = (train_y == 1).sum()
    weight = torch.tensor([1.0, n_real / max(n_ai, 1)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight)

    train_loader = make_loader(train_X, train_y, BATCH_SIZE, shuffle=True)

    # Training loop
    best_recall = 0.0
    best_threshold = 0.5

    for epoch in range(NUM_EPOCHS):
        # Check time budget
        elapsed = time.time() - start_time
        if elapsed > args.timeout_seconds - 120:
            print(f"\n[train.py] Approaching timeout at epoch {epoch+1}. Stopping.")
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

        # Calibrate threshold and evaluate every 5 epochs
        if (epoch + 1) % 5 == 0 or epoch == 0:
            if cal_data is not None:
                threshold = calibrate_threshold(model, cal_data[0], cal_data[1])
            else:
                threshold = 0.5

            if val_data is not None:
                metrics = evaluate(model, val_data[0], val_data[1],
                                   threshold, "validation")

                # Save best model (highest recall_ai with FPR constraint)
                if metrics["fpr"] <= MAX_FPR and metrics["recall_ai"] > best_recall:
                    best_recall = metrics["recall_ai"]
                    best_threshold = threshold
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "threshold": best_threshold,
                        "epoch": epoch + 1,
                        "recall_ai": best_recall,
                        "fpr": metrics["fpr"],
                    }, os.path.join(ARTIFACTS_DIR, "best_model.pt"))
                    print(f"    ** New best model saved (recall_ai={best_recall:.4f})")

            if val_aug_data is not None:
                evaluate(model, val_aug_data[0], val_aug_data[1],
                         threshold, "validation_augmented")

    # Final save (always save last checkpoint too)
    if cal_data is not None:
        final_threshold = calibrate_threshold(model, cal_data[0], cal_data[1])
    else:
        final_threshold = 0.5

    torch.save({
        "model_state_dict": model.state_dict(),
        "threshold": final_threshold,
        "epoch": NUM_EPOCHS,
    }, os.path.join(ARTIFACTS_DIR, "last_model.pt"))

    print(f"\n[train.py] Best recall_ai: {best_recall:.4f} "
          f"at threshold: {best_threshold:.4f}")
    print(f"[train.py] Completed in {time.time()-start_time:.1f}s")


if __name__ == "__main__":
    main()
