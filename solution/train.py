"""train.py - Model training and threshold calibration (Task 1.2).

Trains a CNN on the cleaned training data, calibrates the operating
threshold on the calibration set to enforce FPR ≤ 20% on real images,
and evaluates on both validation and validation_augmented.

Saves the best model checkpoint (by recall_ai under the FPR constraint)
to artifacts/best_model.pt.

Usage: python train.py --timeout_seconds 1800
"""

import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

BATCH_SIZE = 64
LR = 5e-4
K = 32          # base channel width
MAX_FPR = 0.20
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_cnn(k=K):
    """Reference CNN (Appendix B) with BatchNorm for faster convergence."""
    return nn.Sequential(
        # Block 1
        nn.Conv2d(3, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 2
        nn.Conv2d(k, 2 * k, 3, padding=1), nn.BatchNorm2d(2 * k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 3
        nn.Conv2d(2 * k, 4 * k, 3, padding=1), nn.BatchNorm2d(4 * k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 4
        nn.Conv2d(4 * k, 4 * k, 3, padding=1), nn.BatchNorm2d(4 * k), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        # Classifier
        nn.Flatten(),
        nn.Dropout(0.3),
        nn.Linear(4 * k, 2),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_split(name):
    path = os.path.join(ARTIFACTS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["X"], d["y"]


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def get_probs(model, X, y):
    """Return P(ai_generated) and labels as numpy arrays."""
    model.eval()
    loader = make_loader(X, y, shuffle=False)
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.softmax(model(xb), dim=1)[:, 1]
            all_p.append(p)
            all_y.append(yb)
    return torch.cat(all_p).numpy(), torch.cat(all_y).numpy()


def calibrate_threshold(model, cal_X, cal_y, max_fpr=MAX_FPR):
    """Set threshold so that FPR on real images ≤ max_fpr.

    Searches thresholds from 0.05 to 0.95 and picks the one that
    maximises recall_ai while keeping FPR ≤ max_fpr.
    Uses a conservative margin (targets 0.75 * max_fpr) to account
    for distribution shift between calibration and validation.
    """
    probs, labels = get_probs(model, cal_X, cal_y)
    real_probs = probs[labels == 0]
    ai_probs = probs[labels == 1]
    if len(real_probs) == 0:
        return 0.5

    target_fpr = max_fpr * 0.85  # conservative margin
    best_thr, best_recall = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.01):
        fpr = (real_probs >= thr).mean()
        recall = (ai_probs >= thr).mean() if len(ai_probs) > 0 else 0.0
        if fpr <= target_fpr and recall > best_recall:
            best_recall = recall
            best_thr = float(thr)
    return best_thr


def evaluate(model, X, y, threshold, name):
    """Print and return metrics for a split."""
    probs, labels = get_probs(model, X, y)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / len(labels)
    print(f"  [{name}] thr={threshold:.4f}  recall_ai={recall:.4f}  "
          f"fpr={fpr:.4f}  acc={acc:.4f}  (TN={tn} FP={fp} FN={fn} TP={tp})")
    return {"recall_ai": recall, "fpr": fpr, "acc": acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()

    t0 = time.time()
    deadline = t0 + args.timeout_seconds - 120  # stop 120s before timeout

    # Load data
    train_data = load_split("train")
    cal_data = load_split("calibration")
    val_data = load_split("validation")
    val_aug = load_split("validation_augmented")

    if train_data is None:
        print("ERROR: train.npz not found. Run prepare.py first.")
        sys.exit(1)

    X_tr, y_tr = train_data
    print(f"Train: {X_tr.shape}  real={int((y_tr == 0).sum())}  "
          f"ai={int((y_tr == 1).sum())}")

    # Model & optimizer
    model = build_cnn(K)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Class weights to handle 5:1 imbalance (real vs AI)
    n_real = int((y_tr == 0).sum())
    n_ai = int((y_tr == 1).sum())
    weights = torch.tensor([n_ai / n_real, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    train_loader = make_loader(X_tr, y_tr)
    best_recall, best_thr = 0.0, 0.5

    # Training loop
    for epoch in range(100):
        if time.time() > deadline:
            print(f"\nApproaching timeout at epoch {epoch+1}, stopping.")
            break

        model.train()
        total_loss, nb = 0.0, 0
        for xb, yb in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        scheduler.step()

        avg = total_loss / max(nb, 1)
        print(f"\nEpoch {epoch+1}  loss={avg:.4f}  ({time.time()-t0:.0f}s)")

        # Evaluate every epoch (only ~9 epochs fit in time budget)
        thr = (calibrate_threshold(model, cal_data[0], cal_data[1])
               if cal_data else 0.5)

        if val_data:
            m = evaluate(model, val_data[0], val_data[1], thr, "val")
            if m["fpr"] <= MAX_FPR and m["recall_ai"] > best_recall:
                best_recall = m["recall_ai"]
                best_thr = thr
                torch.save({"state_dict": model.state_dict(),
                            "threshold": best_thr, "epoch": epoch + 1,
                            "recall_ai": best_recall, "fpr": m["fpr"]},
                           os.path.join(ARTIFACTS_DIR, "best_model.pt"))
                print(f"  ** saved best (recall_ai={best_recall:.4f})")

        if val_aug:
            evaluate(model, val_aug[0], val_aug[1], thr, "val_aug")

    # Always save final model too
    thr_final = (calibrate_threshold(model, cal_data[0], cal_data[1])
                 if cal_data else 0.5)
    torch.save({"state_dict": model.state_dict(), "threshold": thr_final},
               os.path.join(ARTIFACTS_DIR, "last_model.pt"))

    print(f"\n[train.py] Best recall_ai={best_recall:.4f} thr={best_thr:.4f}")
    print(f"[train.py] Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
