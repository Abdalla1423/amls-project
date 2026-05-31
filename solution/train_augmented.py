"""train_augmented.py - Augmented/robust model training (Task 3).

Trains the CNN from scratch with heavy data augmentation to improve
generalization to unseen image distributions.  Adds calibration_augmented
data to the training pool.  Calibrates threshold on calibration_augmented
to enforce FPR <= 20%.

Usage: python train_augmented.py --timeout_seconds 1800
"""

import argparse
import io
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Paths & hyper-parameters
# ---------------------------------------------------------------------------
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK03_DIR = os.path.join(ARTIFACTS_DIR, "task03")

BATCH_SIZE = 64
LR = 5e-4
K = 32
MAX_FPR = 0.20
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)
torch.use_deterministic_algorithms(True)
torch.set_num_threads(min(8, os.cpu_count() or 1))
torch.set_num_interop_threads(1)


# ---------------------------------------------------------------------------
# Model (same architecture as Task 2)
# ---------------------------------------------------------------------------
def build_cnn(k=K):
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
        nn.Dropout(0.5),
        nn.Linear(4 * k, 2),
    )


# ---------------------------------------------------------------------------
# Augmentation (applied on-the-fly as tensor ops)
# ---------------------------------------------------------------------------
def augment_batch(x):
    """Apply heavy random augmentations to a batch of images (N, 3, H, W).

    ~80% of images are augmented with strong transforms to improve
    generalization to unseen image distributions.
    """
    n, c, h, w = x.shape

    # Decide which images to augment (~80%)
    aug_mask = torch.rand(n) < 0.8
    x_aug = x[aug_mask].clone()
    n_aug = x_aug.shape[0]
    if n_aug == 0:
        return x

    # --- Per-image transforms (applied independently) ---

    # Gaussian noise (σ = 0.05-0.12, applied to ~60%)
    noise_mask = torch.rand(n_aug, 1, 1, 1) < 0.6
    sigma = 0.05 + torch.rand(n_aug, 1, 1, 1) * 0.07  # [0.05, 0.12]
    noise = torch.randn_like(x_aug) * sigma
    x_aug = x_aug + noise * noise_mask.float()

    # Brightness jitter (±0.20, applied to ~50%)
    bright_mask = torch.rand(n_aug, 1, 1, 1) < 0.5
    bright_shift = (torch.rand(n_aug, 1, 1, 1) - 0.5) * 0.40  # [-0.20, 0.20]
    x_aug = x_aug + bright_shift * bright_mask.float()

    # Contrast jitter ([0.6, 1.4], applied to ~50%)
    contrast_mask = torch.rand(n_aug, 1, 1, 1) < 0.5
    contrast_factor = 0.6 + torch.rand(n_aug, 1, 1, 1) * 0.8  # [0.6, 1.4]
    mean = x_aug.mean(dim=(2, 3), keepdim=True)
    x_aug = torch.where(contrast_mask, mean + (x_aug - mean)
                        * contrast_factor, x_aug)

    # Per-channel color jitter (applied to ~40%)
    color_mask = torch.rand(n_aug, 1, 1, 1) < 0.4
    color_shift = (torch.rand(n_aug, c, 1, 1) - 0.5) * 0.16  # [-0.08, 0.08]
    x_aug = x_aug + color_shift * color_mask.float()

    # --- Batch-level transforms ---

    # Random downscale + upscale (applied to ~30%)
    if torch.rand(1).item() < 0.3:
        scale = np.random.choice([0.4, 0.5, 0.6, 0.75])
        small_h, small_w = int(h * scale), int(w * scale)
        x_aug = F.interpolate(x_aug, size=(small_h, small_w), mode="bilinear",
                              align_corners=False)
        x_aug = F.interpolate(x_aug, size=(h, w), mode="bilinear",
                              align_corners=False)

    # Gaussian blur (applied to ~30%)
    if torch.rand(1).item() < 0.3:
        k_size = np.random.choice([3, 5])
        sigma = np.random.uniform(0.5, 1.5)
        ax = torch.arange(k_size, dtype=torch.float32) - k_size // 2
        kernel_1d = torch.exp(-ax ** 2 / (2 * sigma ** 2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = kernel_1d.unsqueeze(1) @ kernel_1d.unsqueeze(0)
        kernel_2d = kernel_2d.expand(c, 1, k_size, k_size)
        pad = k_size // 2
        x_aug = F.conv2d(F.pad(x_aug, [pad] * 4, mode="reflect"),
                         kernel_2d, groups=c)

    # Horizontal flip (~50%)
    flip_mask = torch.rand(n_aug) < 0.5
    x_aug[flip_mask] = x_aug[flip_mask].flip(-1)

    # CutOut / random erasing (~40% of augmented images)
    for i in range(n_aug):
        if torch.rand(1).item() < 0.4:
            eh = int(h * np.random.uniform(0.1, 0.3))
            ew = int(w * np.random.uniform(0.1, 0.3))
            y0 = np.random.randint(0, h - eh + 1)
            x0 = np.random.randint(0, w - ew + 1)
            x_aug[i, :, y0:y0 + eh, x0:x0 + ew] = torch.rand(c, 1, 1)

    x[aug_mask] = x_aug.clamp(0.0, 1.0)
    return x


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
    probs, labels = get_probs(model, cal_X, cal_y)
    real_probs = probs[labels == 0]
    ai_probs = probs[labels == 1]
    if len(real_probs) == 0:
        return 0.5

    target_fpr = max_fpr * 0.85
    best_thr, best_recall = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.01):
        fpr = (real_probs >= thr).mean()
        recall = (ai_probs >= thr).mean() if len(ai_probs) > 0 else 0.0
        if fpr <= target_fpr and recall > best_recall:
            best_recall = recall
            best_thr = float(thr)
    return best_thr


def evaluate(model, X, y, threshold, name):
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
    deadline = t0 + args.timeout_seconds - 120
    os.makedirs(TASK03_DIR, exist_ok=True)

    # Load data
    train_data = load_split("train")
    cal_aug = load_split("calibration_augmented")
    val_data = load_split("validation")
    val_aug = load_split("validation_augmented")

    if train_data is None:
        print("ERROR: train.npz not found. Run prepare.py first.")
        sys.exit(1)

    X_tr, y_tr = train_data

    # Add calibration_augmented data to training pool for more diversity
    if cal_aug is not None:
        X_tr = np.concatenate([X_tr, cal_aug[0]], axis=0)
        y_tr = np.concatenate([y_tr, cal_aug[1]], axis=0)
        print(f"Added cal_aug to training: +{cal_aug[0].shape[0]} samples")

    print(f"Train: {X_tr.shape}  real={int((y_tr == 0).sum())}  "
          f"ai={int((y_tr == 1).sum())}")

    # Train from scratch with heavy augmentation (dropout=0.5)
    model = build_cnn(K)
    print("Training from scratch with heavy augmentation & dropout=0.5")

    # Optimizer & scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    # Class weights
    n_real = int((y_tr == 0).sum())
    n_ai = int((y_tr == 1).sum())
    weights = torch.tensor([n_ai / n_real, 1.0])
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    train_loader = make_loader(X_tr, y_tr)
    best_recall_aug, best_thr = 0.0, 0.5

    # Training loop with augmentation
    for epoch in range(100):
        if time.time() > deadline:
            print(f"\nApproaching timeout at epoch {epoch+1}, stopping.")
            break

        model.train()
        total_loss, nb = 0.0, 0
        for xb, yb in train_loader:
            xb = augment_batch(xb)   # on-the-fly augmentation
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            nb += 1
        scheduler.step()

        avg = total_loss / max(nb, 1)
        print(f"\nEpoch {epoch+1}  loss={avg:.4f}  ({time.time()-t0:.0f}s)")

        # Calibrate on augmented calibration set
        thr = (calibrate_threshold(model, cal_aug[0], cal_aug[1])
               if cal_aug else 0.5)

        if val_data:
            evaluate(model, val_data[0], val_data[1], thr, "val")

        if val_aug:
            m = evaluate(model, val_aug[0], val_aug[1], thr, "val_aug")
            # Optimise for val_aug recall (Task 3 objective)
            if m["fpr"] <= MAX_FPR and m["recall_ai"] > best_recall_aug:
                best_recall_aug = m["recall_ai"]
                best_thr = thr
                torch.save({"state_dict": model.state_dict(),
                            "threshold": best_thr, "epoch": epoch + 1,
                            "recall_ai": best_recall_aug, "fpr": m["fpr"]},
                           os.path.join(ARTIFACTS_DIR, "best_model_augmented.pt"))
                print(f"  ** saved best aug (recall_ai={best_recall_aug:.4f})")

    # Save final model too
    thr_final = (calibrate_threshold(model, cal_aug[0], cal_aug[1])
                 if cal_aug else 0.5)
    torch.save({"state_dict": model.state_dict(), "threshold": thr_final},
               os.path.join(ARTIFACTS_DIR, "last_model_augmented.pt"))

    print(f"\n[train_augmented.py] Best val_aug recall_ai={best_recall_aug:.4f} "
          f"thr={best_thr:.4f}")
    print(f"[train_augmented.py] Done in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
