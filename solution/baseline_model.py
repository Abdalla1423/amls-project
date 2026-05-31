"""baseline_model.py - Classical baseline for Task 2 report.

Extracts hand-crafted features from images and trains a Logistic Regression
and Random Forest classifier. Uses the same threshold calibration and
evaluation as the CNN pipeline for fair comparison.

Usage: python baseline_model.py
"""

import os
import time

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------
def extract_features(X):
    """Extract hand-crafted features from images.

    X shape: (N, 3, H, W), values in [0, 1].

    Features per channel (R, G, B):
      - mean, std, min, max, skewness, kurtosis  (6 × 3 = 18)
      - 16-bin histogram                          (16 × 3 = 48)
    Global features:
      - mean pixel intensity                      (1)
      - edge density (Laplacian-like variance)     (1)
      - inter-channel correlation (3 pairs)        (3)
    Total: 71 features per image.
    """
    N = X.shape[0]
    feats = []

    for i in range(N):
        img = X[i]  # (3, H, W)
        row = []

        # Per-channel statistics
        for c in range(3):
            ch = img[c].ravel()
            m = ch.mean()
            s = ch.std() + 1e-8
            row.append(m)
            row.append(s)
            row.append(ch.min())
            row.append(ch.max())
            # skewness
            row.append(((ch - m) ** 3).mean() / (s ** 3))
            # kurtosis
            row.append(((ch - m) ** 4).mean() / (s ** 4) - 3.0)
            # histogram (16 bins)
            hist, _ = np.histogram(ch, bins=16, range=(0, 1))
            row.extend(hist / len(ch))  # normalized

        # Global features
        row.append(img.mean())

        # Edge density: variance of simple gradient magnitude
        gray = img.mean(axis=0)  # (H, W)
        dx = np.diff(gray, axis=1)
        dy = np.diff(gray, axis=0)
        row.append(dx.var() + dy.var())

        # Inter-channel correlation
        flat = img.reshape(3, -1)
        for c1, c2 in [(0, 1), (0, 2), (1, 2)]:
            corr = np.corrcoef(flat[c1], flat[c2])[0, 1]
            row.append(corr if not np.isnan(corr) else 0.0)

        feats.append(row)

    return np.array(feats, dtype=np.float32)


# ---------------------------------------------------------------------------
# Threshold calibration (same logic as CNN pipeline)
# ---------------------------------------------------------------------------
def calibrate_threshold(probs, labels, max_fpr=0.20):
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


def evaluate(probs, labels, threshold, name):
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
def load_split(name):
    path = os.path.join(ARTIFACTS_DIR, f"{name}.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path)
    return d["X"], d["y"]


def main():
    t0 = time.time()

    # Load data
    train_data = load_split("train")
    cal_data = load_split("calibration")
    val_data = load_split("validation")
    val_aug = load_split("validation_augmented")

    if train_data is None:
        print("ERROR: train.npz not found. Run prepare.py first.")
        return

    X_tr, y_tr = train_data
    print(f"Train: {X_tr.shape}  real={int((y_tr == 0).sum())}  "
          f"ai={int((y_tr == 1).sum())}")

    # Extract features
    print("Extracting training features...")
    F_tr = extract_features(X_tr)
    print(f"  Features shape: {F_tr.shape}  ({time.time()-t0:.0f}s)")

    # Scale features
    scaler = StandardScaler()
    F_tr_scaled = scaler.fit_transform(F_tr)

    # --- Model 1: Logistic Regression ---
    print("\n=== Logistic Regression ===")
    lr = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    lr.fit(F_tr_scaled, y_tr)
    print(f"  Trained ({time.time()-t0:.0f}s)")

    # Calibrate and evaluate
    if cal_data:
        F_cal = scaler.transform(extract_features(cal_data[0]))
        lr_cal_probs = lr.predict_proba(F_cal)[:, 1]
        lr_thr = calibrate_threshold(lr_cal_probs, cal_data[1])
    else:
        lr_thr = 0.5

    if val_data:
        F_val = scaler.transform(extract_features(val_data[0]))
        lr_val_probs = lr.predict_proba(F_val)[:, 1]
        evaluate(lr_val_probs, val_data[1], lr_thr, "val")

    if val_aug:
        F_val_aug = scaler.transform(extract_features(val_aug[0]))
        lr_aug_probs = lr.predict_proba(F_val_aug)[:, 1]
        evaluate(lr_aug_probs, val_aug[1], lr_thr, "val_aug")

    # --- Model 2: Random Forest ---
    print("\n=== Random Forest ===")
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(F_tr_scaled, y_tr)
    print(f"  Trained ({time.time()-t0:.0f}s)")

    if cal_data:
        rf_cal_probs = rf.predict_proba(F_cal)[:, 1]
        rf_thr = calibrate_threshold(rf_cal_probs, cal_data[1])
    else:
        rf_thr = 0.5

    if val_data:
        rf_val_probs = rf.predict_proba(F_val)[:, 1]
        evaluate(rf_val_probs, val_data[1], rf_thr, "val")

    if val_aug:
        rf_aug_probs = rf.predict_proba(F_val_aug)[:, 1]
        evaluate(rf_aug_probs, val_aug[1], rf_thr, "val_aug")

    print(f"\nDone in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
