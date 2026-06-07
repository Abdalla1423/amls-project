import argparse
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix

DEVICE = "cuda" if torch.cuda.is_available() else "cpu" 
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

def load_data_split(split_name):
    path = os.path.join(ARTIFACTS_DIR, f"{split_name}.npz")
    if not os.path.exists(path):
        return None
    data = np.load(path)
    return data["X"], data["y"]

def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).long())
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

def build_cnn(k=K):
    return nn.Sequential(
        # Block 1
        nn.Conv2d(3, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 2
        nn.Conv2d(k, 2*k, 3, padding=1), nn.BatchNorm2d(2*k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 3
        nn.Conv2d(2*k, 4*k, 3, padding=1), nn.BatchNorm2d(4*k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 4
        nn.Conv2d(4*k, 4*k, 3, padding=1), nn.BatchNorm2d(4*k), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        # Classifier
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(4*k, 2),
    )

def get_probs(model, X, y):
    model.eval()
    loader = make_loader(X, y, batch_size=BATCH_SIZE, shuffle=False)
    all_p, all_y = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            p = torch.softmax(model(X_batch.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            all_p.append(p)
            all_y.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)

def calibrate_threshold(model, X_cal, y_cal, max_fpr=MAX_FPR):
    probs, y_true = get_probs(model, X_cal, y_cal)
    real_probs = probs[y_true == 0]
    ai_probs = probs[y_true == 1]
    if len(real_probs) == 0:
        print("  Warning: No real samples in calibration set, using default threshold 0.5")
        return 0.5
    
    target_fpr = max_fpr * 0.85  # safety margin
    best_thr, best_recall = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.01):
        fpr = (real_probs >= thr).mean()
        recall = (ai_probs >= thr).mean() if len(ai_probs) > 0 else 0.0
        if fpr <= target_fpr and recall > best_recall:
            best_recall = recall
            best_thr = float(thr)
    return best_thr

def evaluate(model, X, y, threshold, split_name="validation"):
    probs, y_true = get_probs(model, X, y)
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    print(f"  {split_name}: FPR={fpr:.4f}  Recall_ai={recall:.4f}  Accuracy={acc:.4f}  (threshold={threshold:.4f})")
    return {"fpr": fpr, "recall_ai": recall, "accuracy": acc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=600)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + args.timeout_seconds - 120  # stop 120s before timeout  

    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("\n=== Loading data ===")
    train_data = load_data_split("train")
    cal_data = load_data_split("calibration")
    val_data = load_data_split("validation")
    val_aug = load_data_split("validation_augmented")

    if train_data is None:
        print("ERROR: train.npz not found. Run prepare.py first.")
        sys.exit(1)

    X_tr, y_tr = train_data
    print(f"Train: {X_tr.shape}  real={int((y_tr == 0).sum())}  ai={int((y_tr == 1).sum())}")

    print("\n=== Building model ===")
    model = build_cnn(K).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    n_real = int((y_tr == 0).sum())
    n_ai = int((y_tr == 1).sum())
    weights = torch.tensor([n_ai / n_real, 1.0], device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    train_loader = make_loader(X_tr, y_tr, batch_size=BATCH_SIZE)
    best_recall, best_thr = 0.0, 0.5

    # Training loop
    for epoch in range(100):
        if time.time() > deadline:
            print("\nTimeout reached, stopping training.")
            break

        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        scheduler.step()
        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch {epoch+1} - Loss: {avg_loss:.4f}")

        thr = calibrate_threshold(model, cal_data[0], cal_data[1]) if cal_data else 0.5

        if val_data:
            metrics = evaluate(model, val_data[0], val_data[1], thr, "validation")
            if metrics["recall_ai"] > best_recall and metrics["fpr"] <= MAX_FPR:
                best_recall = metrics["recall_ai"]
                best_thr = thr
                torch.save({"state_dict": model.state_dict(), 
                            "threshold": best_thr, "epoch": epoch+1,
                            "recall_ai": best_recall, "fpr": metrics["fpr"]}, 
                            os.path.join(ARTIFACTS_DIR, "best_model.pt"))
                print(f"  New best model saved with recall={best_recall:.4f} at threshold={best_thr:.4f}")

            if val_aug:
                evaluate(model, val_aug[0], val_aug[1], thr, "val_augmented")
        
    # Always save last model
    thr_final = calibrate_threshold(model, cal_data[0], cal_data[1]) if cal_data else 0.5
    torch.save({"state_dict": model.state_dict(), "threshold": thr_final,}, 
                os.path.join(ARTIFACTS_DIR, "last_model.pt"))

    print(f"\n[train.py] Done in {time.time() - start_time:.1f}s")
    print(f"Best recall_ai={best_recall:.4f} at threshold={best_thr:.4f}")

if __name__ == "__main__":
    main()
