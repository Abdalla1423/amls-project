"""train.py - Model training (Task 2).

Usage: python train.py --timeout_seconds 1800
"""

import argparse
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR

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
NUM_EPOCHS = 50
LR = 0.005
WD = 0.0001

torch.set_num_interop_threads(1)

TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
LOG_FILE = os.path.join(TASK02_DIR, "training_log.txt")
BEST_MODEL_PATH = os.path.join(TASK02_DIR, "best_model.pt")
LAST_MODEL_PATH = os.path.join(TASK02_DIR, "last_model.pt")


def initialize_model_and_optimizer(lr, wd, num_epochs=NUM_EPOCHS):
    model = Given_CNN()
    model.to(DEVICE)
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    return model, optimizer, scheduler


def hyperparameter_tune(lrs, wds):
    """Grid-search over (lr, wd). Each combo re-runs main() and returns best recall_ai."""
    rows = []
    for lr in lrs:
        cols = []
        for wd in wds:
            print(f"Learning rate: {lr}, and Weight Decay: {wd}")
            recall = main(lr=lr, wd=wd)
            cols.append(recall)
        rows.append(cols)

    results = np.array(rows)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(results, aspect="auto")
    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            plt.text(j, i, f"{results[i, j]:.4f}", ha="center", va="center")
    plt.colorbar(im, label="Best Recall_AI")
    plt.xticks(range(len(wds)), [str(wd) for wd in wds])
    plt.yticks(range(len(lrs)), [str(lr) for lr in lrs])
    plt.xlabel("Weight Decay")
    plt.ylabel("Learning Rate")
    plt.title("Grid Search Results (Best Recall_AI)")
    plt.tight_layout()

    save_path = os.path.join(TASK02_DIR, "hyperparameter_tuning.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved parameter tuning plot to: {save_path}")


def main(lr=LR, wd=WD):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args, _ = parser.parse_known_args()

    start_time = time.time()
    deadline = start_time + args.timeout_seconds - 120

    set_seed()
    configure_runtime()
    os.makedirs(TASK02_DIR, exist_ok=True)

    print("\n=== Loading data ===")
    train_data = load_data_split("task02/training_data")
    cal_data = load_data_split("task02/calibration_data")
    val_data = load_data_split("task02/validation_data")

    if train_data is None:
        print("ERROR: training_data_*.npy not found. Run prepare.py first.")
        sys.exit(1)

    X_tr, y_tr = train_data
    print(f"Train: {X_tr.shape}  real={int((y_tr == 0).sum())}  ai={int((y_tr == 1).sum())}")

    model, optimizer, scheduler = initialize_model_and_optimizer(lr, wd)

    # prepare.py does not balance -> use inverse-frequency class weights.
    weights = compute_class_weights(y_tr, balanced=False)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

    train_loader = make_loader(X_tr, y_tr, batch_size=BATCH_SIZE, num_workers=0)
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

    thr_final = calibrate_threshold(model, cal_loader)
    torch.save({"state_dict": model.state_dict(), "threshold": thr_final}, LAST_MODEL_PATH)

    with open(LOG_FILE, mode="a") as file:
        file.write(f"\n[train.py] Done in {time.time() - start_time:.1f}s\n")
        file.write(f"Best recall_ai={best_recall:.2f} at threshold={best_thr:.2f}\n")

    return best_recall


if __name__ == "__main__":
    main()
    # hyperparameter_tune(lrs=[0.01, 0.008, 0.006, 0.005, 0.004, 0.001], wds=[0.0001, 0.001, 0.01])
