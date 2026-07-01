"""train.py - Model training (Task 2).

Usage: python train.py --timeout_seconds 1800
"""

import argparse
import os
import sys
import time
import psutil
import matplotlib.pyplot as plt

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

TIME_OUT = 1800
SEED = 42

BATCH_SIZE = 64
NUM_EPOCHS = 50
LR = 5e-3
WD = 1e-2
MAX_FPR = 0.20
DEVICE = "cpu"

K = 16
IN_CHANNELS = 3

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")

LOG_FILE = os.path.join(TASK02_DIR, "training_log.txt")
CHECKPOINT_PATH = os.path.join(TASK02_DIR, "best_model.pth")

# Custom Dataset class to load images and labels from our cleaned parquet file
class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx])).float()
        return img, torch.tensor(self.labels[idx]).long()
    
# Provided CNN model
class Given_CNN(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=2, k=K):
        super(Given_CNN, self).__init__()

        # Block 1:
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(k)
        self.pool1 = nn.MaxPool2d(2)

        self.block1 = nn.Sequential(self.conv1, self.bnorm1, nn.ReLU(), self.pool1)

        # Block 2:
        self.conv2 = nn.Conv2d(in_channels=k, out_channels=2*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(2*k)
        self.pool2 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(self.conv2, self.bnorm2, nn.ReLU(), self.pool2)

        # Block 3:
        self.conv3 = nn.Conv2d(in_channels=2*k, out_channels=4*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(4*k)
        self.pool3 = nn.MaxPool2d(2)

        # Block 4:
        self.conv4 = nn.Conv2d(in_channels=4*k, out_channels=4*k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(4*k)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.block3 = nn.Sequential(self.conv3, self.bnorm3, nn.ReLU(), self.pool3,
                                    self.conv4, self.bnorm4, nn.ReLU(),
                                    self.global_pool)

        # Classifier:
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(4*k, num_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x

def print_ram_usage(msg=""):
    process = psutil.Process(os.getpid())
    ram_gb = process.memory_info().rss / (1024**3)
    print(f"{msg} RAM used: {ram_gb:.2f} GB")

# Set random seeds for reproducibility
def set_random_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

# Prepare training dataloaders with balanced classes (for both train and test sets)
def load_data_split(split_name):
    x_path = os.path.join(ARTIFACTS_DIR, f"{split_name}_X.npy")
    y_path = os.path.join(ARTIFACTS_DIR, f"{split_name}_y.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        return None

    X = np.load(x_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    return X, y

def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True):
    g = torch.Generator()
    g.manual_seed(42)
    
    dataset = ImageDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=2, generator=g if shuffle else None,
                        persistent_workers=True)

# Initialize model weights with Kaiming He initialization for better convergence
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

# Initialize CNN model and optimizer for deep learning prediction
def initialize_model_and_optimizer(lr, wd):
    model = Given_CNN()
    model.to(DEVICE)
    model.apply(init_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=20)
    return model, optimizer, scheduler

# Initialize loss function for training
def initialize_loss_function(weights):
    return torch.nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)

# Calibrate decision threshold on calibration data to achieve FPR <= 20% while maximizing recall on AI samples
def calibrate_threshold(model, cal_loader, max_fpr=MAX_FPR):
    probs, y_true = get_probs(model, cal_loader)
    real_probs = probs[y_true == 0]
    ai_probs = probs[y_true == 1]
    if len(real_probs) == 0:
        print("  Warning: No real samples in calibration set, using default threshold 0.5")
        return 0.5

    target_fpr = max_fpr * 0.85  # safety margin
    best_thr, best_recall = 0.5, 0.0
    for thr in np.arange(0.05, 0.96, 0.001):
        fpr = (real_probs >= thr).mean()
        recall = (ai_probs >= thr).mean() if len(ai_probs) > 0 else 0.0
        if fpr <= target_fpr and recall > best_recall:
            best_recall = recall
            best_thr = float(thr)
    return best_thr

# Get predicted probabilities from the model for a given dataset
def get_probs(model, loader):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            p = torch.softmax(model(X_batch.to(DEVICE)), dim=1)[:, 1].cpu().numpy()
            all_p.append(p)
            all_y.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)

# Evaluate model performance on a given dataset
def evaluate(model, val_loader, threshold, split_name="validation"):
    probs, y_true = get_probs(model, val_loader)
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    print(f"  {split_name}: FPR={fpr:.4f}  Recall_ai={recall:.4f}  Accuracy={acc:.4f}  (threshold={threshold:.4f})")
    return {"fpr": fpr, "recall_ai": recall, "accuracy": acc}

# Training loop for the CNN
def train_one_epoch(model, optimizer, loss_fn, train_loader, device):
  model.train()
  losses = []
  for batch, labels in train_loader:
    batch = batch.to(device)
    labels = labels.to(device)
    predictions = model(batch)
    loss = loss_fn(predictions, labels)
    losses.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  avg_loss = sum(losses) / len(losses)
  return avg_loss

# Hyperparameter tune learning rate and weight decay
def hyperparameter_tune(lrs, wds):
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
    im = plt.imshow(results, aspect='auto')

    for i in range(results.shape[0]):
        for j in range(results.shape[1]):
            plt.text(
                j,
                i,
                f"{results[i, j]:.4f}",
                ha="center",
                va="center"
            )

    plt.colorbar(im, label="Best Recall_AI")

    plt.xticks(
        range(len(wds)),
        [str(wd) for wd in wds]
    )

    plt.yticks(
        range(len(lrs)),
        [str(lr) for lr in lrs]
    )

    plt.xlabel("Weight Decay")
    plt.ylabel("Learning Rate")
    plt.title("Grid Search Results (Best Recall_AI)")

    plt.tight_layout()
    plt.show()

# Deep learning prediction with a simple CNN classifier
def main(lr=LR, wd=WD):
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + args.timeout_seconds - 120 

    # 0. Set random seeds for reproducibility
    set_random_seeds()

    os.makedirs(TASK02_DIR, exist_ok=True)

    # 1. Load prepared training and validation data from ARTIFACTS_DIR
    print("\n=== Loading data ===")

    print_ram_usage("Start")

    train_data = load_data_split("task02/training_data")
    cal_data = load_data_split("task02/calibration_data")
    val_data = load_data_split("task02/validation_data")
    
    print_ram_usage("After loading")

    if train_data is None:
        print("ERROR: train.npy not found. Run prepare.py first.")
        sys.exit(1)

    X_tr, y_tr = train_data
    print(f"Train: {X_tr.shape}  real={int((y_tr == 0).sum())}  ai={int((y_tr == 1).sum())}")

    # 2.1. Build model, optimizer
    model, optimizer, scheduler = initialize_model_and_optimizer(lr, wd)

    # 2.2. Compute class weights for imbalanced training data and initialize loss function
    n_real = int((y_tr == 0).sum())
    n_ai = int((y_tr == 1).sum())
    weights = torch.tensor([n_ai / n_real, 1.0], device=DEVICE)
    loss_fn = initialize_loss_function(weights)

    train_loader = make_loader(X_tr, y_tr, batch_size=BATCH_SIZE)
    cal_loader = make_loader(cal_data[0], cal_data[1], batch_size=BATCH_SIZE, shuffle=False)
    val_loader = make_loader(val_data[0], val_data[1], batch_size=BATCH_SIZE, shuffle=False)

    best_recall, best_thr = 0.0, 0.5

    # 3. Train with time-budget awareness
    for epoch in range(NUM_EPOCHS):
        e_start_time = time.time()

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
                torch.save({"state_dict": model.state_dict(),
                            "threshold": best_thr, "epoch": epoch+1,
                            "recall_ai": best_recall, "fpr": metrics["fpr"]},
                            os.path.join(TASK02_DIR, "best_model.pt"))
                with open(LOG_FILE, mode="a") as file:
                    file.write(f"New best model saved at epoch {epoch+1} with recall_ai={best_recall:.4f} and FPR={metrics['fpr']:.4f} at threshold={best_thr:.4f}\n")
        
        print(f"Epoch completed in {time.time() - e_start_time:.1f} seconds")
        print_ram_usage(f"After epoch {epoch+1}")

    # Always save last model
    thr_final = calibrate_threshold(model, cal_loader)
    torch.save({"state_dict": model.state_dict(), "threshold": thr_final,}, 
               os.path.join(TASK02_DIR, "last_model.pt"))

    with open(LOG_FILE, mode="a") as file:
        file.write(f"\n[train.py] Done in {time.time() - start_time:.1f}s\n")
        file.write(f"Best recall_ai={best_recall:.2f} at threshold={best_thr:.2f}\n")
    
    return best_recall


if __name__ == "__main__":
    #main()
    #hyperparameter_tune(lrs=[0.0005, 0.001, 0.005], wds=[0.0001, 0.001, 0.01])
