"""train_augmented.py - Augmented/robust model training (Task 3).

Usage: python train_augmented.py --timeout_seconds 1800
"""

import argparse
import os
import sys
import time
import psutil

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torchvision.transforms as T

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
TASK03_DIR = os.path.join(ARTIFACTS_DIR, "task03")

LOG_FILE = os.path.join(TASK03_DIR, "augmented_training_log.txt")
FINETUNE_CHECKPOINT_PATH = os.path.join(TASK02_DIR, "best_model.pt")
CHECKPOINT_PATH = os.path.join(TASK03_DIR, "best_model.pt")

# Custom Dataset class to load images and labels from our cleaned parquet file, with data augmentation for training
class AddGaussianNoise(object):
    def __init__(self, mean=0.0, std_max=0.02):
        self.mean = mean
        self.std_max = std_max

    def __call__(self, tensor):
        std = random.uniform(0.0, self.std_max)
        if std == 0:
            return tensor
        return tensor + torch.randn(tensor.size()) * std + self.mean
    
# Custom Dataset class to load images and labels from our cleaned parquet file, also augment training dataset
class AugmentedImageDataset(Dataset):
    def __init__(self, images, labels, is_training):
        self.images = images
        self.labels = labels
        self.is_training = is_training

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        
        self.normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        self.train_transforms = T.Compose([
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomApply([
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.02)
            ], p=0.5),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))], p=0.1),
            T.RandomApply([AddGaussianNoise(mean=0.0, std_max=0.015)], p=0.1),
        ])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx])).float()

        # Apply augmentations
        if self.is_training:
            img = (img * self.std) + self.mean
            img = torch.clamp(img, 0.0, 1.0)

            img = self.train_transforms(img)
            img = torch.clamp(img, 0.0, 1.0)
            
        img = self.normalize(img)
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

        self.block3 = nn.Sequential(self.conv3, self.bnorm3, nn.ReLU(),
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

def load_model():
    ckpt = torch.load(FINETUNE_CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = Given_CNN()
    model.load_state_dict(ckpt["state_dict"])

    return model

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
    
    dataset = AugmentedImageDataset(X, y, shuffle)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                        num_workers=0, generator=g if shuffle else None)

# Initialize model weights with Kaiming He initialization for better convergence
def init_weights(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)
    elif isinstance(m, torch.nn.BatchNorm2d):
        torch.nn.init.ones_(m.weight)
        torch.nn.init.zeros_(m.bias)

# Initialize CNN model and optimizer for deep learning prediction
def initialize_model_and_optimizer(finetune, num_epochs=NUM_EPOCHS):
    if finetune:
        model = load_model()
    else:
        model = Given_CNN()
        model.to(DEVICE)
        model.apply(init_weights)
        
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-2)
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
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
    for thr in np.arange(0.05, 0.96, 0.01):
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

# Deep learning prediction with a simple CNN classifier
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout_seconds", type=int, default=TIME_OUT)
    parser.add_argument("--finetune", type=bool, default=True)
    args = parser.parse_args()

    start_time = time.time()
    deadline = start_time + args.timeout_seconds - 120 

    # 0. Set random seeds for reproducibility
    set_random_seeds()

    # 1. Load prepared training and validation data from ARTIFACTS_DIR
    print("\n=== Loading data ===")

    print_ram_usage("Start")

    train_data = load_data_split("task02/training_data")
    cal_data = load_data_split("task03/calibration_augmented_data")
    val_data = load_data_split("task03/validation_augmented_data")
    
    print_ram_usage("After loading")

    if train_data is None:
        print("ERROR: train.npz not found. Run prepare.py first.")
        sys.exit(1)

    X_tr, y_tr = train_data
    print(f"Train: {X_tr.shape}  real={int((y_tr == 0).sum())}  ai={int((y_tr == 1).sum())}")

    # 2.1. Build model, optimizer
    model, optimizer, scheduler = initialize_model_and_optimizer(finetune=args.finetune)

    # 2.2. Compute class weights for imbalanced training data and initialize loss function
    n_real = int((y_tr == 0).sum())
    n_ai = int((y_tr == 1).sum())
    weights = torch.tensor([n_ai / n_real, 1.0], device=DEVICE)
    loss_fn = initialize_loss_function(weights)

    train_loader = make_loader(X_tr, y_tr, batch_size=BATCH_SIZE)
    cal_loader = make_loader(cal_data[0], cal_data[1], batch_size=BATCH_SIZE)
    val_loader = make_loader(val_data[0], val_data[1], batch_size=BATCH_SIZE)

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
                            os.path.join(TASK03_DIR, "best_model.pt"))
                with open(LOG_FILE, mode="a") as file:
                    file.write(f"New best model saved at epoch {epoch+1} with recall_ai={best_recall:.4f} and FPR={metrics['fpr']:.4f} at threshold={best_thr:.4f}\n")
        
        print(f"Epoch completed in {time.time() - e_start_time:.1f} seconds")
        print_ram_usage(f"After epoch {epoch+1}")

    with open(LOG_FILE, mode="a") as file:
        file.write(f"\n[train_augmented.py] Done in {time.time() - start_time:.1f}s\n")
        file.write(f"Best recall_ai={best_recall:.2f} at threshold={best_thr:.2f}\n")


if __name__ == "__main__":
    main()
