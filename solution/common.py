"""common.py - Shared model, data loading, and evaluation utilities.

Imported by train.py, train_augmented.py, predict.py, predict_augmented.py,
and explainability.py so the CNN definition and evaluation logic have a
single source of truth.
"""

import os
import random

import numpy as np
from sklearn.metrics import confusion_matrix

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


SEED = 42
BATCH_SIZE = 64
MAX_FPR = 0.20
DEVICE = "cpu"

K = 16
IN_CHANNELS = 3

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.from_numpy(np.array(self.images[idx])).float()
        return img, torch.tensor(self.labels[idx]).long()


class Given_CNN(nn.Module):
    def __init__(self, in_channels=IN_CHANNELS, num_classes=2, k=K):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(k)
        self.pool1 = nn.MaxPool2d(2)
        self.block1 = nn.Sequential(self.conv1, self.bnorm1, nn.ReLU(), self.pool1)

        self.conv2 = nn.Conv2d(k, 2 * k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(2 * k)
        self.pool2 = nn.MaxPool2d(2)
        self.block2 = nn.Sequential(self.conv2, self.bnorm2, nn.ReLU(), self.pool2)

        self.conv3 = nn.Conv2d(2 * k, 4 * k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(4 * k)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = nn.Conv2d(4 * k, 4 * k, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm4 = nn.BatchNorm2d(4 * k)
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        self.block3 = nn.Sequential(
            self.conv3, self.bnorm3, nn.ReLU(), self.pool3,
            self.conv4, self.bnorm4, nn.ReLU(),
            self.global_pool,
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(4 * k, num_classes),
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.classifier(x)
        return x


def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_runtime():
    """Determinism + thread cap. Call once at top-of-main in training scripts."""
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))


def load_data_split(split_name):
    x_path = os.path.join(ARTIFACTS_DIR, f"{split_name}_X.npy")
    y_path = os.path.join(ARTIFACTS_DIR, f"{split_name}_y.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        return None
    X = np.load(x_path, mmap_mode="r")
    y = np.load(y_path, mmap_mode="r")
    return X, y


def make_loader(X, y, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, dataset_cls=None):
    g = torch.Generator()
    g.manual_seed(SEED)

    dataset_cls = dataset_cls or ImageDataset
    dataset = dataset_cls(X, y)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=g if shuffle else None,
        persistent_workers=(num_workers > 0),
    )


def load_checkpoint(path, model_cls=Given_CNN, device=DEVICE):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = model_cls()
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    threshold = float(ckpt.get("threshold", 0.5))
    return model, threshold, ckpt


def get_probs(model, loader, device=DEVICE):
    model.eval()
    all_p, all_y = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            p = torch.softmax(model(X_batch.to(device)), dim=1)[:, 1].cpu().numpy()
            all_p.append(p)
            all_y.append(y_batch.numpy())
    return np.concatenate(all_p), np.concatenate(all_y)


def calibrate_threshold(model, cal_loader, max_fpr=MAX_FPR, device=DEVICE):
    probs, y_true = get_probs(model, cal_loader, device=device)
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


def evaluate(model, val_loader, threshold, split_name="validation", device=DEVICE):
    probs, y_true = get_probs(model, val_loader, device=device)
    y_pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / len(y_true) if len(y_true) > 0 else 0.0
    print(f"  {split_name}: FPR={fpr:.4f}  Recall_ai={recall:.4f}  Accuracy={acc:.4f}  (threshold={threshold:.4f})")
    return {"fpr": fpr, "recall_ai": recall, "accuracy": acc}


def compute_class_weights(y, balanced=True, device=DEVICE):
    n_real = int((y == 0).sum())
    n_ai = int((y == 1).sum())
    if balanced or n_real == 0 or n_ai == 0:
        return torch.tensor([1.0, 1.0], device=device)
    n = n_real + n_ai
    return torch.tensor([n / (2 * n_real), n / (2 * n_ai)], device=device)


def train_one_epoch(model, optimizer, loss_fn, train_loader, device=DEVICE):
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
    return sum(losses) / len(losses)
