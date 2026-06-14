import argparse
import os
import sys
import time
import random
import psutil

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn

SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "solution")
ARTIFACTS_DIR = os.path.join(SOLUTIONS_DIR, "artifacts")
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
CHECKPOINT_PATH = os.path.join(TASK02_DIR, "best_model.pt")

TASK04_DIR = os.path.join(ARTIFACTS_DIR, "task04")

DEVICE = "cpu"
K = 16
IN_CHANNELS = 3

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
    ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model = Given_CNN()
    model.load_state_dict(ckpt["state_dict"])

    return model

def load_data_split(split_name):
    x_path = os.path.join(ARTIFACTS_DIR, f"{split_name}_X.npy")
    y_path = os.path.join(ARTIFACTS_DIR, f"{split_name}_y.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        return None
    X = np.load(x_path, mmap_mode='r')
    y = np.load(y_path, mmap_mode='r')
    return X, y

def denormalize(img_array):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    
    img = (img_array * std) + mean
    img = np.clip(img, 0.0, 1.0)

    return np.transpose(img, (1, 2, 0)) # C x H x W -> H x W x C

def compute_saliency_map(model, img_tensor, target_class):

    model.eval()

    img_tensor.requires_grad_()
    logits = model(img_tensor)

    if target_class is None:
        target_class = logits.argmax(dim=1).item()

    score = logits[0, target_class]
    model.zero_grad()
    score.backward()

    gradients = img_tensor.grad.data
    saliency_map, _ = torch.max(gradients.abs(), dim=1)
    saliency_map = saliency_map.squeeze()

    return saliency_map, target_class

def plot_saliency_maps(model, data, indices, save_path, num_samples=3):
    X, y = data

    fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(8, 3 * num_samples))

    for i, idx in enumerate(indices):
        img_array = X[idx].astype(np.float32)
        label = y[idx]

        img_tensor = torch.tensor(img_array).unsqueeze(0)

        saliency_map, pred_class = compute_saliency_map(model, img_tensor, label)
        saliency_np = saliency_map.cpu().numpy()
        img_vis = denormalize(img_array)

        ax_img = axes[i][0]
        ax_img.imshow(img_vis)

        # Mark green if prediction is correct, red if incorrect
        color = "green" if pred_class == label else "red"
        ax_img.set_title(f"True: {label} | Pred: {pred_class}", color=color)
        ax_img.axis('off')

        ax_sal = axes[i][1]
        ax_sal.imshow(saliency_np, cmap='hot')
        ax_sal.set_title("Saliency Map")
        ax_sal.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "saliency_output.png"), dpi=300, bbox_inches='tight')
    plt.show()

def compute_occlusion_map(model, img_tensor, target_class, patch_size=16, stride=4):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = nn.functional.softmax(logits, dim=1)
        target_class = probs.argmax(dim=1).item()
        baseline_score = probs[0, target_class].item()

    H, W = img_tensor.shape[2:]
    occlusion_map = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patched_img = img_tensor.clone()

                patched_img[0, :, y:y+patch_size, x:x+patch_size] = 0.0

                out_logits = model(patched_img)
                out_probs = nn.functional.softmax(out_logits, dim=1)
                patched_score = out_probs[0, target_class].item()

                occlusion_map[y:y+patch_size, x:x+patch_size] += patched_score
                counts[y:y+patch_size, x:x+patch_size] += 1.0

    occlusion_map = occlusion_map / (counts + 1e-8)
    importance_map = baseline_score - occlusion_map
    importance_map = np.clip(importance_map, 0, None)

    return importance_map, target_class

def plot_occlusion_maps(model, data, indices, save_path, num_samples=3, patch_size=16, stride=4):
    X, y = data

    fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        img_array = X[idx].astype(np.float32)
        label = y[idx]

        img_tensor = torch.tensor(img_array).unsqueeze(0)
        device = next(model.parameters()).device
        img_tensor = img_tensor.to(device)

        importance_map, pred_class = compute_occlusion_map(
            model, img_tensor, target_class=label, patch_size=patch_size, stride=stride
        )

        img_vis = denormalize(img_array)

        ax_img = axes[i][0]
        ax_img.imshow(img_vis)
        color = "green" if pred_class == label else "red"
        ax_img.set_title(f"True: {label} | Pred: {pred_class}", color=color)
        ax_img.axis('off')

        ax_occ = axes[i][1]
        ax_occ.imshow(importance_map, cmap='jet')
        ax_occ.set_title("Occlusion Importance Map")
        ax_occ.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "occlusion_output.png"), dpi=300, bbox_inches='tight')
    plt.show()

def main(num_samples = 5):
    start_time = time.time()

    os.makedirs(TASK04_DIR, exist_ok=True)

    model = load_model()
    data = load_data_split("task02/prepared_validation_data")

    X, y = data
    dataset_size = len(X)
    indices = random.sample(range(dataset_size), num_samples)

    plot_saliency_maps(model, data, indices, TASK04_DIR, num_samples=num_samples)

    plot_occlusion_maps(model, data, indices, TASK04_DIR, num_samples=num_samples,
                        patch_size=16, stride=4)


if __name__ == "__main__":
    main()
