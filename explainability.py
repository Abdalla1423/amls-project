"""explainability.py - Task 1.4: Model Explainability Analysis.

Generates GradCAM heatmaps and error analysis for the Task 2 model.
Produces figures for the report showing what the model attends to
for real vs AI-generated images, and analyzes failure modes (FP/FN).

Usage: python explainability.py
Output: figures saved to explainability_output/
"""

import io
import os

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SOLUTION_DIR = os.path.join(os.path.dirname(__file__), "solution")
ARTIFACTS_DIR = os.path.join(SOLUTION_DIR, "artifacts")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "explainability_output")

TARGET_SIZE = (64, 64)
K = 32


# ---------------------------------------------------------------------------
# Model (must match train.py architecture exactly)
# ---------------------------------------------------------------------------
def build_cnn(k=K):
    return nn.Sequential(
        # Block 1: indices 0-3
        nn.Conv2d(3, k, 3, padding=1), nn.BatchNorm2d(k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 2: indices 4-7
        nn.Conv2d(k, 2*k, 3, padding=1), nn.BatchNorm2d(2*k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 3: indices 8-11
        nn.Conv2d(2*k, 4*k, 3, padding=1), nn.BatchNorm2d(4*k), nn.ReLU(),
        nn.MaxPool2d(2),
        # Block 4: indices 12-15
        nn.Conv2d(4*k, 4*k, 3, padding=1), nn.BatchNorm2d(4*k), nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        # Classifier: indices 16-18
        nn.Flatten(), nn.Dropout(0.3), nn.Linear(4*k, 2),
    )


# ---------------------------------------------------------------------------
# GradCAM for nn.Sequential
# ---------------------------------------------------------------------------
class GradCAM:
    """GradCAM for the last conv block of an nn.Sequential model."""

    def __init__(self, model, target_layer_idx=14):
        """target_layer_idx: index of the ReLU after the last conv (block 4)."""
        self.model = model
        self.target_layer_idx = target_layer_idx
        self.activations = None
        self.gradients = None

        # Register hooks on the target layer
        target_layer = model[target_layer_idx]
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, target_class=None):
        """Compute GradCAM heatmap for input tensor x (1, 3, H, W).
        
        Returns heatmap as numpy array (H, W) in [0, 1].
        """
        self.model.eval()
        x = x.requires_grad_(True)
        logits = self.model(x)

        if target_class is None:
            target_class = logits.argmax(dim=1).item()

        self.model.zero_grad()
        logits[0, target_class].backward()

        # Global average pooling of gradients → channel weights
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)
        cam = F.relu(cam)
        cam = cam.squeeze().numpy()

        # Normalize to [0, 1]
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input size
        cam = cv2.resize(cam, (x.shape[3], x.shape[2]))
        return cam


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def tensor_to_image(t):
    """Convert (3, H, W) float32 tensor to (H, W, 3) uint8 numpy array."""
    img = t.detach().numpy().transpose(1, 2, 0)
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img


def overlay_heatmap(img, heatmap, alpha=0.4):
    """Overlay heatmap on image. Both should be numpy arrays."""
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    overlay = (alpha * heatmap_colored + (1 - alpha) * img).astype(np.uint8)
    return overlay


def get_probs_and_preds(model, X, y, threshold):
    """Run inference and return probs, preds, labels."""
    model.eval()
    ds = torch.utils.data.TensorDataset(
        torch.from_numpy(X), torch.from_numpy(y).long()
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=64, shuffle=False)
    all_p, all_y = [], []
    with torch.no_grad():
        for xb, yb in loader:
            p = torch.softmax(model(xb), dim=1)[:, 1]
            all_p.append(p)
            all_y.append(yb)
    probs = torch.cat(all_p).numpy()
    labels = torch.cat(all_y).numpy()
    preds = (probs >= threshold).astype(int)
    return probs, preds, labels


# ---------------------------------------------------------------------------
# Figure 1: GradCAM grid — TP, TN, FP, FN examples
# ---------------------------------------------------------------------------
def plot_gradcam_grid(model, gradcam, X, probs, preds, labels, output_dir):
    """Plot a 4×4 grid: rows = TP/TN/FP/FN, cols = 4 examples each."""
    categories = {
        "True Positive (AI→AI)": (labels == 1) & (preds == 1),
        "True Negative (Real→Real)": (labels == 0) & (preds == 0),
        "False Positive (Real→AI)": (labels == 0) & (preds == 1),
        "False Negative (AI→Real)": (labels == 1) & (preds == 0),
    }

    fig, axes = plt.subplots(4, 4, figsize=(14, 14))
    fig.suptitle("GradCAM Explanations by Prediction Category", fontsize=14, y=0.98)

    for row, (cat_name, mask) in enumerate(categories.items()):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            for col in range(4):
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(cat_name, fontsize=10)
            continue

        # Pick examples spread across confidence range
        cat_probs = probs[indices]
        sorted_idx = np.argsort(cat_probs)
        n = len(sorted_idx)
        pick = [sorted_idx[int(i * n / 4)] for i in range(min(4, n))]

        for col, idx_in_cat in enumerate(pick):
            global_idx = indices[idx_in_cat]
            x_tensor = torch.from_numpy(X[global_idx:global_idx+1])
            heatmap = gradcam(x_tensor, target_class=1)  # explain AI class
            img = tensor_to_image(x_tensor.squeeze(0))
            overlay = overlay_heatmap(img, heatmap)

            axes[row, col].imshow(overlay)
            axes[row, col].set_title(
                f"p={probs[global_idx]:.2f}", fontsize=9
            )
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(cat_name, fontsize=10, rotation=90, labelpad=60)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = os.path.join(output_dir, "gradcam_grid.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 2: GradCAM comparison — Real vs AI (correct predictions only)
# ---------------------------------------------------------------------------
def plot_real_vs_ai_attention(model, gradcam, X, probs, preds, labels, output_dir):
    """Show 2 rows: top row = correctly classified real, bottom = correctly classified AI."""
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    fig.suptitle("Model Attention: Real vs AI-Generated Images", fontsize=14, y=1.02)

    for row, (label_val, title) in enumerate([(0, "Real Images (TN)"), (1, "AI Images (TP)")]):
        mask = (labels == label_val) & (preds == label_val)
        indices = np.where(mask)[0]
        np.random.seed(42)
        chosen = np.random.choice(indices, size=min(6, len(indices)), replace=False)

        for col, idx in enumerate(chosen):
            x_tensor = torch.from_numpy(X[idx:idx+1])

            # Original image
            if col == 0:
                axes[row, col].set_ylabel(title, fontsize=11)

            heatmap = gradcam(x_tensor, target_class=1)
            img = tensor_to_image(x_tensor.squeeze(0))
            overlay = overlay_heatmap(img, heatmap)
            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"p(AI)={probs[idx]:.2f}", fontsize=9)
            axes[row, col].axis("off")

    plt.tight_layout()
    path = os.path.join(output_dir, "real_vs_ai_attention.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 3: Confidence distribution for TP/TN/FP/FN
# ---------------------------------------------------------------------------
def plot_confidence_distribution(probs, preds, labels, threshold, output_dir):
    """Histogram of model confidence split by category."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # By true label
    axes[0].hist(probs[labels == 0], bins=50, alpha=0.7, label="Real", color="blue")
    axes[0].hist(probs[labels == 1], bins=50, alpha=0.7, label="AI", color="red")
    axes[0].axvline(threshold, color="black", linestyle="--", label=f"thr={threshold:.2f}")
    axes[0].set_xlabel("P(AI)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Confidence Distribution by True Label")
    axes[0].legend()

    # By prediction outcome
    tp = (labels == 1) & (preds == 1)
    tn = (labels == 0) & (preds == 0)
    fp = (labels == 0) & (preds == 1)
    fn = (labels == 1) & (preds == 0)
    for mask, name, color in [
        (tp, "TP", "green"), (tn, "TN", "blue"),
        (fp, "FP", "orange"), (fn, "FN", "red"),
    ]:
        if mask.any():
            axes[1].hist(probs[mask], bins=30, alpha=0.6, label=f"{name} ({mask.sum()})", color=color)
    axes[1].axvline(threshold, color="black", linestyle="--")
    axes[1].set_xlabel("P(AI)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Confidence Distribution by Outcome")
    axes[1].legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "confidence_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 4: FP/FN analysis — show the hardest cases
# ---------------------------------------------------------------------------
def plot_failure_cases(model, gradcam, X, probs, preds, labels, output_dir):
    """Show the most confident FPs and FNs with GradCAM."""
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle("Failure Analysis: Most Confident Errors", fontsize=14, y=1.02)

    for row, (title, mask, sort_asc) in enumerate([
        ("False Positives (Real→AI, highest P(AI))", (labels == 0) & (preds == 1), False),
        ("False Negatives (AI→Real, lowest P(AI))", (labels == 1) & (preds == 0), True),
    ]):
        indices = np.where(mask)[0]
        if len(indices) == 0:
            for col in range(5):
                axes[row, col].axis("off")
            axes[row, 0].set_ylabel(title, fontsize=9)
            continue

        # Sort by confidence (most confident errors first)
        cat_probs = probs[indices]
        if sort_asc:
            order = np.argsort(cat_probs)  # lowest p(AI) first for FN
        else:
            order = np.argsort(-cat_probs)  # highest p(AI) first for FP

        for col in range(min(5, len(order))):
            global_idx = indices[order[col]]
            x_tensor = torch.from_numpy(X[global_idx:global_idx+1])
            heatmap = gradcam(x_tensor, target_class=1)
            img = tensor_to_image(x_tensor.squeeze(0))
            overlay = overlay_heatmap(img, heatmap)

            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"p(AI)={probs[global_idx]:.3f}", fontsize=9)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(title, fontsize=9, rotation=90, labelpad=60)

    plt.tight_layout()
    path = os.path.join(output_dir, "failure_cases.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Occlusion sensitivity
# ---------------------------------------------------------------------------
# Idea: GradCAM is gradient-based and can mislead (saturated activations,
# noisy gradients). Occlusion is a model-agnostic, purely forward-pass method:
# slide a grey patch across the image, re-run the model, and record how much
# P(AI) drops at each location. Where P drops the most = that region was the
# strongest evidence for "AI". Two independent methods that agree are more
# trustworthy than either alone.
def occlusion_map(model, x, patch_size=8, stride=4, occluder_value=0.5,
                  target_class=1):
    """Compute occlusion sensitivity map for one image.

    Args:
        x: tensor (1, 3, H, W) in [0, 1].
        patch_size: side of the grey square (8 ≈ 1/8 of a 64-px image).
        stride: how many pixels we move between patches (smaller = smoother
                map but more forward passes; 4 → 15×15 = 225 evaluations).
        occluder_value: pixel value to fill the patch with (0.5 = mid-grey,
                neutral; 0.0 black or 1.0 white would inject their own bias).
        target_class: 1 = explain the AI logit.

    Returns:
        sensitivity map (H, W) where higher = bigger drop in P(AI) when
        that region is hidden = that region was more important for the
        AI prediction. Normalised to [0, 1] for display.
    """
    model.eval()
    _, _, H, W = x.shape

    # Baseline probability with no occlusion.
    with torch.no_grad():
        p_base = torch.softmax(model(x), dim=1)[0, target_class].item()

    # Build a list of all (top, left) positions for the patches.
    # We iterate by stride and let the last position clamp to the image edge
    # so the entire image is covered even when (H - patch_size) is not a
    # multiple of stride.
    positions = []
    for top in range(0, H - patch_size + 1, stride):
        for left in range(0, W - patch_size + 1, stride):
            positions.append((top, left))

    # Build a batch of occluded copies — much faster than one-at-a-time.
    occluded = x.repeat(len(positions), 1, 1, 1).clone()
    for i, (top, left) in enumerate(positions):
        occluded[i, :, top:top + patch_size, left:left + patch_size] = (
            occluder_value
        )

    # Forward pass on the whole batch in chunks (memory safety on CPU).
    drops = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)
    chunk = 64
    with torch.no_grad():
        for start in range(0, len(positions), chunk):
            xb = occluded[start:start + chunk]
            p = torch.softmax(model(xb), dim=1)[:, target_class].numpy()
            for j, (top, left) in enumerate(positions[start:start + chunk]):
                # Spread the drop over the patch footprint, average across
                # overlapping patches via `counts`.
                drop = p_base - p[j]
                drops[top:top + patch_size, left:left + patch_size] += drop
                counts[top:top + patch_size, left:left + patch_size] += 1

    # Average: every pixel was covered by `counts[i,j]` patches.
    sensitivity = drops / np.maximum(counts, 1)

    # Clip negatives (occluding can occasionally *increase* P(AI), which we
    # do not visualise as "important") and normalise to [0, 1].
    sensitivity = np.clip(sensitivity, 0, None)
    if sensitivity.max() > 0:
        sensitivity = sensitivity / sensitivity.max()
    return sensitivity, p_base


# ---------------------------------------------------------------------------
# Figure 6: Occlusion sensitivity — TP vs FN (mirrors failure_cases.png)
# ---------------------------------------------------------------------------
def plot_occlusion_failure_cases(model, X, probs, preds, labels, output_dir):
    """Compare occlusion maps for confident TPs vs the worst FNs.

    Why TP vs FN (not FP vs FN like the GradCAM version)?
    Because the question we're answering with occlusion is the cross-check:
    when the model gets an AI image right (TP) does it really hinge on a
    specific region, and when it misses (FN) is the evidence absent or
    distributed everywhere? That's a sharper diagnostic than FP vs FN.
    """
    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    fig.suptitle(
        "Occlusion Sensitivity: True Positives vs False Negatives",
        fontsize=14, y=1.02,
    )

    rows = [
        ("True Positives (AI→AI, highest P(AI))",
         (labels == 1) & (preds == 1), False),
        ("False Negatives (AI→Real, lowest P(AI))",
         (labels == 1) & (preds == 0), True),
    ]

    for row, (title, mask, sort_asc) in enumerate(rows):
        indices = np.where(mask)[0]
        cat_probs = probs[indices]
        order = np.argsort(cat_probs) if sort_asc else np.argsort(-cat_probs)

        for col in range(min(5, len(order))):
            global_idx = indices[order[col]]
            x_tensor = torch.from_numpy(X[global_idx:global_idx + 1])

            sens, p_base = occlusion_map(model, x_tensor, target_class=1)
            img = tensor_to_image(x_tensor.squeeze(0))
            overlay = overlay_heatmap(img, sens)

            axes[row, col].imshow(overlay)
            axes[row, col].set_title(f"p(AI)={p_base:.3f}", fontsize=9)
            axes[row, col].axis("off")

        axes[row, 0].set_ylabel(title, fontsize=9, rotation=90, labelpad=60)

    plt.tight_layout()
    path = os.path.join(output_dir, "occlusion_failure_cases.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 7: Average occlusion map — does it agree with average GradCAM?
# ---------------------------------------------------------------------------
def plot_average_occlusion(model, X, labels, output_dir, n_samples=60):
    """Average occlusion sensitivity over Real vs AI images.

    Smaller n_samples than average_heatmaps (200) because each occlusion
    map costs ~225 forward passes. Sixty samples per class is enough to
    see whether the spatial pattern matches GradCAM's average.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Average Occlusion Sensitivity: Real vs AI-Generated",
                 fontsize=14)

    for col, (label_val, title) in enumerate(
        [(0, "Real Images"), (1, "AI-Generated Images")]
    ):
        indices = np.where(labels == label_val)[0]
        np.random.seed(42)
        chosen = np.random.choice(
            indices, size=min(n_samples, len(indices)), replace=False,
        )

        avg = np.zeros(TARGET_SIZE, dtype=np.float32)
        for k, idx in enumerate(chosen):
            x_tensor = torch.from_numpy(X[idx:idx + 1])
            sens, _ = occlusion_map(model, x_tensor, target_class=1)
            avg += sens
            if (k + 1) % 10 == 0:
                print(f"    {title}: {k + 1}/{len(chosen)}")
        avg /= len(chosen)

        im = axes[col].imshow(avg, cmap="jet", vmin=0, vmax=avg.max())
        axes[col].set_title(f"{title} (n={len(chosen)})")
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, "average_occlusion.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Figure 5: Average heatmap — what does the model attend to on average?
# ---------------------------------------------------------------------------
def plot_average_heatmaps(model, gradcam, X, labels, output_dir, n_samples=200):
    """Compute and compare average GradCAM heatmaps for real vs AI images."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle("Average GradCAM Heatmap: Real vs AI-Generated", fontsize=14)

    for col, (label_val, title) in enumerate([(0, "Real Images"), (1, "AI-Generated Images")]):
        indices = np.where(labels == label_val)[0]
        np.random.seed(42)
        chosen = np.random.choice(indices, size=min(n_samples, len(indices)), replace=False)

        avg_heatmap = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.float32)
        for idx in chosen:
            x_tensor = torch.from_numpy(X[idx:idx+1])
            heatmap = gradcam(x_tensor, target_class=1)
            avg_heatmap += heatmap
        avg_heatmap /= len(chosen)

        im = axes[col].imshow(avg_heatmap, cmap="jet", vmin=0, vmax=avg_heatmap.max())
        axes[col].set_title(f"{title} (n={len(chosen)})")
        axes[col].axis("off")
        plt.colorbar(im, ax=axes[col], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, "average_heatmaps.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load model
    ckpt_path = os.path.join(ARTIFACTS_DIR, "best_model.pt")
    if not os.path.exists(ckpt_path):
        print(f"ERROR: {ckpt_path} not found. Run train.py first.")
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = build_cnn(K)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    threshold = ckpt.get("threshold", 0.5)
    print(f"Loaded model: threshold={threshold:.4f}, "
          f"recall_ai={ckpt.get('recall_ai', 'N/A')}, epoch={ckpt.get('epoch', 'N/A')}")

    # Load validation data
    val_path = os.path.join(ARTIFACTS_DIR, "validation.npz")
    if not os.path.exists(val_path):
        print(f"ERROR: {val_path} not found. Run prepare.py first.")
        return

    val = np.load(val_path)
    X, y = val["X"], val["y"]
    print(f"Validation: {X.shape}, real={int((y==0).sum())}, ai={int((y==1).sum())}")

    # Run inference
    probs, preds, labels = get_probs_and_preds(model, X, y, threshold)
    tn, fp, fn, tp = confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    fpr = fp / (fp + tn)
    recall = tp / (tp + fn)
    print(f"Results: recall_ai={recall:.4f}, FPR={fpr:.4f}, "
          f"TP={tp}, TN={tn}, FP={fp}, FN={fn}")

    # Initialize GradCAM (target = ReLU after last conv, index 14)
    gradcam = GradCAM(model, target_layer_idx=14)

    # Generate all figures
    print("\nGenerating figures...")

    print("1. GradCAM grid (TP/TN/FP/FN)...")
    plot_gradcam_grid(model, gradcam, X, probs, preds, labels, OUTPUT_DIR)

    print("2. Real vs AI attention comparison...")
    plot_real_vs_ai_attention(model, gradcam, X, probs, preds, labels, OUTPUT_DIR)

    print("3. Confidence distributions...")
    plot_confidence_distribution(probs, preds, labels, threshold, OUTPUT_DIR)

    print("4. Failure case analysis...")
    plot_failure_cases(model, gradcam, X, probs, preds, labels, OUTPUT_DIR)

    print("5. Average heatmaps...")
    plot_average_heatmaps(model, gradcam, X, labels, OUTPUT_DIR)

    print("6. Occlusion: TP vs FN failure cases...")
    plot_occlusion_failure_cases(model, X, probs, preds, labels, OUTPUT_DIR)

    print("7. Average occlusion sensitivity (Real vs AI)...")
    plot_average_occlusion(model, X, labels, OUTPUT_DIR)

    print(f"\nAll figures saved to {OUTPUT_DIR}/")
    print("Files:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        print(f"  {f}")


if __name__ == "__main__":
    main()
