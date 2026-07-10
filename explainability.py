"""explainability.py - Occlusion, Grad-CAM, and error analysis (Task 4)."""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

SOLUTIONS_DIR = os.path.join(os.path.dirname(__file__), "solution")
sys.path.insert(0, SOLUTIONS_DIR)

from common import Given_CNN, load_data_split, set_seed  # noqa: E402

ARTIFACTS_DIR = os.path.join(SOLUTIONS_DIR, "artifacts")
TASK02_DIR = os.path.join(ARTIFACTS_DIR, "task02")
TASK04_DIR = os.path.join(ARTIFACTS_DIR, "task04")
CHECKPOINT_PATH = os.path.join(TASK02_DIR, "best_model.pt")

DEVICE = "cpu"


def load_model():
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
    model = Given_CNN()
    model.load_state_dict(ckpt["state_dict"])
    model.to(DEVICE)
    threshold = float(ckpt.get("threshold", 0.5))
    return model, threshold


def denormalize(img_array):
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img = (img_array * std) + mean
    img = np.clip(img, 0.0, 1.0)
    return np.transpose(img, (1, 2, 0))


def predict_class(model, img_tensor):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        return int(logits.argmax(dim=1).item())


def get_val_predictions(model, data, threshold, batch_size=128):
    """Forward-pass the whole validation set once; return probs and preds."""
    X, y = data
    model.eval()
    all_probs = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            batch = torch.from_numpy(np.array(X[start:start + batch_size])).float().to(DEVICE)
            probs = torch.softmax(model(batch), dim=1)[:, 1].cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs, axis=0)
    preds = (probs >= threshold).astype(np.int64)
    labels = np.asarray(y).astype(np.int64)
    return probs, preds, labels


# --- Occlusion --------------------------------------------------------------

def compute_occlusion_map(model, img_tensor, target_class, patch_size=16, stride=4):
    model.eval()
    with torch.no_grad():
        logits = model(img_tensor)
        probs = nn.functional.softmax(logits, dim=1)
        baseline_score = probs[0, target_class].item()

    H, W = img_tensor.shape[2:]
    occlusion_map = np.zeros((H, W), dtype=np.float32)
    counts = np.zeros((H, W), dtype=np.float32)

    with torch.no_grad():
        for y in range(0, H - patch_size + 1, stride):
            for x in range(0, W - patch_size + 1, stride):
                patched = img_tensor.clone()
                patched[0, :, y:y + patch_size, x:x + patch_size] = 0.0
                out_probs = nn.functional.softmax(model(patched), dim=1)
                patched_score = out_probs[0, target_class].item()
                occlusion_map[y:y + patch_size, x:x + patch_size] += patched_score
                counts[y:y + patch_size, x:x + patch_size] += 1.0

    occlusion_map = occlusion_map / (counts + 1e-8)
    return np.clip(baseline_score - occlusion_map, 0, None)


def plot_occlusion_maps(model, data, indices, save_path, num_samples=3, patch_size=16, stride=4):
    X, y = data
    fig, axes = plt.subplots(nrows=num_samples, ncols=2, figsize=(8, 4 * num_samples))
    if num_samples == 1:
        axes = [axes]

    for i, idx in enumerate(indices):
        img_array = X[idx].astype(np.float32)
        label = int(y[idx])
        img_tensor = torch.tensor(img_array).unsqueeze(0).to(DEVICE)

        pred_class = predict_class(model, img_tensor)
        importance = compute_occlusion_map(model, img_tensor, target_class=label,
                                           patch_size=patch_size, stride=stride)

        color = "green" if pred_class == label else "red"
        axes[i][0].imshow(denormalize(img_array))
        axes[i][0].set_title(f"True: {label} | Pred: {pred_class}", color=color)
        axes[i][0].axis("off")

        axes[i][1].imshow(importance, cmap="jet")
        axes[i][1].set_title("Occlusion (w.r.t. true class)")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "occlusion_output.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Grad-CAM ---------------------------------------------------------------

def compute_gradcam(model, img_tensor, target_class, target_layer):
    """Grad-CAM: weight last-conv feature maps by grad-of-target-class w.r.t. them."""
    model.eval()
    activations, gradients = {}, {}

    def fwd_hook(_module, _inp, out):
        activations["value"] = out.detach()

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        logits = model(img_tensor)
        model.zero_grad()
        logits[0, target_class].backward()

        acts = activations["value"][0]        # (C, h, w)
        grads = gradients["value"][0]         # (C, h, w)
        weights = grads.mean(dim=(1, 2))      # (C,)
        cam = torch.relu((weights[:, None, None] * acts).sum(dim=0))
    finally:
        h1.remove()
        h2.remove()

    cam = cam.cpu().numpy()
    if cam.max() > 0:
        cam = cam / cam.max()

    H, W = img_tensor.shape[2:]
    cam_tensor = torch.from_numpy(cam)[None, None].float()
    cam_up = nn.functional.interpolate(cam_tensor, size=(H, W), mode="bilinear", align_corners=False)
    return cam_up.squeeze().numpy()


def plot_gradcam(model, data, indices, save_path, num_samples=5):
    X, y = data
    target_layer = model.conv4
    fig, axes = plt.subplots(nrows=num_samples, ncols=3, figsize=(10, 3 * num_samples))
    if num_samples == 1:
        axes = axes[None, :]

    for i, idx in enumerate(indices):
        img_array = X[idx].astype(np.float32)
        label = int(y[idx])
        img_tensor = torch.tensor(img_array).unsqueeze(0).to(DEVICE)

        pred_class = predict_class(model, img_tensor)
        cam = compute_gradcam(model, img_tensor, target_class=label, target_layer=target_layer)
        img_vis = denormalize(img_array)

        color = "green" if pred_class == label else "red"
        axes[i][0].imshow(img_vis)
        axes[i][0].set_title(f"True: {label} | Pred: {pred_class}", color=color)
        axes[i][0].axis("off")

        axes[i][1].imshow(cam, cmap="jet")
        axes[i][1].set_title("Grad-CAM")
        axes[i][1].axis("off")

        axes[i][2].imshow(img_vis)
        axes[i][2].imshow(cam, cmap="jet", alpha=0.45)
        axes[i][2].set_title("Overlay")
        axes[i][2].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "gradcam_output.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Failure cases (TP / FP / FN) -------------------------------------------

def sample_group(rng, idx_pool, n):
    if len(idx_pool) == 0:
        return []
    n = min(n, len(idx_pool))
    return list(rng.choice(idx_pool, size=n, replace=False))


def plot_failure_cases(model, data, preds, labels, save_path, per_group=3):
    """Grid of TP / FP / FN examples with Grad-CAM overlays."""
    X, _ = data
    target_layer = model.conv4
    rng = np.random.default_rng(42)

    tp = np.where((preds == 1) & (labels == 1))[0]
    fp = np.where((preds == 1) & (labels == 0))[0]
    fn = np.where((preds == 0) & (labels == 1))[0]

    groups = [("TP (AI→AI)", sample_group(rng, tp, per_group)),
              ("FP (Real→AI)", sample_group(rng, fp, per_group)),
              ("FN (AI→Real)", sample_group(rng, fn, per_group))]

    n_rows = per_group
    n_cols = len(groups)
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(4 * n_cols, 4 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    for col, (title, idxs) in enumerate(groups):
        for row in range(n_rows):
            ax = axes[row][col]
            if row >= len(idxs):
                ax.axis("off")
                continue
            idx = idxs[row]
            img_array = X[idx].astype(np.float32)
            label = int(labels[idx])
            img_tensor = torch.tensor(img_array).unsqueeze(0).to(DEVICE)

            cam = compute_gradcam(model, img_tensor, target_class=label, target_layer=target_layer)
            ax.imshow(denormalize(img_array))
            ax.imshow(cam, cmap="jet", alpha=0.45)
            ax.set_title(title if row == 0 else "", fontsize=11)
            ax.axis("off")

    plt.suptitle("Failure-case analysis (Grad-CAM overlay)", y=1.0)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "failure_cases.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Average attention: real vs AI ------------------------------------------

def plot_avg_attention_real_vs_ai(model, data, save_path, n_per_class=40, patch_size=16, stride=8):
    """Average Grad-CAM and occlusion maps over N real and N AI samples."""
    X, y = data
    labels = np.asarray(y).astype(np.int64)
    real_idx = np.where(labels == 0)[0]
    ai_idx = np.where(labels == 1)[0]

    rng = np.random.default_rng(42)
    real_pick = rng.choice(real_idx, size=min(n_per_class, len(real_idx)), replace=False)
    ai_pick = rng.choice(ai_idx, size=min(n_per_class, len(ai_idx)), replace=False)

    target_layer = model.conv4

    def average_maps(indices):
        sum_cam = None
        sum_occ = None
        for idx in indices:
            img_tensor = torch.tensor(X[idx].astype(np.float32)).unsqueeze(0).to(DEVICE)
            label = int(labels[idx])
            cam = compute_gradcam(model, img_tensor, target_class=label, target_layer=target_layer)
            occ = compute_occlusion_map(model, img_tensor, target_class=label,
                                        patch_size=patch_size, stride=stride)
            sum_cam = cam if sum_cam is None else sum_cam + cam
            sum_occ = occ if sum_occ is None else sum_occ + occ
        return sum_cam / len(indices), sum_occ / len(indices)

    print(f"  Averaging Grad-CAM+occlusion over {len(real_pick)} real / {len(ai_pick)} AI samples...")
    real_cam, real_occ = average_maps(real_pick)
    ai_cam, ai_occ = average_maps(ai_pick)

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(9, 9))
    for ax, arr, title in [
        (axes[0][0], real_cam, f"Avg Grad-CAM — Real (n={len(real_pick)})"),
        (axes[0][1], ai_cam, f"Avg Grad-CAM — AI (n={len(ai_pick)})"),
        (axes[1][0], real_occ, f"Avg Occlusion — Real (n={len(real_pick)})"),
        (axes[1][1], ai_occ, f"Avg Occlusion — AI (n={len(ai_pick)})"),
    ]:
        im = ax.imshow(arr, cmap="jet")
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Where does the model look? (class-averaged attention)")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "avg_attention_real_vs_ai.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Confidence distribution ------------------------------------------------

def plot_confidence_distribution(probs, labels, threshold, save_path):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(probs[labels == 0], bins=40, alpha=0.6, label="Real (class 0)", color="tab:blue")
    ax.hist(probs[labels == 1], bins=40, alpha=0.6, label="AI (class 1)", color="tab:orange")
    ax.axvline(threshold, color="k", linestyle="--", label=f"threshold = {threshold:.3f}")
    ax.set_xlabel("P(AI)")
    ax.set_ylabel("Count")
    ax.set_title("Predicted probability distribution by true class")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "confidence_distribution.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# --- Main -------------------------------------------------------------------

def main(num_samples=5):
    set_seed()
    os.makedirs(TASK04_DIR, exist_ok=True)

    model, threshold = load_model()
    data = load_data_split("task02/validation_data")
    if data is None:
        print("ERROR: validation_data_*.npy not found. Run prepare.py first.")
        sys.exit(1)

    X, _ = data
    indices = random.sample(range(len(X)), num_samples)

    print("Plotting per-sample occlusion and Grad-CAM...")
    plot_occlusion_maps(model, data, indices, TASK04_DIR, num_samples=num_samples,
                        patch_size=16, stride=4)
    plot_gradcam(model, data, indices, TASK04_DIR, num_samples=num_samples)

    print("Running validation forward pass for error analysis...")
    probs, preds, labels = get_val_predictions(model, data, threshold)

    print("Plotting confidence distribution...")
    plot_confidence_distribution(probs, labels, threshold, TASK04_DIR)

    print("Plotting failure cases (TP/FP/FN)...")
    plot_failure_cases(model, data, preds, labels, TASK04_DIR, per_group=3)

    print("Plotting average attention: real vs AI...")
    plot_avg_attention_real_vs_ai(model, data, TASK04_DIR, n_per_class=40)

    print(f"\nDone. Outputs written to {TASK04_DIR}")


if __name__ == "__main__":
    main()
