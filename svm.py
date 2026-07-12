import os
import io
import time
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, UnidentifiedImageError
import torchvision.transforms.functional as tf
 
from skimage.feature import hog
import torch
import glob
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")
TASK01_DIR = os.path.join(ARTIFACTS_DIR, "task01")

TIME_OUT = 600
SEED = 42

IMAGE_SIZE = 64
BATCH_SIZE = 64

C_GRID = [0.0001, 0.001, 0.01, 0.1, 1]
K_FOLDS = 5

HOG_PARAMS = dict(
    orientations=16, 
    pixels_per_cell=(4, 4),
    cells_per_block=(1, 1),
    channel_axis=-1
)

class ImageDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.as_tensor(self.labels[idx], dtype=torch.long)
        return img, label

# Set random seeds for reproducibility
def set_random_seeds(seed=SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(8, os.cpu_count() or 1))

# Prepare the data loader
def make_loader(X, y, batch_size = BATCH_SIZE):
    dataset = ImageDataset(X, y)
    return DataLoader(dataset, batch_size=batch_size, num_workers=0)

# Center-resize an image to (target_h, target_w) and return it in CHW format
def preprocess_single_image(img_bytes, target_h = IMAGE_SIZE, target_w = IMAGE_SIZE):
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((target_w, target_h), Image.BICUBIC)
    img_arr = np.array(img)
    return np.transpose(img_arr, (2, 0, 1)) 

# Extract features and labels from the dataframe
def load_dataframe_to_arrays(df):

    processed_images = []
    labels = []
 
    for row in df.itertuples(index=False):
        try:
            processed_img = preprocess_single_image(row.image)
        except UnidentifiedImageError as e:
            print(f"Skipping unreadable image: {e}")
            continue
 
        processed_img = processed_img.astype(np.float32) / 255.0
        processed_images.append(processed_img)
        labels.append(row.source_class)
 
    X = np.stack(processed_images)
    y = np.array(labels)
    y = np.where(y > 0, 1, y).astype(np.int8)
 
    return X, y

# Read train parquet file under input_path in batches, preprocess it
def process_parquet(input_path, batch_size = 500):

    print(f"Processing {input_path}...")
 
    file_paths = sorted(glob.glob(os.path.join(input_path, "*.parquet")))
    if not file_paths:
        print(f"No parquet files found in {input_path}")
        return None, None
 
    X_parts, y_parts = [], []
 
    for fp in file_paths:
        print(f"Reading {fp} in batches...")
        try:
            parquet_file = pq.ParquetFile(fp)
        except Exception as e:
            print(f"Could not open {fp} due to: {e}")
            continue
 
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["image", "source_class"]):
            df = batch.to_pandas()
            X_part, y_part = load_dataframe_to_arrays(df)
            X_parts.append(X_part)
            y_parts.append(y_part)
            del df, batch
 
    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y

def extract_hog_features_single(img, params = HOG_PARAMS, visualize = False):
    img_np = img.permute(1, 2, 0).numpy()  # CHW -> HWC, what skimage expects
    if visualize:
        features, hog_img = hog(img_np, visualize=True, **params)
        return features, hog_img
    return hog(img_np, **params)

def extract_hog_features_batch(X, y, batch_size = BATCH_SIZE):
    X_hog, y_hog = [], []
 
    print("Extracting HOG features")
    data_loader = make_loader(X, y, batch_size)
    with torch.no_grad():
        for imgs, labels in tqdm(data_loader):
            for img, label in zip(imgs, labels):
                features = extract_hog_features_single(img)
                X_hog.append(features)
                y_hog.append(label.item())
 
    return np.array(X_hog), np.array(y_hog)

def visualize_hog_features(path, n_samples = 5):
 
    df = pd.read_parquet(path)
    samples = df.sample(n_samples)
 
    fig, axes = plt.subplots(2, n_samples, figsize=(3 * n_samples, 6))
 
    for i, (_, row) in enumerate(samples.iterrows()):
        pil_img = Image.open(io.BytesIO(row["image"]))
        tensor_img = tf.to_tensor(pil_img)
        _, hog_img = extract_hog_features_single(tensor_img, visualize=True)
 
        axes[0, i].imshow(tensor_img.permute(1, 2, 0).numpy())
        axes[0, i].set_title(str(row["source_class"]))
        axes[0, i].axis("off")
 
        axes[1, i].imshow(hog_img)
        axes[1, i].set_title(f"hog_{row['source_class']}")
        axes[1, i].axis("off")
 
    plt.tight_layout()
    plt.show()

# Compute per-class weights inversely proportional to class frequency
def get_balanced_class_weights(y):
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes, weights))

# Stratified k-fold Cross validation
def k_fold_cross_validation(svm, X, y, k = K_FOLDS):

    kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=SEED)
 
    accuracies, balanced_accuracies, f1_scores = [], [], []
    fprs, recalls = [], []
 
    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"Fold {fold + 1}/{k}")
 
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
 
        svm.fit(X_train, y_train)
        y_pred = svm.predict(X_test)
 
        acc = accuracy_score(y_test, y_pred)
        bal_acc = balanced_accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
 
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
 
        accuracies.append(acc)
        balanced_accuracies.append(bal_acc)
        f1_scores.append(f1)
        fprs.append(fpr)
        recalls.append(recall)
 
        print(f"  Accuracy: {acc:.4f}  Balanced Accuracy: {bal_acc:.4f}  F1: {f1:.4f}  FPR: {fpr:.4f}  Recall: {recall:.4f}")
 
    print("K-fold cross validation completed.")
    print(f"Average Accuracy:          {np.mean(accuracies):.4f}")
    print(f"Average Balanced Accuracy: {np.mean(balanced_accuracies):.4f}")
    print(f"Average F1:                {np.mean(f1_scores):.4f}")
    print(f"Average FPR:               {np.mean(fprs):.4f}")
    print(f"Average Recall:            {np.mean(recalls):.4f}\n")
 
    return {
        "accuracy": accuracies,
        "balanced_accuracy": balanced_accuracies,
        "f1": f1_scores,
        "fpr": fprs,
        "recall": recalls,
    }

def run_c_sweep(X, y):
 
    results = {}
    for c in C_GRID:
        print(f"=== C = {c} ===")
        set_random_seeds() 
        svm = LinearSVC(C=c, max_iter=1000, class_weight="balanced")
        results[c] = k_fold_cross_validation(svm, X, y, k=K_FOLDS)
 
    return results
 
def main():
    X, y = process_parquet(TASK01_DIR)
    if X is None:
      return
    X, y = extract_hog_features_batch(X, y, batch_size=BATCH_SIZE)
    run_c_sweep(X, y)

if __name__ == "__main__":
    main()