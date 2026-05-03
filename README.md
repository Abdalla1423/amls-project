# AMLS 2026 – AI Image Detection

Binary classification pipeline to detect AI-generated images.

## Prerequisites

- **Docker Desktop** installed and running
- Dataset downloaded and placed in `data/` (see structure below)

## Project Structure

```
amls-project/
├── data/                          # Dataset (read-only at runtime)
│   ├── train/
│   ├── calibration/
│   ├── calibration_augmented/
│   ├── validation/
│   ├── validation_augmented/
│   └── predict/
├── solution/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── clean.py                   # Task 1.1 – Data exploration & cleaning
│   ├── prepare.py                 # Task 1.2 – Data preparation
│   ├── train.py                   # Task 1.2 – Model training
│   ├── predict.py                 # Task 1.2 – Inference → artifacts/task02/predictions.csv
│   ├── train_augmented.py         # Task 1.3 – Augmented training
│   ├── predict_augmented.py       # Task 1.3 – Inference → artifacts/task03/predictions.csv
│   └── artifacts/                 # Created at runtime (models, predictions)
│       ├── task02/predictions.csv
│       └── task03/predictions.csv
└── README.md
```

## How to Run (Docker Desktop)

All commands are run from the **project root** (`amls-project/`).

### 1. Build the Docker image

```bash
docker build -t amls ./solution
```

This installs Python 3.11, all pip dependencies, and CPU-only PyTorch 2.5.1.

### 2. Run the full pipeline

```bash
docker run --cpus 8 \
  -v "$(pwd)/data:/workspace/solution/data:ro" \
  -v "$(pwd)/solution/artifacts:/workspace/solution/artifacts" \
  amls bash -c "\
    python clean.py --timeout_seconds 600 && \
    python prepare.py --timeout_seconds 600 && \
    python train.py --timeout_seconds 1800 && \
    python predict.py --timeout_seconds 600 && \
    python train_augmented.py --timeout_seconds 1800 && \
    python predict_augmented.py --timeout_seconds 600"
```

**What this does:**
- `--cpus 8` — limits the container to 8 CPU cores (matches grading environment)
- `-v .../data:ro` — mounts the dataset as **read-only** inside the container
- `-v .../artifacts` — mounts the artifacts folder so results persist after the container stops

### 3. Run individual scripts

To run a single step (e.g. just cleaning):

```bash
docker run --cpus 8 \
  -v "$(pwd)/data:/workspace/solution/data:ro" \
  -v "$(pwd)/solution/artifacts:/workspace/solution/artifacts" \
  amls python clean.py --timeout_seconds 600
```

### 4. Check results

After the pipeline finishes, predictions are in:
- `solution/artifacts/task02/predictions.csv` (Task 2)
- `solution/artifacts/task03/predictions.csv` (Task 3)

## Local Development (without Docker)

```bash
cd solution
python -m venv ../.venv
source ../.venv/bin/activate
pip install -r requirements.txt
pip install --index-url https://download.pytorch.org/whl/cpu torch==2.5.1
```

Then run scripts directly (make sure `data/` is symlinked or copied into `solution/`):

```bash
ln -s ../data data
python clean.py --timeout_seconds 600
python prepare.py --timeout_seconds 600
python train.py --timeout_seconds 1800
python predict.py --timeout_seconds 600
python train_augmented.py --timeout_seconds 1800
python predict_augmented.py --timeout_seconds 600
```

## Script Execution Order

| # | Script | Timeout | Purpose |
|---|--------|---------|---------|
| 1 | `clean.py` | 600s | Explore and clean training data |
| 2 | `prepare.py` | 600s | Prepare data for training (not predict/) |
| 3 | `train.py` | 1800s | Train model, calibrate threshold (FPR ≤ 20%) |
| 4 | `predict.py` | 600s | Run inference → `artifacts/task02/predictions.csv` |
| 5 | `train_augmented.py` | 1800s | Train robust model with augmentation |
| 6 | `predict_augmented.py` | 600s | Run inference → `artifacts/task03/predictions.csv` |

## Notes

- **No internet** is available inside the container at runtime — all dependencies must be in the image.
- All output must go to `artifacts/`, never to `data/` (which is read-only).
- Each script receives `--timeout_seconds` and will be killed if it exceeds the limit.
- The Docker image size should stay under 4 GB.
