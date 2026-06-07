# AMLS Project Context (for Claude Code)

## Project: AI Image Detection Binary Classification
- Path: /Users/I747530/Documents/AMLS_project/amls-project/solution
- Constraint: User rewrites code by hand (no AI-generated code allowed per project rules)
- Exercise PDF: /Users/I747530/Documents/AMLS_project/amls-project/AMLS_2026_Exercise.pdf
- Venv: /Users/I747530/Documents/AMLS_project/amls-project/.venv

## Task 2: CNN Model (COMPLETE)
- 4 conv blocks, K=32, 64×64 images, BICUBIC resize
- Block pattern: Conv→BN→ReLU→MaxPool (blocks 1-3), Conv→BN→ReLU→AdaptiveAvgPool(1) (block 4)
- Classifier: Flatten→Dropout(0.3)→Linear(128,2)
- AdamW, LR=5e-4, weight_decay=1e-3, CosineAnnealingLR T_max=20
- CrossEntropyLoss, label_smoothing=0.1, class_weights=[5.0, 1.0]
- Calibration margin: 0.85 * max_fpr
- Best: recall_ai=0.8451, FPR=0.1862, threshold=0.70
- SEED=42, deterministic

## Task 3: Augmented Model (COMPLETE)
- Trained from scratch with Dropout(0.5), heavy on-the-fly augmentation
- Best: val_aug recall≈0.71, FPR≈0.19
- Key insight: val_augmented contains DIFFERENT images (domain generalization)

## Task 4: Explainability (IN PROGRESS)

### What's Done
- GradCAM script: /Users/I747530/Documents/AMLS_project/amls-project/explainability.py
- Analysis script: /Users/I747530/Documents/AMLS_project/amls-project/explainability_analysis.py
- Output: /Users/I747530/Documents/AMLS_project/amls-project/explainability_output/
- 5 figures generated: gradcam_grid.png, real_vs_ai_attention.png, confidence_distribution.png, failure_cases.png, average_heatmaps.png
- GradCAM targets layer index 14 (ReLU after last conv, block 4)

### Quantitative Results
- threshold=0.70, recall_ai=0.8408, FPR=0.1862
- TP=787, TN=153, FP=35, FN=149
- TP confidence: mean=0.8381, std=0.0557
- TN confidence: mean=0.3723, std=0.2013
- FP confidence: mean=0.7718, std=0.0428 (clustered just above threshold)
- FN confidence: mean=0.5026, std=0.1749 (wide spread)
- Overlap zone (0.4-0.8): 101 real + 310 AI images
- GradCAM intensity: AI=0.3620 vs Real=0.2054 (AI activates stronger)
- Center/periphery ratio: AI=1.09 (center-biased), Real=0.95 (periphery-biased)
- Correlation between avg real vs AI heatmaps: 0.7061
- FP heatmap intensity=0.3217 (closer to AI than real)
- FN heatmap intensity=0.2496 (closer to real than AI)
- Both classes show strongest attention in bottom-right quadrant

### What's Needed
- Occlusion/perturbation analysis (exercise lists it as a direction)
- Critical interpretation of all figures for the report
- The exercise says: "Explanations should not be treated as automatically correct; rather, you should discuss whether the explanation appears plausible and whether it reveals remaining shortcut behavior or dataset bias."

### Model & Data Locations
- Model: solution/artifacts/best_model.pt (Task 2)
- Validation data: solution/artifacts/validation.npz (X, y arrays, 1124 samples)
- Model architecture: build_cnn(k=32) with nn.Sequential, Dropout(0.3)

### Files Status
- clean.py/clean2.py: COMPLETE
- prepare.py/prepare2.py: COMPLETE
- train.py/train2.py: COMPLETE (train2.py is user's rewrite)
- predict.py/predict2.py: COMPLETE
- train_augmented.py: COMPLETE (needs user rewrite)
- predict_augmented.py: COMPLETE (needs user rewrite)
- baseline_model.py: COMPLETE (LR recall=0.35, RF recall=0.56)

### Design Decisions
- SIGALRM removed from ALL scripts (grader kills externally)
- --timeout_seconds arg kept (grader passes it)
- train scripts use deadline approach instead
- TASK02_DIR removed from train.py (predict.py handles its own output dir)
