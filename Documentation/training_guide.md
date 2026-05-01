# CultiKure — Training Guide

## Dataset: PlantVillage

| Property | Value |
|----------|-------|
| Source | [Kaggle — emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease) |
| Total Images | ~54,306 |
| Classes | 39 (disease and healthy) |
| Crops Covered | 14 (Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato) |
| Image Size | Variable (resized to 224×224) |
| Format | JPEG/PNG |

### Class Distribution (39 Classes)

| Index | Class | Approx. Samples |
|-------|-------|----------------|
| 0 | Apple___Apple_scab | 630 |
| 1 | Apple___Black_rot | 621 |
| 2 | Apple___Cedar_apple_rust | 275 |
| 3 | Apple___healthy | 1645 |
| 4 | Background_without_leaves | 1143 |
| ... | ... | ... |
| 38 | Tomato___healthy | 1591 |

---

## Dataset Setup

### Via Kaggle CLI

```bash
pip install kaggle
kaggle datasets download -d emmarex/plantdisease -p training/data/ --unzip
mv training/data/PlantVillage training/data/PlantVillage
```

### Directory Structure Expected

```
training/data/PlantVillage/
├── Apple___Apple_scab/
│   ├── img001.JPG
│   └── ...
├── Apple___Black_rot/
│   └── ...
└── Tomato___healthy/
    └── ...
```

---

## Model Architecture: VGG16

```
VGG16 (pretrained on ImageNet)
├── features (13 conv layers + 5 max pools) — frozen initially
│   ├── Conv2d(3, 64) + ReLU × 2 + MaxPool
│   ├── Conv2d(64, 128) + ReLU × 2 + MaxPool
│   ├── Conv2d(128, 256) + ReLU × 3 + MaxPool
│   ├── Conv2d(256, 512) + ReLU × 3 + MaxPool
│   └── Conv2d(512, 512) + ReLU × 3 + MaxPool
├── avgpool (AdaptiveAvgPool2d → 7×7)
└── classifier (replaced for PlantVillage)
    ├── Linear(25088, 4096) + ReLU + Dropout(0.5)
    ├── Linear(4096, 4096) + ReLU + Dropout(0.5)
    └── Linear(4096, 39)
```

**Total parameters:** ~138M  
**Trainable (initial):** ~119M (classifier only)  
**After unfreeze:** all 138M

---

## Training Configuration

| Hyperparameter | Value | Notes |
|----------------|-------|-------|
| Input size | 224×224 | Standard VGG input |
| Batch size | 32 | Adjust based on GPU memory |
| Learning rate | 1e-4 | For classifier; 1e-5 for features after unfreeze |
| Optimizer | Adam | weight_decay=1e-4 |
| LR Scheduler | CosineAnnealingLR | T_max=25, eta_min=1e-6 |
| Epochs | 25 | Unfreeze features at epoch 12 |
| Loss | CrossEntropyLoss | label_smoothing=0.1 |
| Mixed Precision | AMP (fp16) | On CUDA only |
| Train/Val Split | 80/20 | Random seed=42 |

---

## Data Augmentation

Applied to training set only:

| Transform | Parameters |
|-----------|-----------|
| RandomResizedCrop | 224×224, scale=(0.7, 1.0) |
| RandomHorizontalFlip | p=0.5 |
| RandomVerticalFlip | p=0.5 |
| RandomRotation | ±15° |
| ColorJitter | brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05 |
| ToTensor | Convert PIL → float32 tensor |
| Normalize | mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225] |

Validation set: only Resize + ToTensor + Normalize (no augmentation).

---

## Training Commands

```bash
cd training

# Standard training
python train.py --data_dir data/PlantVillage --epochs 25 --batch_size 32

# Quick test run
python train.py --data_dir data/PlantVillage --epochs 3 --batch_size 16

# Export to ONNX after training
python export_onnx.py \
    --checkpoint checkpoints/best_model.pth \
    --output ../model_repository/cultikure_vgg/1/model.onnx
```

---

## Expected Performance

Based on PlantVillage benchmarks:

| Metric | Expected Value |
|--------|---------------|
| Validation Accuracy | 95–98% |
| Weighted F1 | 0.95–0.97 |
| Training time (GPU) | ~2–4 hours (25 epochs, batch=32) |
| Inference latency (Triton GPU) | <50ms |

---

## Output Files

After training:

```
training/checkpoints/
├── best_model.pth             # Best validation F1 checkpoint
├── classification_report.txt  # Per-class precision/recall/F1
├── training_history.json      # Loss/accuracy per epoch
└── training_curves.png        # Loss/accuracy/F1 plots
```
