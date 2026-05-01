"""
CultiKure — VGG16 Fine-Tuning Script
======================================
Fine-tunes torchvision VGG16 on the PlantVillage dataset.

Usage:
    python train.py --data_dir data/PlantVillage --epochs 25 --batch_size 32

Requirements:
    pip install torch torchvision tqdm scikit-learn matplotlib

Dataset layout expected:
    data/PlantVillage/
        Apple___Apple_scab/
            img1.jpg ...
        Apple___Black_rot/
            ...

After training:
    - Best checkpoint → checkpoints/best_model.pth
    - Run export_onnx.py to produce model.onnx for Triton
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES   = 39
IMG_SIZE      = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# PlantVillage class → index mapping (same as CNN.py)
IDX_TO_CLASS = {
    0: 'Apple___Apple_scab', 1: 'Apple___Black_rot', 2: 'Apple___Cedar_apple_rust',
    3: 'Apple___healthy', 4: 'Background_without_leaves', 5: 'Blueberry___healthy',
    6: 'Cherry___Powdery_mildew', 7: 'Cherry___healthy',
    8: 'Corn___Cercospora_leaf_spot Gray_leaf_spot', 9: 'Corn___Common_rust',
    10: 'Corn___Northern_Leaf_Blight', 11: 'Corn___healthy', 12: 'Grape___Black_rot',
    13: 'Grape___Esca_(Black_Measles)', 14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    15: 'Grape___healthy', 16: 'Orange___Haunglongbing_(Citrus_greening)',
    17: 'Peach___Bacterial_spot', 18: 'Peach___healthy',
    19: 'Pepper,_bell___Bacterial_spot', 20: 'Pepper,_bell___healthy',
    21: 'Potato___Early_blight', 22: 'Potato___Late_blight', 23: 'Potato___healthy',
    24: 'Raspberry___healthy', 25: 'Soybean___healthy', 26: 'Squash___Powdery_mildew',
    27: 'Strawberry___Leaf_scorch', 28: 'Strawberry___healthy',
    29: 'Tomato___Bacterial_spot', 30: 'Tomato___Early_blight',
    31: 'Tomato___Late_blight', 32: 'Tomato___Leaf_Mold',
    33: 'Tomato___Septoria_leaf_spot',
    34: 'Tomato___Spider_mites Two-spotted_spider_mite',
    35: 'Tomato___Target_Spot', 36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    37: 'Tomato___Tomato_mosaic_virus', 38: 'Tomato___healthy',
}


# ---------------------------------------------------------------------------
# Data transforms
# ---------------------------------------------------------------------------
def get_transforms(augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------
def build_vgg16(num_classes: int, freeze_features: bool = True) -> nn.Module:
    """
    Load pretrained VGG16 and replace the classifier head.
    Optionally freeze the feature extractor for the first epochs.
    """
    model = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    # Replace classifier
    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, num_classes),
    )
    return model


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------
def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in tqdm(loader, desc="  Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    all_preds, all_labels = [], []

    for images, labels in tqdm(loader, desc="  Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += images.size(0)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    return total_loss / total, correct / total, f1, all_preds, all_labels


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_training_curves(history: dict, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("CultiKure VGG16 Training", fontweight="bold")

    axes[0].plot(history["train_loss"], label="Train", color="#16a34a")
    axes[0].plot(history["val_loss"],   label="Val",   color="#f97316")
    axes[0].set_title("Loss"); axes[0].legend(); axes[0].grid(alpha=.3)

    axes[1].plot(history["train_acc"], label="Train", color="#16a34a")
    axes[1].plot(history["val_acc"],   label="Val",   color="#f97316")
    axes[1].set_title("Accuracy"); axes[1].legend(); axes[1].grid(alpha=.3)

    axes[2].plot(history["val_f1"], label="Val F1", color="#60a5fa")
    axes[2].set_title("Weighted F1"); axes[2].legend(); axes[2].grid(alpha=.3)

    plt.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Training curves → {out_dir / 'training_curves.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    use_amp = device.type == "cuda"

    # Directories
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    # Dataset
    print(f"\nLoading dataset from: {args.data_dir}")
    full_dataset = datasets.ImageFolder(args.data_dir, transform=get_transforms(augment=True))
    print(f"  Total samples: {len(full_dataset)} | Classes: {len(full_dataset.classes)}")

    val_size  = int(len(full_dataset) * 0.2)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size],
                                     generator=torch.Generator().manual_seed(42))
    val_ds.dataset.transform = get_transforms(augment=False)  # No augment for val

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

    # Model
    model = build_vgg16(NUM_CLASSES, freeze_features=True).to(device)
    print(f"  Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()),
                     lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    history = {k: [] for k in ["train_loss", "val_loss", "train_acc", "val_acc", "val_f1"]}
    best_f1 = 0.0
    unfreeze_done = False

    for epoch in range(1, args.epochs + 1):
        # Unfreeze feature extractor halfway through
        if not unfreeze_done and epoch == args.epochs // 2:
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer.add_param_group({"params": model.features.parameters(), "lr": args.lr * 0.1})
            unfreeze_done = True
            print(f"  [Epoch {epoch}] Feature extractor unfrozen (LR=1e-5)")

        t0 = time.time()
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        vl_loss, vl_acc, vl_f1, preds, labels = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        history["val_f1"].append(vl_f1)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Loss {tr_loss:.4f}/{vl_loss:.4f} | "
            f"Acc {tr_acc:.4f}/{vl_acc:.4f} | "
            f"F1 {vl_f1:.4f} | {elapsed:.0f}s"
        )

        if vl_f1 > best_f1:
            best_f1 = vl_f1
            ckpt_path = ckpt_dir / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best F1={best_f1:.4f} → saved to {ckpt_path}")

    # Final report
    print("\n── Final Evaluation ──────────────────────────────")
    _, _, _, preds, labels = eval_epoch(model, val_loader, criterion, device)
    class_names = [IDX_TO_CLASS.get(i, str(i)) for i in range(NUM_CLASSES)]
    report = classification_report(labels, preds, target_names=class_names, zero_division=0)
    print(report)

    report_path = ckpt_dir / "classification_report.txt"
    report_path.write_text(report)

    history_path = ckpt_dir / "training_history.json"
    history_path.write_text(json.dumps(history, indent=2))

    save_training_curves(history, ckpt_dir)
    print(f"\n✓ Training complete. Best val F1: {best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_dir / 'best_model.pth'}")
    print(f"  Next step:  python export_onnx.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VGG16 on PlantVillage")
    parser.add_argument("--data_dir",    default="data/PlantVillage", help="Dataset root")
    parser.add_argument("--epochs",      type=int,   default=25)
    parser.add_argument("--batch_size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--workers",     type=int,   default=4)
    args = parser.parse_args()
    main(args)
