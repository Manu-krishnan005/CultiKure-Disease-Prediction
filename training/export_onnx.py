"""
Export trained ResNet50/VGG16 checkpoint to ONNX format for NVIDIA Triton.

Usage:
    # Export existing ResNet50 checkpoint (trained_model.pth):
    python export_onnx.py --arch resnet50 \
                          --checkpoint ../App/trained_model.pth \
                          --output ../model_repository/cultikure_vgg/1/model.onnx

    # Export newly trained VGG16 checkpoint:
    python export_onnx.py --arch vgg16 \
                          --checkpoint checkpoints/best_model.pth \
                          --output ../model_repository/cultikure_vgg/1/model.onnx

Verifies the ONNX model with onnxruntime before finishing.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torchvision import models


NUM_CLASSES = 39
IMG_SIZE    = 224


def build_model(arch: str, num_classes: int) -> nn.Module:
    if arch == "resnet50":
        m = models.resnet50(weights=None)
        m.fc = nn.Linear(2048, num_classes)
    elif arch == "vgg16":
        m = models.vgg16(weights=None)
        m.classifier = nn.Sequential(
            nn.Linear(25088, 4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096,  4096), nn.ReLU(inplace=True), nn.Dropout(0.5),
            nn.Linear(4096,  num_classes),
        )
    else:
        raise ValueError(f"Unsupported architecture: {arch}. Use 'resnet50' or 'vgg16'")
    return m


def export(checkpoint: Path, output: Path, arch: str, opset: int = 17):
    device = torch.device("cpu")  # Export on CPU for portability

    print(f"Architecture: {arch}")
    print(f"Loading checkpoint: {checkpoint}")
    model = build_model(arch, NUM_CLASSES)
    state_dict = torch.load(str(checkpoint), map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)

    # Dummy input — batch=1, C=3, H=W=224
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    print(f"Exporting to ONNX (opset {opset}): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        str(output),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch_size"},
            "output": {0: "batch_size"},
        },
        opset_version=opset,
        do_constant_folding=True,
        verbose=False,
    )
    print(f"  [OK] ONNX model written ({output.stat().st_size / 1e6:.1f} MB)")

    # Verify with ONNX Runtime
    print("Verifying with onnxruntime...")
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
        in_name  = sess.get_inputs()[0].name
        out_name = sess.get_outputs()[0].name

        dummy_np = dummy.numpy().astype(np.float32)
        result   = sess.run([out_name], {in_name: dummy_np})[0]

        assert result.shape == (1, NUM_CLASSES), f"Unexpected shape: {result.shape}"
        print(f"  [OK] ONNX Runtime verification passed. Output shape: {result.shape}")
    except ImportError:
        print("  [WARN] onnxruntime not installed — skipping runtime verification")
        print("    Run: pip install onnxruntime-gpu")

    print(f"\n  [OK] Export complete.")
    print(f"  Triton model repo: {output.parent.parent}/")
    print(f"    ├── config.pbtxt")
    print(f"    └── 1/model.onnx  ({output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export PyTorch model → ONNX for Triton")
    parser.add_argument("--arch",       default="resnet50", choices=["resnet50", "vgg16"])
    parser.add_argument("--checkpoint", default="../App/trained_model.pth",
                        help="Path to trained .pth checkpoint")
    parser.add_argument("--output",     default="../model_repository/cultikure_vgg/1/model.onnx",
                        help="Output ONNX path")
    parser.add_argument("--opset",      type=int, default=17, help="ONNX opset version")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    out  = Path(args.output)

    if not ckpt.exists():
        print(f"ERROR: Checkpoint not found: {ckpt}")
        sys.exit(1)

    export(ckpt, out, args.arch, args.opset)
