# CultiKure — Deployment Guide

## Prerequisites

| Requirement | Version |
|-------------|---------|
| Docker Engine | 24.0+ |
| Docker Compose | v2.x |
| NVIDIA GPU | Any CUDA-capable card |
| NVIDIA Driver | 525+ |
| NVIDIA Container Toolkit | Latest |
| Anthropic API Key | Required for `/explain` |

---

## Quick Start (Docker Compose)

### 1. Clone the Repository

```bash
git clone https://github.com/Manu-krishnan005/CultiKure-Disease-Prediction.git
cd CultiKure-Disease-Prediction
```

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and set:
#   ANTHROPIC_API_KEY=sk-ant-...
#   USE_TRITON=true
```

### 3. Place the ONNX Model

The ONNX model must be present in the Triton model repository:

```
model_repository/
└── cultikure_vgg/
    ├── config.pbtxt
    └── 1/
        └── model.onnx  ← Place your exported model here
```

**Option A — Use the training pipeline:**
```bash
cd training
python train.py --data_dir data/PlantVillage --epochs 25
python export_onnx.py --checkpoint checkpoints/best_model.pth \
                      --output ../model_repository/cultikure_vgg/1/model.onnx
```

**Option B — Convert existing checkpoint:**
```bash
cd training
python export_onnx.py --checkpoint ../App/trained_model.pth \
                      --output ../model_repository/cultikure_vgg/1/model.onnx
```

> **Note:** The existing `trained_model.pth` uses the `CNN` architecture (not VGG16). Use Option A for a proper VGG16 model.

### 4. Pull the Pre-built Image (optional)

```bash
docker pull Manu-krishnan005/cultikure:latest
```

### 5. Launch the Stack

```bash
docker-compose up --build
```

> For GPU passthrough, ensure NVIDIA Container Toolkit is installed and `--gpus all` support is available via the `deploy.resources` in `docker-compose.yml`.

**Access points:**
- Flask app: http://localhost:5000
- Triton REST API: http://localhost:8000
- Triton metrics: http://localhost:8002/metrics

### 6. Test the Deployment

```bash
# Health check
curl http://localhost:8000/v2/health/ready
curl http://localhost:5000/

# Test inference
curl -X POST http://localhost:5000/explain \
  -H "Content-Type: application/json" \
  -d '{"disease": "Tomato___Early_blight"}'
```

---

## Local Development (without Docker)

```bash
cd App
pip install -r requirements.txt

# Set env vars
set ANTHROPIC_API_KEY=your_key
set USE_TRITON=false
set FLASK_ENV=development

python app.py
# Visit http://localhost:5000
```

---

## GPU Requirements

| Component | GPU Memory Required |
|-----------|-------------------|
| Triton + VGG16 ONNX | ~600 MB |
| Training (batch=32) | ~4 GB |
| Training (batch=64) | ~8 GB |

---

## Docker Hub

The pre-built Flask image is available at:

```
docker pull Manu-krishnan005/cultikure:latest
```

**Image details:**
- Base: `python:3.10-slim`
- Size: ~1.2 GB
- Architecture: linux/amd64

---

## Saving and Archiving the Docker Image

```bash
# Save to tar
docker save Manu-krishnan005/cultikure:latest | gzip > cultikure_latest.tar.gz

# Compress as .rar (if desired)
# rar a cultikure.rar cultikure_latest.tar.gz

# Load on another machine
docker load < cultikure_latest.tar.gz
```

---

## nvidia-smi Verification

During inference, GPU utilization should be visible:

```bash
watch -n 1 nvidia-smi
```

Expected output during active inference:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI ...  Driver Version: ...  CUDA Version: 12.x                    |
|-------------------------------+----------------------+----------------------|
| GPU  Name         ... | Volatile Uncorr. ECC   |        Memory-Usage |
|===============================+======================+======================|
|   0  NVIDIA ...       |       0%       |     614MiB /  VRAM MiB |
+-----------------------------------------------------------------------------+
| Processes:                                                                  |
|  GPU   PID   Type   Process name                             GPU Memory    |
|   0  ...    C    tritonserver                               600MiB         |
+-----------------------------------------------------------------------------+
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `Triton server not ready` | Wait 30s after `docker-compose up`, Triton loads slowly |
| `Model 'cultikure_vgg' not ready` | Check `model_repository/cultikure_vgg/1/model.onnx` exists |
| `ANTHROPIC_API_KEY not set` | Set key in `.env`, `docker-compose down && up` |
| `CUDA out of memory` | Reduce `max_batch_size` in `config.pbtxt` |
| `Port 5000 already in use` | Change port mapping in `docker-compose.yml` |
