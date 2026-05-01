# CultiKure — System Architecture

## Overview

CultiKure is an end-to-end plant disease detection system that integrates deep learning inference, GPU-accelerated serving, and LLM-based diagnosis into a single containerised application.

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User / Browser                          │
└──────────────────────────────┬──────────────────────────────────┘
                               │ HTTP (port 5000)
                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Flask Application                             │
│                    (cultikure-app container)                     │
│                                                                  │
│   ┌──────────────┐   ┌─────────────────┐   ┌────────────────┐  │
│   │  Route /     │   │  Route /submit  │   │ Route /explain │  │
│   │  /index      │   │  (POST)         │   │ (POST, JSON)   │  │
│   │  /market     │   │                 │   │                │  │
│   │  /contact    │   │ 1. Validate img │   │ 1. Parse class │  │
│   └──────────────┘   │ 2. Preprocess   │   │ 2. Call Claude │  │
│                       │ 3. Run Infer   │   │ 3. Return JSON │  │
│                       │ 4. Look up CSV │   └────────────────┘  │
│                       │ 5. Render HTML │                         │
│                       └────────┬────────┘                        │
└────────────────────────────────┼────────────────────────────────┘
                                 │ HTTP REST (port 8000)
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              NVIDIA Triton Inference Server                       │
│              (triton container, nvcr.io/nvidia/tritonserver)     │
│                                                                  │
│   POST /v2/models/cultikure_vgg/infer                           │
│                                                                  │
│   ┌──────────────────────────────────────────────────────────┐  │
│   │                  cultikure_vgg model                      │  │
│   │                                                          │  │
│   │   Backend: ONNX Runtime                                  │  │
│   │   Execution: CUDAExecutionProvider                       │  │
│   │   Input:  [batch, 3, 224, 224] FP32                     │  │
│   │   Output: [batch, 39]         FP32 (logits)             │  │
│   └──────────────────────────────────────────────────────────┘  │
│                             │                                    │
│                      NVIDIA GPU (CUDA)                           │
└─────────────────────────────────────────────────────────────────┘
                                 │
                                 │ (Anthropic API, external HTTPS)
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│              Anthropic Claude API                                │
│              Model: claude-sonnet-4-20250514                     │
│                                                                  │
│   Returns: JSON { explanation, treatment_steps,                  │
│                   preventive_measures, fertilizers,              │
│                   severity, contagious }                         │
└─────────────────────────────────────────────────────────────────┘
```

## Component Descriptions

| Component | Technology | Role |
|-----------|-----------|------|
| Frontend | HTML5, CSS3 (custom), JavaScript (Vanilla) | User interface for image upload and result display |
| Flask Backend | Python 3.10, Flask 3.1, Gunicorn | REST API server, request routing, preprocessing, response assembly |
| VGG16 Model | PyTorch → ONNX (opset 17) | Fine-tuned on PlantVillage dataset, 39-class disease classification |
| Triton Server | NVIDIA Triton 23.10, ONNX Runtime | GPU-accelerated model serving with dynamic batching |
| LLM Service | Anthropic Claude `claude-sonnet-4-20250514` | Plain-language diagnosis generation |
| Docker | Docker Engine 24+, Compose 2.x | Container orchestration |

## Data Flow

1. **Upload** — User uploads a leaf image (JPG/PNG) via the `/index` page
2. **Validation** — Flask validates file type and integrity
3. **Preprocessing** — PIL resize to 224×224, ImageNet normalization, tensor conversion
4. **Triton Inference** — Tensor sent to Triton via HTTP REST; ONNX model runs on GPU
5. **Fallback** — If Triton unavailable, local PyTorch model used
6. **Label Lookup** — Predicted class index mapped to disease name via `disease_info.csv`
7. **LLM Diagnosis** — `/explain` called (async from frontend) → Claude generates structured JSON
8. **Render** — `submit.html` displays disease info, confidence gauge, LLM panel, supplement link

## Deployment Topology

```
Host Machine (CUDA-capable)
├── Docker Network: cultikure-net
│   ├── triton:8000,8001,8002  ← Triton Inference Server
│   └── cultikure-app:5000     ← Flask Application
├── Volume: model_repository/  ← ONNX model (read-only mount)
└── Volume: uploads_data/      ← Uploaded images (persistent)
```
