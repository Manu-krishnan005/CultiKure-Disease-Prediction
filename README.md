# CultiKure — Plant Disease Detection System

[![Docker](https://img.shields.io/badge/Docker-Hub-blue)](https://hub.docker.com/r/Manu-krishnan005/cultikure)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![Model](https://img.shields.io/badge/Model-VGG16%20ONNX-orange)](Documentation/training_guide.md)

**CultiKure** is an end-to-end, production-grade plant disease detection system combining:
- 🔬 **VGG16 deep learning model** fine-tuned on PlantVillage (39 classes)
- ⚡ **NVIDIA Triton Inference Server** for GPU-accelerated ONNX serving
- 🤖 **Anthropic Claude AI** (`claude-sonnet-4-20250514`) for LLM-powered diagnosis
- 🐳 **Docker Compose** for one-command deployment

---

## Quick Start

```bash
git clone https://github.com/Manu-krishnan005/CultiKure-Disease-Prediction.git
cd CultiKure-Disease-Prediction

# Configure
cp .env.example .env
# Set ANTHROPIC_API_KEY in .env

# Place model (see Documentation/training_guide.md)
# model_repository/cultikure_vgg/1/model.onnx

# Deploy
docker-compose up --build
```

Open http://localhost:5000

---

## Architecture

```
Browser → Flask (port 5000) → Triton Server (port 8000) → VGG16 ONNX (GPU)
                           ↘ Anthropic Claude API (LLM diagnosis)
```

---

## Features

| Feature | Technology |
|---------|-----------|
| Disease detection (39 classes) | VGG16, PlantVillage dataset |
| GPU inference | NVIDIA Triton, ONNX Runtime, CUDA |
| LLM diagnosis | Anthropic Claude `claude-sonnet-4-20250514` |
| Confidence score | Softmax probability gauge |
| Supplement recommendations | Curated CSV database |
| Containerised deployment | Docker Compose, GPU passthrough |

---

## Supported Crops & Diseases

14 crop types — 39 classes including:
- Apple (scab, black rot, cedar rust, healthy)
- Tomato (bacterial spot, early blight, late blight, leaf mold, mosaic virus, yellow curl virus, healthy, ...)
- Corn, Grape, Potato, Pepper, Strawberry, Peach, Cherry, and more

---

## Documentation

| Document | Description |
|----------|-------------|
| [System Architecture](Documentation/system_architecture.md) | Component diagram and data flow |
| [API Reference](Documentation/api_reference.md) | All Flask endpoints with schemas |
| [Training Guide](Documentation/training_guide.md) | Dataset, preprocessing, training config |
| [Triton Configuration](Documentation/triton_configuration.md) | Model repo, config.pbtxt, GPU setup |
| [Deployment Guide](Documentation/deployment_guide.md) | Docker Compose, GPU requirements |

---

## Local Development

```bash
cd App
pip install -r requirements.txt
set USE_TRITON=false
set ANTHROPIC_API_KEY=your_key_here
set FLASK_ENV=development
python app.py
```

---

## Docker Hub

```bash
docker pull Manu-krishnan005/cultikure:latest
```

---

## Project Structure

```
CultiKure/
├── App/                   # Flask application
│   ├── app.py             # Main Flask app (fixed + extended)
│   ├── CNN.py             # CNN model class
│   ├── llm_service.py     # Anthropic Claude integration
│   ├── triton_client.py   # Triton HTTP client
│   ├── requirements.txt   # Pinned dependencies
│   ├── templates/         # Jinja2 HTML templates
│   └── static/            # CSS, JS, uploads
├── training/              # Training scripts
│   ├── train.py           # VGG16 fine-tuning
│   └── export_onnx.py     # ONNX export for Triton
├── model_repository/      # Triton model repository
│   └── cultikure_vgg/
│       ├── config.pbtxt   # Triton model config
│       └── 1/model.onnx   # ONNX model (place here)
├── tests/                 # API test suite
│   └── test_api.py        # pytest tests
├── Documentation/         # Technical documentation
├── Dockerfile             # Flask app container
├── docker-compose.yml     # Full stack orchestration
└── .env.example           # Environment template
```

---

## License

MIT License — see LICENSE file.
