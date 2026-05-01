# CultiKure — Triton Inference Server Configuration

## Model Repository Layout

```
model_repository/
└── cultikure_vgg/
    ├── config.pbtxt        ← Model configuration (this file)
    └── 1/
        └── model.onnx      ← ONNX model file
```

The top-level directory name (`cultikure_vgg`) is the **model name** used in API calls.  
The `1/` subdirectory is the **model version** — Triton serves the latest version by default.

---

## config.pbtxt Explained

```protobuf
name: "cultikure_vgg"         # Model name (must match directory name)
backend: "onnxruntime"         # Use ONNX Runtime backend
max_batch_size: 8              # Maximum batch size for dynamic batching

input [
  {
    name: "input"              # Must match ONNX model's input node name
    data_type: TYPE_FP32       # float32 — same as PyTorch default
    dims: [ 3, 224, 224 ]      # Excluding batch dimension (handled by max_batch_size)
  }
]

output [
  {
    name: "output"             # Must match ONNX model's output node name
    data_type: TYPE_FP32       # Raw logits (softmax applied in Flask)
    dims: [ 39 ]               # 39 PlantVillage classes
  }
]

# GPU execution configuration
instance_group [
  {
    count: 1                   # One model instance
    kind: KIND_GPU             # Run on GPU
    gpus: [ 0 ]                # GPU device index
  }
]

# ONNX Runtime execution accelerator
optimization {
  execution_accelerators {
    gpu_execution_accelerator [
      {
        name: "cuda"
        parameters {
          key: "cudnn_conv_algo_search"
          value: "DEFAULT"
        }
      }
    ]
  }
  graph { level: 1 }           # ONNX graph optimization level (0=none, 1=basic, 2=extended)
}

# Dynamic batching configuration
dynamic_batching {
  preferred_batch_size: [ 1, 4, 8 ]     # Optimal batch sizes
  max_queue_delay_microseconds: 100      # Max wait before sending incomplete batch
}
```

---

## How to Launch Triton

### With Docker (production)

```bash
docker run --gpus all -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  nvcr.io/nvidia/tritonserver:23.10-py3 \
  tritonserver --model-repository=/models
```

### With Docker Compose

```bash
docker-compose up triton
```

---

## Verifying Triton is Ready

```bash
# Server health
curl -s http://localhost:8000/v2/health/ready

# Model health
curl -s http://localhost:8000/v2/models/cultikure_vgg/ready

# Model metadata
curl -s http://localhost:8000/v2/models/cultikure_vgg | python -m json.tool
```

---

## Testing Inference via Triton

```python
import numpy as np
import tritonclient.http as httpclient

client = httpclient.InferenceServerClient("localhost:8000")

# Create dummy input (batch=1, 3 channels, 224x224)
img = np.random.randn(1, 3, 224, 224).astype(np.float32)

inp = httpclient.InferInput("input", img.shape, "FP32")
inp.set_data_from_numpy(img)

out = httpclient.InferRequestedOutput("output")
resp = client.infer("cultikure_vgg", [inp], outputs=[out])

logits = resp.as_numpy("output")
print(f"Output shape: {logits.shape}")    # (1, 39)
print(f"Predicted class: {np.argmax(logits)}")
```

---

## Prometheus Metrics (port 8002)

Triton exposes metrics at `http://localhost:8002/metrics`:

| Metric | Description |
|--------|-------------|
| `nv_inference_request_success` | Total successful inference requests |
| `nv_inference_request_failure` | Total failed requests |
| `nv_inference_exec_count` | Total model executions |
| `nv_inference_request_duration_us` | Request latency histogram (µs) |
| `nv_gpu_utilization` | GPU utilization (%) |
| `nv_gpu_memory_used_bytes` | GPU memory used |

---

## Troubleshooting

| Issue | Check |
|-------|-------|
| Model not ready | Ensure `model.onnx` exists in `cultikure_vgg/1/` |
| Shape mismatch | ONNX input name must be `"input"`, output name `"output"` |
| GPU not used | Check `nvidia-smi`, ensure `KIND_GPU` in config |
| OOM error | Reduce `max_batch_size` in `config.pbtxt` |
