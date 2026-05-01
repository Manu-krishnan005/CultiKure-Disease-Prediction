# CultiKure — API Reference

All endpoints served by the Flask application on port **5000**.

---

## Page Routes (HTML)

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Home page |
| GET | `/index` | AI Engine — image upload page |
| GET | `/market` | Supplement marketplace |
| GET | `/contact` | Contact page |
| GET | `/mobile-device` | Mobile detection redirect page |

---

## POST /submit

Image upload and inference endpoint.

**Request** — `multipart/form-data`

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image` | File | Yes | Leaf image (JPG, PNG, WebP, BMP) — max 16 MB |

**Response** — HTML page (`submit.html`) rendered with:

| Template Variable | Type | Description |
|-------------------|------|-------------|
| `title` | string | Disease name from `disease_info.csv` |
| `desc` | string | Disease description |
| `prevent` | string | Prevention/treatment steps |
| `image_url` | string | Reference disease image URL |
| `pred` | int | Predicted class index (0–38) |
| `sname` | string | Supplement product name |
| `simage` | string | Supplement image URL |
| `buy_link` | string | Purchase URL |
| `confidence` | float | Softmax confidence (0–100, 1 decimal) |
| `backend` | string | `"triton"` or `"local"` |
| `disease_class` | string | PlantVillage class label |

**Error Responses**

| Status | Condition |
|--------|-----------|
| 400 | No file, unsupported type, or invalid image |
| 413 | File exceeds 16 MB |
| 500 | Inference or label lookup failure |

---

## POST /explain

LLM-powered disease explanation endpoint.

**Request** — `application/json`

```json
{
  "disease": "Tomato___Early_blight"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `disease` | string | Yes | PlantVillage class label (e.g. `Tomato___Early_blight`) |

**Response** — `200 OK`, `application/json`

```json
{
  "disease": "Tomato___Early_blight",
  "readable_name": "Tomato — Early blight",
  "explanation": "Early blight is a fungal disease caused by Alternaria solani...",
  "treatment_steps": [
    "Remove and destroy infected leaves immediately.",
    "Apply copper-based fungicide every 7–10 days.",
    "Ensure adequate plant spacing for airflow."
  ],
  "preventive_measures": [
    "Rotate crops annually — avoid planting tomatoes in the same bed.",
    "Water at the base of the plant, not overhead.",
    "Mulch around plant base to prevent soil splash."
  ],
  "fertilizers": [
    "Potassium-rich fertilizer (e.g. K2SO4) to strengthen cell walls.",
    "Calcium nitrate to improve resistance to fungal infection."
  ],
  "severity": "medium",
  "contagious": true,
  "generated_at": "2026-04-30T14:00:00Z"
}
```

**Error Responses**

| Status | Condition | Body |
|--------|-----------|------|
| 400 | Missing `disease` key or empty string | `{ "error": "..." }` |
| 503 | Anthropic API key not set | `{ "error": "ANTHROPIC_API_KEY not set" }` |
| 500 | Internal server error | `{ "error": "Internal server error." }` |

**Notes**
- Results are cached per-process (LRU cache, 64 entries)
- Healthy plant classes return immediately without calling Claude
- Uses model `claude-sonnet-4-20250514` with `max_tokens=1024`

---

## Triton Inference Server (Internal)

The Flask app communicates with Triton at `${TRITON_URL}` (default: `triton:8000`).

### Health Check
```
GET http://triton:8000/v2/health/ready
```

### Model Inference
```
POST http://triton:8000/v2/models/cultikure_vgg/infer
Content-Type: application/json
```

**Body:**
```json
{
  "inputs": [
    {
      "name": "input",
      "shape": [1, 3, 224, 224],
      "datatype": "FP32",
      "data": [...]
    }
  ]
}
```

**Response:**
```json
{
  "outputs": [
    {
      "name": "output",
      "shape": [1, 39],
      "datatype": "FP32",
      "data": [0.01, 0.03, ...]
    }
  ]
}
```

Softmax is applied in the Flask `triton_client.py` to convert logits to probabilities.
