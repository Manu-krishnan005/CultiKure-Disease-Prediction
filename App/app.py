"""
CultiKure — Fixed & Extended Flask Application
================================================
Key fixes over original:
  - Correct model: trained_model.pth is ResNet50 (fc: 2048→39 classes)
  - predict() is now an internal function, not a broken route
  - Full input validation (file type, size)
  - Triton integration with local-PyTorch fallback
  - /explain endpoint backed by Anthropic Claude
  - Proper HTTP error handlers (400, 404, 500)
  - Confidence score passed to template
  - Auto-creates static/uploads directory
"""

import io
import os
import logging
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, jsonify, redirect, render_template, request, url_for
from PIL import Image
from torchvision import models, transforms
import pandas as pd

from CNN import idx_to_classes
from llm_service import get_disease_explanation
from triton_client import is_triton_available, predict_via_triton

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit

# ---------------------------------------------------------------------------
# Constants & Paths
# ---------------------------------------------------------------------------
APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "static" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}

NUM_CLASSES = 39

# ---------------------------------------------------------------------------
# CSV data
# ---------------------------------------------------------------------------
disease_info = pd.read_csv(APP_DIR / "disease_info.csv", encoding="cp1252")
supplement_info = pd.read_csv(APP_DIR / "supplement_info.csv", encoding="cp1252")

# ---------------------------------------------------------------------------
# Image pre-processing pipeline (ImageNet normalisation)
# ---------------------------------------------------------------------------
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
)

# ---------------------------------------------------------------------------
# Local PyTorch model (fallback / default when Triton is disabled)
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info("Using device: %s", device)

model: nn.Module | None = None


def _load_local_model() -> nn.Module:
    """
    Load the ResNet50 checkpoint from disk once.
    The existing trained_model.pth uses ResNet50 architecture
    with a custom fc layer: Linear(2048, 39).
    """
    global model
    if model is not None:
        return model

    checkpoint_path = APP_DIR / "trained_model.pth"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at '{checkpoint_path}'. "
            "Please place trained_model.pth in the App/ directory."
        )

    m = models.resnet50(weights=None)
    m.fc = nn.Linear(2048, NUM_CLASSES)  # Match the saved checkpoint
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)
    m.load_state_dict(state)
    m.to(device)
    m.eval()
    model = m
    logger.info("ResNet50 model loaded from %s on %s", checkpoint_path, device)
    return model


# Pre-load on startup (fail fast)
try:
    _load_local_model()
except Exception as exc:
    logger.warning("Could not load local model at startup: %s", exc)


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------
def _preprocess_image(image_file) -> tuple[torch.Tensor, np.ndarray]:
    """
    Open image from a Flask FileStorage object, apply transforms.
    Returns (torch_tensor [1,3,224,224], numpy_array [1,3,224,224] float32).
    """
    img = Image.open(image_file).convert("RGB")
    tensor = transform(img).unsqueeze(0)                   # (1,3,224,224)
    numpy_arr = tensor.numpy().astype(np.float32)          # for Triton
    return tensor, numpy_arr


def predict_local(tensor: torch.Tensor) -> tuple[int, float]:
    """Run inference on the local PyTorch model."""
    m = _load_local_model()
    with torch.no_grad():
        tensor = tensor.to(device)
        logits = m(tensor)                                 # (1, 39)
        probs = F.softmax(logits, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())
    return pred_idx, confidence


def run_inference(image_file) -> tuple[int, float, str]:
    """
    Orchestrate inference: try Triton first, fall back to local PyTorch.
    Returns (class_index, confidence, backend_used).
    """
    tensor, numpy_arr = _preprocess_image(image_file)

    if is_triton_available():
        try:
            start = time.perf_counter()
            pred_idx, confidence = predict_via_triton(numpy_arr)
            latency_ms = (time.perf_counter() - start) * 1000
            logger.info(
                "Triton inference: class=%d conf=%.3f latency=%.1fms",
                pred_idx, confidence, latency_ms,
            )
            return pred_idx, confidence, "triton"
        except Exception as exc:
            logger.warning("Triton inference failed, falling back to local: %s", exc)

    start = time.perf_counter()
    pred_idx, confidence = predict_local(tensor)
    latency_ms = (time.perf_counter() - start) * 1000
    logger.info(
        "Local inference: class=%d conf=%.3f latency=%.1fms",
        pred_idx, confidence, latency_ms,
    )
    return pred_idx, confidence, "local"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------
def _validate_image(file_storage) -> str | None:
    """
    Returns an error string if invalid, or None if OK.
    """
    if not file_storage or file_storage.filename == "":
        return "No file selected."

    ext = Path(file_storage.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return (
            f"Unsupported file type '{ext}'. "
            f"Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Try opening to validate it's actually an image
    try:
        file_storage.stream.seek(0)
        Image.open(file_storage.stream).verify()
        file_storage.stream.seek(0)
    except Exception:
        return "The uploaded file is not a valid image."

    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def home_page():
    return render_template("home.html")


@app.route("/contact")
def contact():
    return render_template("contact-us.html")


@app.route("/index")
def ai_engine_page():
    return render_template("index.html")


@app.route("/mobile-device")
def mobile_device_detected_page():
    return render_template("mobile-device.html")


@app.route("/submit", methods=["GET", "POST"])
def submit():
    if request.method == "GET":
        return redirect(url_for("ai_engine_page"))

    image_file = request.files.get("image")
    error = _validate_image(image_file)
    if error:
        return render_template("index.html", error=error), 400

    # Save uploaded file
    filename = image_file.filename
    file_path = UPLOAD_DIR / filename
    image_file.stream.seek(0)
    image_file.save(str(file_path))

    # Run inference
    try:
        image_file.stream.seek(0)
        pred_idx, confidence, backend = run_inference(image_file.stream)
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        return render_template("index.html", error=f"Inference failed: {exc}"), 500

    # Look up disease info
    try:
        title = disease_info["disease_name"][pred_idx]
        description = disease_info["description"][pred_idx]
        prevent = disease_info["Possible Steps"][pred_idx]
        image_url = disease_info["image_url"][pred_idx]
        supplement_name = supplement_info["supplement name"][pred_idx]
        supplement_image_url = supplement_info["supplement image"][pred_idx]
        supplement_buy_link = supplement_info["buy link"][pred_idx]
        disease_class = idx_to_classes.get(pred_idx, "Unknown")
    except (KeyError, IndexError) as exc:
        logger.error("Label lookup error for pred_idx=%d: %s", pred_idx, exc)
        return render_template("index.html", error="Label lookup failed."), 500

    return render_template(
        "submit.html",
        title=title,
        desc=description,
        prevent=prevent,
        image_url=image_url,
        pred=pred_idx,
        sname=supplement_name,
        simage=supplement_image_url,
        buy_link=supplement_buy_link,
        confidence=round(confidence * 100, 1),
        backend=backend,
        disease_class=disease_class,
        uploaded_image=filename,
    )


@app.route("/explain", methods=["POST"])
def explain():
    """
    LLM explanation endpoint.

    Request JSON:  { "disease": "Tomato___Early_blight" }
    Response JSON: { explanation, treatment_steps, preventive_measures,
                     fertilizers, severity, contagious, generated_at,
                     disease, readable_name }
    """
    data = request.get_json(silent=True)
    if not data or "disease" not in data:
        return jsonify({"error": "Request body must be JSON with a 'disease' key."}), 400

    disease_class = str(data["disease"]).strip()
    if not disease_class:
        return jsonify({"error": "'disease' must not be empty."}), 400

    try:
        result = get_disease_explanation(disease_class)
        return jsonify(result)
    except RuntimeError as exc:
        # e.g. API key missing
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        logger.error("Explain endpoint error: %s", exc, exc_info=True)
        return jsonify({"error": "Internal server error."}), 500


@app.route("/market", methods=["GET", "POST"])
def market():
    return render_template(
        "market.html",
        supplement_image=list(supplement_info["supplement image"]),
        supplement_name=list(supplement_info["supplement name"]),
        disease=list(disease_info["disease_name"]),
        buy=list(supplement_info["buy link"]),
    )


# ---------------------------------------------------------------------------
# Error handlers
# ---------------------------------------------------------------------------
@app.errorhandler(400)
def bad_request(e):
    return render_template("error.html", code=400, message="Bad Request", detail=str(e)), 400


@app.errorhandler(404)
def not_found(e):
    return render_template("error.html", code=404, message="Page Not Found", detail=str(e)), 404


@app.errorhandler(413)
def too_large(e):
    return render_template(
        "error.html", code=413, message="File Too Large", detail="Max upload size is 16 MB."
    ), 413


@app.errorhandler(500)
def server_error(e):
    return render_template("error.html", code=500, message="Internal Server Error", detail=str(e)), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=os.environ.get("FLASK_ENV") == "development",
    )
