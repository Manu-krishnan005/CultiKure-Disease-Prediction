"""
CultiKure Triton Inference Client
----------------------------------
Sends pre-processed image tensors to NVIDIA Triton Inference Server
via its HTTP REST API and returns the predicted class index + confidence.

Falls back to local PyTorch inference if Triton is unavailable or
if USE_TRITON env var is set to 'false'.
"""

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)

TRITON_URL = os.environ.get("TRITON_URL", "localhost:8000")
MODEL_NAME = "cultikure_vgg"
USE_TRITON = os.environ.get("USE_TRITON", "false").lower() == "true"


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / e.sum()


def predict_via_triton(img_array: np.ndarray) -> tuple[int, float]:
    """
    Send an image tensor to Triton and get (class_idx, confidence).

    Args:
        img_array: np.ndarray of shape (1, 3, 224, 224), dtype float32,
                   normalised with ImageNet mean/std.

    Returns:
        (predicted_class_index, confidence_score_0_to_1)

    Raises:
        RuntimeError: if Triton is unreachable or returns an error.
    """
    try:
        import tritonclient.http as httpclient  # soft import
    except ImportError:
        raise RuntimeError(
            "tritonclient[http] is not installed. "
            "Run: pip install tritonclient[http]"
        )

    client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)

    if not client.is_server_ready():
        raise RuntimeError(f"Triton server at {TRITON_URL} is not ready.")

    if not client.is_model_ready(MODEL_NAME):
        raise RuntimeError(f"Model '{MODEL_NAME}' is not ready on Triton.")

    # Build input
    infer_input = httpclient.InferInput("input", img_array.shape, "FP32")
    infer_input.set_data_from_numpy(img_array)

    # Build output
    infer_output = httpclient.InferRequestedOutput("output")

    response = client.infer(
        model_name=MODEL_NAME,
        inputs=[infer_input],
        outputs=[infer_output],
    )

    logits = response.as_numpy("output")[0]          # shape: (39,)
    probs = _softmax(logits)
    pred_idx = int(np.argmax(probs))
    confidence = float(probs[pred_idx])

    return pred_idx, confidence


def is_triton_available() -> bool:
    """Quick health check — returns True if Triton responds."""
    if not USE_TRITON:
        return False
    try:
        import tritonclient.http as httpclient
        client = httpclient.InferenceServerClient(url=TRITON_URL, verbose=False)
        return client.is_server_ready()
    except Exception:
        return False
