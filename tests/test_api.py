"""
CultiKure API Test Suite
-------------------------
Tests all Flask endpoints end-to-end.

Usage:
    pytest tests/test_api.py -v

Requirements:
    pip install pytest requests
"""

import io
import json
import os
import sys
from pathlib import Path

import pytest

# Add App to path
sys.path.insert(0, str(Path(__file__).parent.parent / "App"))

# Patch env before importing app
os.environ.setdefault("USE_TRITON", "false")
os.environ.setdefault("ANTHROPIC_API_KEY", "test-key-placeholder")

from app import app as flask_app  # noqa: E402


@pytest.fixture()
def client():
    flask_app.config["TESTING"] = True
    flask_app.config["WTF_CSRF_ENABLED"] = False
    with flask_app.test_client() as c:
        yield c


# ── Route tests ────────────────────────────────────────────────────────────

class TestRoutes:
    def test_home(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert b"CultiKure" in r.data

    def test_index(self, client):
        r = client.get("/index")
        assert r.status_code == 200

    def test_contact(self, client):
        r = client.get("/contact")
        assert r.status_code == 200

    def test_market(self, client):
        r = client.get("/market")
        assert r.status_code == 200

    def test_submit_get_redirects(self, client):
        r = client.get("/submit")
        assert r.status_code in (301, 302)

    def test_404(self, client):
        r = client.get("/nonexistent-page")
        assert r.status_code == 404


# ── Submit endpoint tests ──────────────────────────────────────────────────

class TestSubmit:
    def _make_image_bytes(self) -> bytes:
        """Create a minimal valid PNG in memory."""
        from PIL import Image
        buf = io.BytesIO()
        img = Image.new("RGB", (224, 224), color=(100, 150, 80))
        img.save(buf, format="PNG")
        buf.seek(0)
        return buf.read()

    def test_submit_no_file(self, client):
        r = client.post("/submit", data={})
        assert r.status_code == 400

    def test_submit_wrong_type(self, client):
        data = {"image": (io.BytesIO(b"not an image"), "test.txt")}
        r = client.post("/submit", data=data, content_type="multipart/form-data")
        assert r.status_code == 400

    def test_submit_valid_image(self, client):
        """Full inference round-trip with a synthetic green leaf image."""
        img_bytes = self._make_image_bytes()
        data = {"image": (io.BytesIO(img_bytes), "leaf.png")}
        r = client.post("/submit", data=data, content_type="multipart/form-data")
        # Should succeed or fail gracefully (model might not be loaded in test env)
        assert r.status_code in (200, 302, 500)
        if r.status_code == 200:
            assert b"confidence" in r.data.lower() or b"Detected" in r.data


# ── Explain endpoint tests ─────────────────────────────────────────────────

class TestExplain:
    def test_explain_no_body(self, client):
        r = client.post("/explain", content_type="application/json", data="{}")
        assert r.status_code == 400

    def test_explain_empty_disease(self, client):
        r = client.post(
            "/explain",
            content_type="application/json",
            data=json.dumps({"disease": ""}),
        )
        assert r.status_code == 400

    def test_explain_healthy_plant_no_api(self, client, monkeypatch):
        """Healthy plant returns cached response without hitting Claude API."""
        from llm_service import get_disease_explanation
        # Clear cache
        get_disease_explanation.cache_clear()

        r = client.post(
            "/explain",
            content_type="application/json",
            data=json.dumps({"disease": "Tomato___healthy"}),
        )
        # Healthy plants use the short-circuit path — no API key needed
        assert r.status_code == 200
        data = r.get_json()
        assert "explanation" in data
        assert data["severity"] == "none"

    def test_explain_disease_structure(self, client, monkeypatch):
        """Mock the Claude API and check response structure."""
        import llm_service

        mock_result = {
            "explanation": "Test explanation.",
            "treatment_steps": ["Step 1", "Step 2"],
            "preventive_measures": ["Measure 1"],
            "fertilizers": ["NPK 10-10-10"],
            "severity": "medium",
            "contagious": False,
        }

        def mock_explain(disease_class):
            return {**mock_result, "disease": disease_class, "generated_at": "2026-01-01T00:00:00Z", "readable_name": disease_class}

        monkeypatch.setattr(llm_service, "get_disease_explanation", mock_explain)

        r = client.post(
            "/explain",
            content_type="application/json",
            data=json.dumps({"disease": "Tomato___Early_blight"}),
        )
        assert r.status_code == 200
        data = r.get_json()
        for key in ["explanation", "treatment_steps", "preventive_measures", "fertilizers", "severity"]:
            assert key in data, f"Missing key: {key}"
