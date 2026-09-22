"""Backend tests for EmotionAI API.

These tests use FastAPI's TestClient and mock the model loader so
TensorFlow does not need to be loaded during CI/CD.

Run (from repo root, venv active):
    $env:PYTHONPATH = "d:\\Apna Kaam\\Emotion_Detector\\backend"
    pytest backend/tests/test_api.py -v

Root of previous failures: routes.py calls loader.is_loaded() as a
function reference imported from app.ml.loader. Patching must happen
BEFORE the app modules are imported, with module cache cleared first.
"""

from __future__ import annotations

import io
import sys
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image


# ── Shared mock metadata ──────────────────────────────────────────────────────

MOCK_META = {
    "model_name": "emotion_cnn",
    "model_version": "0.1.0",
    "task": "facial_expression_recognition",
    "num_classes": 5,
    "index_to_class": {
        "0": "Angry", "1": "Happy", "2": "Sad", "3": "Surprise", "4": "Neutral"
    },
    "class_to_index": {
        "Angry": 0, "Happy": 1, "Sad": 2, "Surprise": 3, "Neutral": 4
    },
    "input": {
        "height": 48, "width": 48, "channels": 1,
        "color_mode": "grayscale", "normalization": "divide_by_255",
    },
    "disclaimer": "Classifies visible facial expressions.",
}


def _clear_app_modules() -> None:
    """Remove cached app modules so patches apply on fresh import."""
    for key in list(sys.modules.keys()):
        if key.startswith("app"):
            del sys.modules[key]


# ── App fixture ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def client():
    """Create a TestClient with model loading fully mocked.

    Patches are applied at the loader module level before any app module
    is imported, so routes.py picks up the mocked functions.
    """
    _clear_app_modules()

    with (
        patch("app.ml.loader.load_model"),
        patch("app.ml.loader.is_loaded", return_value=True),
        patch("app.ml.loader.get_metadata", return_value=MOCK_META),
    ):
        from app.main import app  # noqa: PLC0415
        with TestClient(app, raise_server_exceptions=False) as c:
            yield c

    _clear_app_modules()


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealth:
    def test_health_returns_200(self, client: TestClient) -> None:
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_schema(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert "status" in data
        assert "model_loaded" in data
        assert "version" in data

    def test_health_status_ok(self, client: TestClient) -> None:
        data = client.get("/health").json()
        assert data["status"] == "ok"

    def test_health_no_sensitive_info(self, client: TestClient) -> None:
        resp_text = client.get("/health").text
        assert "Traceback" not in resp_text
        assert "models/" not in resp_text


# ── /model-info ───────────────────────────────────────────────────────────────

class TestModelInfo:
    def test_model_info_returns_200(self, client: TestClient) -> None:
        resp = client.get("/model-info")
        assert resp.status_code == 200

    def test_model_info_has_expressions(self, client: TestClient) -> None:
        data = client.get("/model-info").json()
        assert "supported_expressions" in data
        assert set(data["supported_expressions"]) == {
            "Angry", "Happy", "Sad", "Surprise", "Neutral"
        }

    def test_model_info_has_input_spec(self, client: TestClient) -> None:
        data = client.get("/model-info").json()
        assert data["input"]["height"] == 48
        assert data["input"]["width"] == 48
        assert data["input"]["channels"] == 1

    def test_model_info_has_disclaimer(self, client: TestClient) -> None:
        data = client.get("/model-info").json()
        assert "disclaimer" in data
        assert len(data["disclaimer"]) > 10


# ── /predict ─────────────────────────────────────────────────────────────────

def _make_jpeg_bytes(size: tuple[int, int] = (200, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(128, 90, 60)).save(buf, format="JPEG")
    return buf.getvalue()


def _make_png_bytes(size: tuple[int, int] = (200, 200)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color=(100, 150, 200)).save(buf, format="PNG")
    return buf.getvalue()


def _post_image(
    client: TestClient,
    image_bytes: bytes,
    content_type: str = "image/jpeg",
):
    return client.post(
        "/predict",
        files={"file": ("test.jpg", image_bytes, content_type)},
    )


class TestPredict:
    def test_no_face_returns_200_face_detected_false(self, client: TestClient) -> None:
        from app.ml.inference import InferenceResult  # noqa: PLC0415
        result = InferenceResult(
            face_detected=False, label=None, confidence=None,
            probabilities=None, processing_time_ms=12.3,
        )
        with patch("app.api.routes.inference.predict", return_value=result):
            resp = _post_image(client, _make_jpeg_bytes())
        assert resp.status_code == 200
        data = resp.json()
        assert data["face_detected"] is False
        assert data["prediction"] is None

    def test_face_detected_returns_prediction(self, client: TestClient) -> None:
        from app.ml.inference import InferenceResult  # noqa: PLC0415
        result = InferenceResult(
            face_detected=True, label="Happy", confidence=0.87,
            probabilities={
                "Angry": 0.02, "Happy": 0.87, "Sad": 0.03,
                "Surprise": 0.01, "Neutral": 0.07,
            },
            processing_time_ms=45.0,
        )
        with patch("app.api.routes.inference.predict", return_value=result):
            resp = _post_image(client, _make_jpeg_bytes())
        assert resp.status_code == 200
        data = resp.json()
        assert data["face_detected"] is True
        assert data["prediction"]["label"] == "Happy"
        assert 0 <= data["prediction"]["confidence"] <= 1
        assert set(data["probabilities"].keys()) == {
            "Angry", "Happy", "Sad", "Surprise", "Neutral"
        }

    def test_probabilities_are_floats(self, client: TestClient) -> None:
        from app.ml.inference import InferenceResult  # noqa: PLC0415
        result = InferenceResult(
            face_detected=True, label="Neutral", confidence=0.60,
            probabilities={
                "Angry": 0.05, "Happy": 0.10, "Sad": 0.10,
                "Surprise": 0.15, "Neutral": 0.60,
            },
            processing_time_ms=50.0,
        )
        with patch("app.api.routes.inference.predict", return_value=result):
            resp = _post_image(client, _make_jpeg_bytes())
        for val in resp.json()["probabilities"].values():
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0

    def test_wrong_content_type_returns_400(self, client: TestClient) -> None:
        resp = _post_image(client, b"not an image", content_type="text/plain")
        assert resp.status_code == 400
        data = resp.json()
        assert data["success"] is False
        assert "error" in data

    def test_empty_file_returns_400(self, client: TestClient) -> None:
        resp = _post_image(client, b"")
        assert resp.status_code == 400

    def test_accepts_png(self, client: TestClient) -> None:
        from app.ml.inference import InferenceResult  # noqa: PLC0415
        result = InferenceResult(
            face_detected=False, label=None, confidence=None,
            probabilities=None, processing_time_ms=10.0,
        )
        with patch("app.api.routes.inference.predict", return_value=result):
            resp = _post_image(client, _make_png_bytes(), content_type="image/png")
        assert resp.status_code == 200

    def test_invalid_image_bytes_returns_400(self, client: TestClient) -> None:
        with patch(
            "app.api.routes.inference.predict",
            side_effect=ValueError("Could not decode image"),
        ):
            resp = _post_image(client, _make_jpeg_bytes())
        assert resp.status_code == 400

    def test_response_has_processing_time(self, client: TestClient) -> None:
        from app.ml.inference import InferenceResult  # noqa: PLC0415
        result = InferenceResult(
            face_detected=False, label=None, confidence=None,
            probabilities=None, processing_time_ms=33.7,
        )
        with patch("app.api.routes.inference.predict", return_value=result):
            resp = _post_image(client, _make_jpeg_bytes())
        data = resp.json()
        assert "processing_time_ms" in data
        assert isinstance(data["processing_time_ms"], (int, float))


# ── CORS ──────────────────────────────────────────────────────────────────────

class TestCORS:
    def test_cors_allowed_origin(self, client: TestClient) -> None:
        resp = client.get("/health", headers={"Origin": "http://localhost:3000"})
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_cors_options_preflight(self, client: TestClient) -> None:
        resp = client.options(
            "/predict",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        assert resp.status_code in (200, 204)
