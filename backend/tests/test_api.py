"""Core API tests: startup, health, uploads, validation and response structure.

Run from the backend/ folder:   python -m pytest -v
"""
import hashlib
import json

import pytest

from app.config import settings
from app.main import app
from helpers import make_image_bytes, upload


# ---------- startup / basic endpoints ----------
def test_model_loaded_at_startup(client):
    assert app.state.prediction_service.is_ready


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["input_size"] == [224, 224]
    assert len(body["classes"]) == 4
    assert body["explain_available"] is True
    assert body["resize_method"] == settings.resize_method
    assert body["auth_required"] is False


def test_health_reports_the_model_file_fingerprint(client):
    expected = hashlib.sha256(settings.model_path.read_bytes()).hexdigest()
    assert client.get("/health").json()["model_sha256"] == expected


# ---------- class mapping ----------
def test_class_mapping_comes_from_class_names_file(client):
    with open(settings.class_names_path) as file:
        expected = [name.upper() for name in json.load(file)]
    assert app.state.prediction_service.class_names == expected
    assert set(expected) == {"NV", "MEL", "BCC", "BKL"}


# ---------- valid uploads ----------
@pytest.mark.parametrize(
    "fmt,filename,content_type",
    [("JPEG", "a.jpg", "image/jpeg"), ("JPEG", "a.jpeg", "image/jpeg"), ("PNG", "a.png", "image/png")],
)
def test_valid_upload_response_structure(client, fmt, filename, content_type):
    response = upload(client, make_image_bytes(fmt), filename, content_type)
    assert response.status_code == 200
    body = response.json()
    assert body["success"] is True
    assert set(body) >= {"success", "predicted_class", "confidence", "predictions", "disclaimer"}
    assert set(body["predictions"]) == {"NV", "MEL", "BCC", "BKL"}


def test_confidence_scores_are_consistent(client):
    body = upload(client, make_image_bytes()).json()
    probs = body["predictions"]
    assert all(0.0 <= p <= 1.0 for p in probs.values())
    assert sum(probs.values()) == pytest.approx(1.0, abs=1e-3)  # softmax output
    assert body["predicted_class"] == max(probs, key=probs.get)
    assert body["confidence"] == pytest.approx(max(probs.values()))


def test_png_with_alpha_channel_is_accepted(client):
    response = upload(client, make_image_bytes("PNG", mode="RGBA"), "a.png", "image/png")
    assert response.status_code == 200


def test_grayscale_image_is_accepted(client):
    response = upload(client, make_image_bytes("PNG", mode="L"), "a.png", "image/png")
    assert response.status_code == 200


def test_same_image_gives_same_prediction(client):
    data = make_image_bytes()
    assert upload(client, data).json() == upload(client, data).json()


# ---------- invalid uploads ----------
def test_wrong_file_type_rejected(client):
    response = upload(client, b"hello", "notes.txt", "text/plain")
    assert response.status_code == 415
    assert response.json()["detail"]


def test_wrong_extension_rejected(client):
    response = upload(client, make_image_bytes(), "lesion.gif", "image/jpeg")
    assert response.status_code == 415


def test_corrupted_image_rejected(client):
    response = upload(client, b"this is not really a jpeg", "bad.jpg", "image/jpeg")
    assert response.status_code == 422


def test_truncated_image_rejected(client):
    data = make_image_bytes("PNG", size=(500, 500))
    response = upload(client, data[: len(data) // 2], "cut.png", "image/png")
    assert response.status_code == 422


def test_empty_file_rejected(client):
    assert upload(client, b"").status_code == 400


def test_missing_file_field(client):
    assert client.post("/api/v1/predict").status_code == 422


def test_oversized_file_rejected(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    # Incompressible PNG of about 1.9 MB: over the 1 MB limit, under the middleware's multipart allowance
    response = upload(client, make_image_bytes("PNG", size=(800, 800)), "big.png", "image/png")
    assert response.status_code == 413


# ---------- model unavailable ----------
def test_predict_returns_503_when_model_not_loaded(client, monkeypatch):
    monkeypatch.setattr(app.state.prediction_service, "model", None)
    assert upload(client, make_image_bytes()).status_code == 503
    assert client.get("/health").json()["status"] == "degraded"
