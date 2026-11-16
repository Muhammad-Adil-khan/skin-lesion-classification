"""Tests for the 'uncertain result' rules, response metadata and Grad-CAM heatmaps."""
import base64
import io
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from app.config import settings
from app.main import app
from app.services.prediction_service import assess_uncertainty
from helpers import make_image_bytes, upload


# ---------- uncertainty rules (pure function) ----------
def test_confident_result_is_not_flagged():
    assert assess_uncertainty(np.array([0.90, 0.05, 0.03, 0.02])) == []


def test_low_confidence_is_flagged():
    reasons = assess_uncertainty(np.array([0.40, 0.30, 0.20, 0.10]))
    assert any("below" in r for r in reasons)


def test_two_close_classes_are_flagged_even_if_top_is_above_the_confidence_limit():
    reasons = assess_uncertainty(np.array([0.55, 0.40, 0.03, 0.02]))  # top 0.55 >= 0.50, but gap is only 0.15
    assert len(reasons) == 1 and "close together" in reasons[0]


def test_thresholds_come_from_settings(monkeypatch):
    monkeypatch.setattr(settings, "min_confidence", 0.0)
    monkeypatch.setattr(settings, "min_margin", 0.0)
    assert assess_uncertainty(np.array([0.26, 0.25, 0.25, 0.24])) == []


# ---------- fields in the response ----------
def test_response_has_the_new_fields(client):
    body = upload(client, make_image_bytes(seed=3)).json()
    assert isinstance(body["is_uncertain"], bool)
    assert body["is_uncertain"] == bool(body["uncertainty_reasons"])
    assert isinstance(body["warnings"], list)
    assert body["explanation_png_base64"] is None  # only on request
    assert len(body["model_version"]) == 12
    assert body["model_version"] == client.get("/health").json()["model_sha256"][:12]
    assert "not a clinically validated" in body["disclaimer"]


def test_original_response_fields_are_unchanged(client):
    """The documented format must keep working for existing clients."""
    body = upload(client, make_image_bytes(seed=3)).json()
    assert {"success", "predicted_class", "confidence", "predictions"} <= set(body)


# ---------- Grad-CAM ----------
def decode_overlay(body) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(body["explanation_png_base64"])))


def test_explain_returns_a_png_overlay_the_size_of_the_upload(client):
    body = upload(client, make_image_bytes(size=(300, 200), seed=4), params={"explain": "true"}).json()
    overlay = decode_overlay(body)
    assert overlay.format == "PNG" and overlay.mode == "RGB"
    assert overlay.size == (300, 200)


def test_large_overlays_are_shrunk_but_keep_their_proportions(client):
    body = upload(client, make_image_bytes(size=(1000, 500), seed=4), params={"explain": "true"}).json()
    assert decode_overlay(body).size == (512, 256)


def test_overlay_actually_differs_from_the_original(client):
    original = make_image_bytes("PNG", size=(300, 200), seed=5)
    body = upload(client, original, "a.png", "image/png", params={"explain": "true"}).json()
    overlay = np.asarray(decode_overlay(body), dtype=float)
    base = np.asarray(Image.open(io.BytesIO(original)).convert("RGB"), dtype=float)
    assert np.abs(overlay - base).mean() > 5


def test_explain_false_gives_no_overlay(client):
    body = upload(client, make_image_bytes(), params={"explain": "false"}).json()
    assert body["explanation_png_base64"] is None


def test_explain_does_not_change_the_prediction(client):
    data = make_image_bytes(seed=6)
    plain = upload(client, data).json()
    explained = upload(client, data, params={"explain": "true"}).json()
    for key in ("predicted_class", "confidence", "predictions"):
        assert plain[key] == explained[key]


def test_heatmap_values_are_normalised_to_0_1(client):
    service = app.state.prediction_service
    tensor = np.random.default_rng(0).uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    predicted = int(np.asarray(service._infer_fn(tensor))[0].argmax())

    for class_index in range(4):
        heatmap = service._gradcam_heatmap(tensor, class_index)
        assert heatmap.ndim == 2
        assert heatmap.min() >= 0 and heatmap.max() <= 1.0 + 1e-6
        # Scaled so the strongest region is 1. A class with no supporting evidence gives an all-zero map.
        assert heatmap.max() == pytest.approx(1.0) or heatmap.max() == 0.0

    assert service._gradcam_heatmap(tensor, predicted).max() == pytest.approx(1.0)


def test_empty_heatmap_is_reported_instead_of_silently_showing_nothing(client, monkeypatch):
    service = app.state.prediction_service
    monkeypatch.setattr(service, "_gradcam_heatmap", lambda tensor, class_index: np.zeros((7, 7), dtype=np.float32))
    body = upload(client, make_image_bytes(), params={"explain": "true"}).json()
    assert body["explanation_png_base64"] is not None
    assert any("heatmap is empty" in w for w in body["warnings"])


def test_heatmap_depends_on_the_class_asked_about(client, real_model):
    service = app.state.prediction_service
    tensor = np.random.default_rng(1).uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    maps = [service._gradcam_heatmap(tensor, class_index=i) for i in range(4)]
    assert not np.allclose(maps[0], maps[1])


def test_logits_path_used_by_gradcam_reproduces_the_model_output(client):
    """Grad-CAM works on raw scores; this guards that the rebuilt softmax equals the real output."""
    service = app.state.prediction_service
    tensor = np.random.default_rng(2).uniform(0, 255, (1, 224, 224, 3)).astype(np.float32)
    _, features = service._grad_model(tensor, training=False)
    logits = np.asarray(features) @ np.asarray(service._output_layer.kernel) + np.asarray(service._output_layer.bias)
    rebuilt = np.exp(logits) / np.exp(logits).sum()
    assert np.allclose(rebuilt, np.asarray(service._infer_fn(tensor)), atol=1e-4)


def test_explain_degrades_gracefully_if_gradcam_is_unavailable(client, monkeypatch):
    monkeypatch.setattr(app.state.prediction_service, "_grad_model", None)
    response = upload(client, make_image_bytes(), params={"explain": "true"})
    assert response.status_code == 200
    body = response.json()
    assert body["explanation_png_base64"] is None
    assert any("heatmap" in w for w in body["warnings"])
    assert client.get("/health").json()["explain_available"] is False


# ---------- concurrency ----------
def test_parallel_requests_all_succeed_with_identical_results(client):
    data = make_image_bytes(seed=7)

    def call(explain):
        return upload(client, data, params={"explain": str(explain).lower()})

    with ThreadPoolExecutor(max_workers=6) as pool:
        responses = list(pool.map(call, [False, True] * 4))

    assert all(r.status_code == 200 for r in responses)
    assert len({str(r.json()["predictions"]) for r in responses}) == 1
