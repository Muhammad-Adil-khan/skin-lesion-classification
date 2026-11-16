"""Tests for image validation, quality warnings and the resize methods."""
import io

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from app.config import Settings, settings
from app.utils.preprocessing import check_image_quality, resize_to_model_input
from helpers import make_image_bytes, solid_image_bytes, upload


# ---------- size limits ----------
def test_image_with_too_many_pixels_is_rejected(client):
    """A tiny file can decode into a huge image (decompression bomb), so pixels are capped too."""
    buffer = io.BytesIO()
    Image.new("L", (6000, 6000)).save(buffer, "PNG")  # 36 million pixels, only a few KB on disk
    assert len(buffer.getvalue()) < 1_000_000

    response = upload(client, buffer.getvalue(), "bomb.png", "image/png")
    assert response.status_code == 422
    assert "too large" in response.json()["detail"].lower()


def test_pillow_decompression_bomb_is_rejected_cleanly(client):
    buffer = io.BytesIO()
    Image.new("1", (14000, 14000)).save(buffer, "PNG")  # ~196 million pixels: Pillow itself refuses these
    response = upload(client, buffer.getvalue(), "bomb.png", "image/png")
    assert response.status_code == 422


def test_pixel_cap_is_configurable(client, monkeypatch):
    monkeypatch.setattr(settings, "max_image_pixels", 1000)
    assert upload(client, make_image_bytes(size=(300, 200))).status_code == 422


def test_tiny_image_is_rejected(client):
    response = upload(client, make_image_bytes(size=(32, 32)))
    assert response.status_code == 422
    assert "too small" in response.json()["detail"].lower()


def test_image_at_minimum_size_is_accepted(client):
    side = settings.min_image_side
    assert upload(client, make_image_bytes(size=(side, side))).status_code == 200


# ---------- quality warnings ----------
def test_normal_image_has_no_quality_warnings(client):
    assert upload(client, make_image_bytes(seed=1)).json()["warnings"] == []


def test_blank_image_warns_about_blur_and_contrast(client):
    warnings = upload(client, solid_image_bytes((128, 128, 128))).json()["warnings"]
    assert any("blurry" in w for w in warnings)
    assert any("contrast" in w for w in warnings)


def test_dark_image_warns(client):
    warnings = upload(client, solid_image_bytes((5, 5, 5))).json()["warnings"]
    assert any("dark" in w for w in warnings)


def test_overexposed_image_warns(client):
    warnings = upload(client, solid_image_bytes((250, 250, 250))).json()["warnings"]
    assert any("bright" in w for w in warnings)


def test_blurring_an_image_triggers_the_blur_warning():
    rng = np.random.default_rng(0)
    sharp = rng.uniform(0, 255, (224, 224, 3)).astype(np.float32)
    blurred = np.asarray(Image.fromarray(sharp.astype(np.uint8)).resize((8, 8)).resize((224, 224), Image.BILINEAR),
                         dtype=np.float32)
    assert not any("blurry" in w for w in check_image_quality(sharp))
    assert any("blurry" in w for w in check_image_quality(blurred))


# ---------- resize methods ----------
@pytest.fixture
def checkerboard():
    """High-frequency pattern: shows clearly how the resize methods differ."""
    pixels = (np.indices((450, 600)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return Image.fromarray(np.stack([pixels] * 3, axis=-1))


@pytest.mark.parametrize("method", ["tf_bilinear", "pil_bilinear", "pil_nearest"])
def test_every_resize_method_gives_the_model_input_format(method, checkerboard, monkeypatch):
    monkeypatch.setattr(settings, "resize_method", method)
    result = resize_to_model_input(checkerboard, (224, 224))
    assert result.shape == (224, 224, 3)
    assert result.dtype == np.float32
    assert 0 <= result.min() and result.max() <= 255


def test_resize_keeps_the_0_255_range_because_the_model_rescales_itself(monkeypatch):
    """The model has its own Rescaling(1/255) layer, so values must NOT be pre-scaled to 0-1."""
    white = Image.new("RGB", (300, 300), (255, 255, 255))
    for method in ("tf_bilinear", "pil_bilinear", "pil_nearest"):
        monkeypatch.setattr(settings, "resize_method", method)
        assert resize_to_model_input(white, (224, 224)).max() == pytest.approx(255)


def test_resize_methods_really_differ(checkerboard, monkeypatch):
    outputs = {}
    for method in ("tf_bilinear", "pil_bilinear", "pil_nearest"):
        monkeypatch.setattr(settings, "resize_method", method)
        outputs[method] = resize_to_model_input(checkerboard, (224, 224))
    assert np.abs(outputs["tf_bilinear"] - outputs["pil_bilinear"]).mean() > 1
    assert np.abs(outputs["tf_bilinear"] - outputs["pil_nearest"]).mean() > 1


def test_resize_method_can_be_set_from_the_environment(monkeypatch):
    monkeypatch.setenv("SKIN_RESIZE_METHOD", "pil_nearest")
    assert Settings().resize_method == "pil_nearest"


def test_unknown_resize_method_is_rejected():
    with pytest.raises(ValidationError):
        Settings(resize_method="bogus")
