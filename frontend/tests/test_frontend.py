"""Frontend tests. Run from the frontend/ folder:  python -m pytest -v

Most tests mock the backend. The last test talks to a REAL backend and is skipped
unless one is running at SKIN_API_BASE_URL (default http://localhost:8000).
"""
import base64
import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests
from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import streamlit_app as app  # noqa: E402

GOOD_BODY = {
    "success": True,
    "predicted_class": "MEL",
    "confidence": 0.9,
    "predictions": {"BCC": 0.03, "BKL": 0.02, "MEL": 0.9, "NV": 0.05},
    "is_uncertain": False,
    "uncertainty_reasons": [],
    "warnings": [],
    "explanation_png_base64": None,
    "model_version": "abc123def456",
    "disclaimer": "Educational decision-support prototype only.",
}


def tiny_png_base64() -> str:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (8, 8), (200, 50, 50)).save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode()


def valid_image_bytes() -> bytes:
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (40, 30), (180, 120, 100)).save(buffer, "PNG")
    return buffer.getvalue()


def fake_response(status=200, body=None, json_error=False, headers=None):
    response = MagicMock(status_code=status, headers=headers or {})
    if json_error:
        response.json.side_effect = ValueError("not json")
    else:
        response.json.return_value = body
    return response


def call(response=None, exception=None, **kwargs):
    with patch("streamlit_app.requests.post", return_value=response, side_effect=exception) as post:
        result = app.request_prediction(b"bytes", "x.jpg", "image/jpeg", **kwargs)
    return result, post


# ---------------------------------------------------------------- request_prediction
def test_success_returns_result():
    (result, error), _ = call(fake_response(200, GOOD_BODY))
    assert error is None and result["predicted_class"] == "MEL"


def test_backend_down_gives_friendly_error():
    (result, error), _ = call(exception=requests.exceptions.ConnectionError())
    assert result is None and "Cannot connect" in error


def test_timeout_gives_friendly_error():
    (_, error), _ = call(exception=requests.exceptions.Timeout())
    assert "too long" in error


@pytest.mark.parametrize("status,detail", [(415, "Unsupported file type."), (422, "Not a valid image."),
                                           (503, "Model is not available."), (401, "Invalid or missing API key.")])
def test_http_errors_show_backend_detail(status, detail):
    (result, error), _ = call(fake_response(status, {"detail": detail}))
    assert result is None and detail in error and str(status) in error


def test_rate_limit_error_tells_the_user_how_long_to_wait():
    (_, error), _ = call(fake_response(429, {"detail": "Too many requests."}, headers={"Retry-After": "17"}))
    assert "Too many requests." in error and "17 seconds" in error and "429" in error


def test_fastapi_validation_error_list_is_handled():
    (_, error), _ = call(fake_response(422, {"detail": [{"msg": "Field required"}]}))
    assert "Field required" in error


def test_non_json_response_is_handled():
    (_, error), _ = call(fake_response(502, json_error=True))
    assert "unexpected response" in error


def test_response_missing_fields_is_handled():
    (_, error), _ = call(fake_response(200, {"success": True}))
    assert "missing" in error


def test_explain_flag_is_sent_as_a_query_parameter():
    _, post = call(fake_response(200, GOOD_BODY), explain=True)
    assert post.call_args.kwargs["params"] == {"explain": "true"}
    _, post = call(fake_response(200, GOOD_BODY), explain=False)
    assert post.call_args.kwargs["params"] is None


def test_api_key_is_sent_when_configured(monkeypatch):
    monkeypatch.setattr(app, "API_KEY", "secret-key")
    _, post = call(fake_response(200, GOOD_BODY))
    assert post.call_args.kwargs["headers"] == {"X-API-Key": "secret-key"}


def test_no_key_header_when_not_configured(monkeypatch):
    monkeypatch.setattr(app, "API_KEY", "")
    _, post = call(fake_response(200, GOOD_BODY))
    assert post.call_args.kwargs["headers"] == {}


# ---------------------------------------------------------------- health check
def health(body, monkeypatch=None):
    with patch("streamlit_app.requests.get", return_value=fake_response(200, body)):
        return app.check_backend_health()


def test_health_check_backend_down():
    with patch("streamlit_app.requests.get", side_effect=requests.exceptions.ConnectionError()):
        healthy, message = app.check_backend_health()
    assert healthy is False and "not reachable" in message


def test_health_check_model_not_loaded():
    healthy, message = health({"model_loaded": False})
    assert healthy is False and "model is not loaded" in message


def test_health_check_flags_a_missing_api_key(monkeypatch):
    monkeypatch.setattr(app, "API_KEY", "")
    healthy, message = health({"model_loaded": True, "auth_required": True})
    assert healthy is False and "API key" in message


def test_health_check_ok_with_api_key(monkeypatch):
    monkeypatch.setattr(app, "API_KEY", "secret-key")
    assert health({"model_loaded": True, "auth_required": True})[0] is True


# ---------------------------------------------------------------- the page, driven like a user
def new_app(post_response=None, post_exception=None):
    """Start the real page with the backend mocked. Returns (AppTest, post mock, patches to stop)."""
    patches = [
        patch("requests.get", return_value=fake_response(200, {"model_loaded": True})),
        patch("requests.post", return_value=post_response, side_effect=post_exception),
    ]
    mocks = [p.start() for p in patches]
    at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
    return at, mocks[1], patches


@pytest.fixture
def run_page():
    started = []

    def start(post_response=None, post_exception=None):
        at, post, patches = new_app(post_response, post_exception)
        started.extend(patches)
        return at, post

    yield start
    for p in started:
        p.stop()


def upload_and_predict(at, name="lesion.jpg", content=None):
    at.file_uploader[0].upload(name, content or valid_image_bytes(), "image/jpeg").run()
    at.button[0].click().run()
    return at


def test_page_renders_title_disclaimer_and_uploader(run_page):
    at, _ = run_page()
    assert not at.exception
    assert at.title[0].value == "Skin Disease Detection System"
    assert "not** a clinically validated" in at.info[0].value
    assert len(at.file_uploader) == 1
    assert len(at.button) == 0  # no Predict button until an image is chosen


def test_page_shows_backend_down_in_the_sidebar_without_crashing():
    with patch("requests.get", side_effect=requests.exceptions.ConnectionError()):
        at = AppTest.from_file(str(Path(app.__file__)), default_timeout=30).run()
    assert not at.exception and at.sidebar.error


def test_upload_then_predict_shows_class_confidence_and_all_bars(run_page):
    at, post = run_page(fake_response(200, GOOD_BODY))
    at.file_uploader[0].upload("lesion.jpg", valid_image_bytes(), "image/jpeg").run()
    assert len(at.button) == 1 and not at.metric  # preview + button, nothing predicted yet

    at.button[0].click().run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric} == {"Predicted class": "MEL", "Confidence": "90.0%"}
    assert len(at.get("progress")) == 4
    assert any("not a clinically validated" in w.value or "prototype" in w.value for w in at.warning)
    assert post.call_count == 1


def test_corrupted_upload_does_not_crash_the_page_and_the_backend_explains_why(run_page):
    """A text file renamed to .jpg cannot be previewed, but the user must still get a clear message."""
    at, post = run_page(fake_response(422, {"detail": "The file is not a valid or readable image."}))
    at.file_uploader[0].upload("fake.jpg", b"definitely not an image", "image/jpeg").run()
    assert not at.exception
    assert any("could not be previewed" in w.value for w in at.warning)

    at.button[0].click().run()
    assert not at.exception
    assert any("not a valid or readable image" in e.value for e in at.error)
    assert post.call_count == 1


def test_uncertain_result_shows_a_clear_warning(run_page):
    body = {**GOOD_BODY, "confidence": 0.41, "is_uncertain": True,
            "uncertainty_reasons": ["The top confidence (41%) is below 50%."]}
    at, _ = run_page(fake_response(200, body))
    upload_and_predict(at)
    assert any("Uncertain result" in w.value and "below 50%" in w.value for w in at.warning)
    assert at.metric[0].label == "Most likely class"


def test_image_quality_warnings_are_shown(run_page):
    at, _ = run_page(fake_response(200, {**GOOD_BODY, "warnings": ["The image looks blurry or out of focus."]}))
    upload_and_predict(at)
    assert any("blurry" in i.value for i in at.info)


def test_heatmap_is_shown_when_the_backend_sends_one(run_page):
    at, _ = run_page(fake_response(200, {**GOOD_BODY, "explanation_png_base64": tiny_png_base64()}))
    upload_and_predict(at)
    assert not at.exception
    assert any("Grad-CAM" in e.label for e in at.expander)
    assert len(at.get("image")) == 2  # the uploaded preview + the heatmap


def test_no_heatmap_section_when_none_is_returned(run_page):
    at, _ = run_page(fake_response(200, GOOD_BODY))
    upload_and_predict(at)
    assert not at.expander and len(at.get("image")) == 1


def test_heatmap_checkbox_controls_what_is_requested(run_page):
    at, post = run_page(fake_response(200, GOOD_BODY))
    at.file_uploader[0].upload("a.jpg", valid_image_bytes(), "image/jpeg").run()
    assert at.checkbox[0].value is True
    at.button[0].click().run()
    assert post.call_args.kwargs["params"] == {"explain": "true"}

    at.checkbox[0].uncheck().run()
    at.button[0].click().run()
    assert post.call_args.kwargs["params"] is None


def test_backend_error_is_shown_as_a_readable_message(run_page):
    at, _ = run_page(fake_response(422, {"detail": "The file is not a valid or readable image."}))
    upload_and_predict(at)
    assert not at.exception
    assert any("not a valid or readable image" in e.value and "422" in e.value for e in at.error)
    assert not at.metric


def test_backend_unreachable_shows_a_friendly_error(run_page):
    at, _ = run_page(post_exception=requests.exceptions.ConnectionError())
    upload_and_predict(at)
    assert not at.exception
    assert any("Cannot connect" in e.value for e in at.error)


def test_result_stays_on_screen_when_the_page_reruns(run_page):
    """Toggling a widget reruns the script; the result must not vanish or trigger a new request."""
    at, post = run_page(fake_response(200, GOOD_BODY))
    upload_and_predict(at)
    at.checkbox[0].uncheck().run()
    assert at.metric and at.metric[0].value == "MEL"
    assert post.call_count == 1


def test_choosing_a_different_image_clears_the_old_result(run_page):
    at, _ = run_page(fake_response(200, GOOD_BODY))
    upload_and_predict(at, name="first.jpg")
    assert at.metric
    at.file_uploader[0].upload("second.jpg", valid_image_bytes() + b" ", "image/jpeg").run()
    assert not at.metric and not at.error


# ---------------------------------------------------------------- real backend (skipped if not running)
def _backend_up():
    try:
        return requests.get(app.HEALTH_URL, timeout=2).json().get("model_loaded") is True
    except Exception:
        return False


@pytest.mark.skipif(not _backend_up(), reason="real backend not running")
def test_live_frontend_to_backend_roundtrip():
    import numpy as np
    from PIL import Image

    buffer = io.BytesIO()
    Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)).save(buffer, "PNG")
    result, error = app.request_prediction(buffer.getvalue(), "x.png", "image/png", explain=True)
    assert error is None
    assert set(result["predictions"]) == {"NV", "MEL", "BCC", "BKL"}
    assert isinstance(result["is_uncertain"], bool) and len(result["model_version"]) == 12
    assert base64.b64decode(result["explanation_png_base64"])[:8] == b"\x89PNG\r\n\x1a\n"

    # The real backend rejecting a bad file must come back as a readable error
    result, error = app.request_prediction(b"junk", "bad.jpg", "image/jpeg")
    assert result is None and "422" in error
