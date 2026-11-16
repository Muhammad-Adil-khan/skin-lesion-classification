"""Security tests: API key, rate limiting, request-size cap and response headers."""
import time

from app import security
from app.config import settings
from helpers import make_image_bytes, upload


# ---------- API key ----------
def test_api_key_not_required_by_default(client):
    assert upload(client, make_image_bytes()).status_code == 200


def test_api_key_required_when_configured(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret-key")
    data = make_image_bytes()

    assert upload(client, data).status_code == 401  # missing
    assert upload(client, data, headers={"X-API-Key": "wrong"}).status_code == 401  # wrong
    assert upload(client, data, headers={"X-API-Key": "secret-key"}).status_code == 200  # right


def test_health_and_root_stay_open_when_api_key_is_set(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret-key")
    assert client.get("/").status_code == 200
    health = client.get("/health").json()
    assert health["auth_required"] is True
    assert "secret-key" not in str(health)  # the key itself is never exposed


def test_401_body_has_the_standard_error_shape(client, monkeypatch):
    monkeypatch.setattr(settings, "api_key", "secret-key")
    body = upload(client, make_image_bytes()).json()
    assert body == {"success": False, "detail": "Invalid or missing API key."}


# ---------- rate limiting ----------
def test_rate_limit_returns_429_with_retry_after(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 3)
    data = make_image_bytes()

    codes = [upload(client, data).status_code for _ in range(5)]
    assert codes == [200, 200, 200, 429, 429]

    blocked = upload(client, data)
    assert int(blocked.headers["retry-after"]) >= 1
    assert blocked.json()["success"] is False
    assert client.get("/health").status_code == 200  # only /api POSTs are limited


def test_rate_limit_is_per_client(monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)

    assert security._seconds_until_allowed("10.0.0.1") is None  # first request: allowed
    assert security._seconds_until_allowed("10.0.0.1") is not None  # second: told how long to wait
    assert security._seconds_until_allowed("10.0.0.2") is None  # another address has its own allowance


def test_rate_limit_window_expires(client, monkeypatch):
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    monkeypatch.setattr(security, "RATE_WINDOW_SECONDS", 1)
    data = make_image_bytes()

    assert upload(client, data).status_code == 200
    assert upload(client, data).status_code == 429
    time.sleep(1.2)
    assert upload(client, data).status_code == 200


def test_failed_key_attempts_count_toward_the_limit(client, monkeypatch):
    """Otherwise an attacker could guess keys without ever being slowed down."""
    monkeypatch.setattr(settings, "api_key", "secret-key")
    monkeypatch.setattr(settings, "rate_limit_per_minute", 2)
    codes = [upload(client, b"x", headers={"X-API-Key": f"guess{i}"}).status_code for i in range(4)]
    assert codes == [401, 401, 429, 429]


def test_rate_limit_can_be_turned_off(client):
    data = make_image_bytes()
    assert all(upload(client, data).status_code == 200 for _ in range(5))  # limit is 0 in tests


# ---------- request size ----------
def test_huge_request_is_rejected_before_it_is_parsed(client, monkeypatch):
    monkeypatch.setattr(settings, "max_upload_mb", 1)  # allowed: 1 MB file + 1 MB multipart overhead
    response = upload(client, b"0" * (3 * 1024 * 1024), "big.jpg", "image/jpeg")
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()


# ---------- headers ----------
def test_api_responses_are_not_cacheable(client, monkeypatch):
    assert upload(client, make_image_bytes()).headers["cache-control"] == "no-store"  # success
    assert upload(client, b"nope", "x.txt", "text/plain").headers["cache-control"] == "no-store"  # route error

    # Errors returned early by the security layer must carry the header too
    monkeypatch.setattr(settings, "api_key", "k")
    assert upload(client, make_image_bytes()).headers["cache-control"] == "no-store"  # 401
    monkeypatch.setattr(settings, "api_key", "")
    monkeypatch.setattr(settings, "max_upload_mb", 1)
    assert upload(client, b"0" * (3 * 1024 * 1024)).headers["cache-control"] == "no-store"  # 413
    monkeypatch.setattr(settings, "max_upload_mb", 10)
    monkeypatch.setattr(settings, "rate_limit_per_minute", 1)
    upload(client, make_image_bytes())
    blocked = upload(client, make_image_bytes())
    assert blocked.status_code == 429 and blocked.headers["cache-control"] == "no-store"  # 429
