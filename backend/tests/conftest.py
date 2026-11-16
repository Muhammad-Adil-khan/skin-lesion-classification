"""Shared test setup.

* The real trained model is used when it exists.
* If it does not (for example in CI, where the 57 MB file is not checked out), a tiny stand-in
  model with the same input/output shape is built, so the API logic can still be tested.
  Tests that need the real model use the `real_model` fixture and are skipped in that case.
"""
import pytest
from fastapi.testclient import TestClient

from app.config import settings
from app.main import app
from app.security import reset_rate_limiter

REAL_MODEL_PRESENT = settings.model_path.is_file()


@pytest.fixture(scope="session", autouse=True)
def model_in_use(tmp_path_factory):
    if REAL_MODEL_PRESENT:
        yield "real"
        return

    import keras

    inputs = keras.Input((224, 224, 3))
    x = keras.layers.Rescaling(1 / 255)(inputs)
    x = keras.layers.Conv2D(8, 3, strides=4, activation="relu", name="stand_in_conv")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(4, activation="softmax")(x)
    path = tmp_path_factory.mktemp("stand_in_model") / "model.keras"
    keras.Model(inputs, outputs).save(path)

    original = settings.model_path
    settings.model_path = path
    yield "stand-in"
    settings.model_path = original


@pytest.fixture(scope="session")
def client(model_in_use):
    # "with" runs the app's lifespan, so the model is loaded once for the whole test session
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def real_model(model_in_use):
    if model_in_use != "real":
        pytest.skip("needs the real trained model")


@pytest.fixture(autouse=True)
def clean_security_state(monkeypatch):
    """Every test starts with no API key, no rate limit and an empty rate-limit table."""
    reset_rate_limiter()
    monkeypatch.setattr(settings, "rate_limit_per_minute", 0)
    monkeypatch.setattr(settings, "api_key", "")
    yield
    reset_rate_limiter()
