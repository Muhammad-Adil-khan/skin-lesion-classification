"""Small helpers shared by the test files."""
import io

import numpy as np
from PIL import Image


def make_image_bytes(fmt="JPEG", size=(300, 200), mode="RGB", seed=None) -> bytes:
    """A random-noise image. `size` is (width, height)."""
    rng = np.random.default_rng(seed)
    pixels = rng.integers(0, 255, (size[1], size[0], 3), dtype=np.uint8)
    buffer = io.BytesIO()
    Image.fromarray(pixels).convert(mode).save(buffer, format=fmt)
    return buffer.getvalue()


def solid_image_bytes(color, size=(300, 200), fmt="PNG") -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(client, data: bytes, filename="lesion.jpg", content_type="image/jpeg", params=None, headers=None):
    """POST an image to the prediction endpoint."""
    return client.post(
        "/api/v1/predict",
        files={"file": (filename, data, content_type)},
        params=params,
        headers=headers,
    )
