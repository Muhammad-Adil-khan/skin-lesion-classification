"""Turn a Grad-CAM heatmap into a picture that can be sent to the frontend."""
import base64
import io

import numpy as np
from PIL import Image

MAX_DISPLAY_SIDE = 512  # keep the response small


def _jet_colormap(values: np.ndarray) -> np.ndarray:
    """Map values in 0-1 to RGB (blue = low, red = high). Returns uint8 of shape (..., 3)."""
    red = np.clip(1.5 - np.abs(4 * values - 3), 0, 1)
    green = np.clip(1.5 - np.abs(4 * values - 2), 0, 1)
    blue = np.clip(1.5 - np.abs(4 * values - 1), 0, 1)
    return (np.stack([red, green, blue], axis=-1) * 255).astype(np.uint8)


def heatmap_overlay_base64(image: Image.Image, heatmap: np.ndarray, alpha: float = 0.4) -> str:
    """Blend the heatmap (2-D array, values 0-1) over the image and return a base64 PNG."""
    base = image.convert("RGB")
    base.thumbnail((MAX_DISPLAY_SIDE, MAX_DISPLAY_SIDE))  # shrinks in place, keeps aspect ratio

    heat_small = Image.fromarray((np.clip(heatmap, 0, 1) * 255).astype(np.uint8))
    heat = np.asarray(heat_small.resize(base.size, Image.BILINEAR), dtype=np.float32) / 255.0

    colored = _jet_colormap(heat).astype(np.float32)
    blended = (1 - alpha) * np.asarray(base, dtype=np.float32) + alpha * colored

    buffer = io.BytesIO()
    Image.fromarray(blended.astype(np.uint8)).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
