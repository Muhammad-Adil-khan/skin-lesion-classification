"""Turn uploaded image bytes into the tensor the model expects, plus simple quality checks."""
import io
from dataclasses import dataclass

import numpy as np
from PIL import Image, UnidentifiedImageError

from app.config import settings


class InvalidImageError(Exception):
    """Raised when the uploaded bytes are not a usable image."""


@dataclass
class PreparedImage:
    tensor: np.ndarray  # shape (1, height, width, 3), float32, values 0-255
    image: Image.Image  # the decoded RGB image at its original size (used for the heatmap)
    quality_warnings: list[str]


def open_image(image_bytes: bytes) -> Image.Image:
    """Decode the bytes into an RGB image. Rejects unreadable, tiny and huge images."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size  # read from the file header - nothing is decoded yet
        if width * height > settings.max_image_pixels:
            raise InvalidImageError(
                f"Image dimensions are too large ({width}x{height}). "
                f"The limit is {settings.max_image_pixels:,} pixels."
            )
        image.load()  # decode fully so corrupted or truncated files fail here
    except InvalidImageError:
        raise
    except Image.DecompressionBombError as error:
        raise InvalidImageError("Image dimensions are too large.") from error
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as error:
        raise InvalidImageError("The file is not a valid or readable image.") from error

    if min(width, height) < settings.min_image_side:
        raise InvalidImageError(
            f"Image is too small ({width}x{height}). Each side must be at least "
            f"{settings.min_image_side} pixels."
        )
    return image.convert("RGB")


def resize_to_model_input(image: Image.Image, input_size: tuple[int, int]) -> np.ndarray:
    """Resize an RGB image to (height, width, 3) float32 with values still in 0-255."""
    height, width = input_size

    if settings.resize_method == "tf_bilinear":
        import tensorflow as tf  # imported here so this module stays light to import

        pixels = np.asarray(image, dtype=np.float32)
        return tf.image.resize(pixels, (height, width), method="bilinear", antialias=False).numpy()

    resample = Image.BILINEAR if settings.resize_method == "pil_bilinear" else Image.NEAREST
    return np.asarray(image.resize((width, height), resample), dtype=np.float32)  # PIL wants (w, h)


def check_image_quality(pixels: np.ndarray) -> list[str]:
    """Return human-readable warnings about the image the model will actually see."""
    gray = pixels.mean(axis=2)
    warnings = []

    # Variance of the Laplacian is a common sharpness measure: blurry images score low
    laplacian = (
        -4 * gray[1:-1, 1:-1] + gray[:-2, 1:-1] + gray[2:, 1:-1] + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    if laplacian.var() < settings.blur_warning_threshold:
        warnings.append("The image looks blurry or out of focus.")

    mean_brightness, contrast = gray.mean(), gray.std()
    if mean_brightness < settings.dark_warning_threshold:
        warnings.append("The image is very dark.")
    elif mean_brightness > settings.bright_warning_threshold:
        warnings.append("The image is very bright (overexposed).")
    if contrast < settings.low_contrast_threshold:
        warnings.append("The image has almost no contrast or detail.")
    return warnings


def prepare_image(image_bytes: bytes, input_size: tuple[int, int]) -> PreparedImage:
    """Full pipeline: decode -> RGB -> resize -> quality checks -> add batch dimension.

    The saved model already contains its own Rescaling(1/255) and Normalization
    (ImageNet mean/std) layers, so we must NOT scale or normalize here again.
    """
    image = open_image(image_bytes)
    pixels = resize_to_model_input(image, input_size)
    warnings = check_image_quality(pixels)
    return PreparedImage(np.expand_dims(pixels, axis=0), image, warnings)
