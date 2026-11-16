"""Application settings. Every value can be overridden with an environment variable
that starts with SKIN_ (for example SKIN_MAX_UPLOAD_MB=5)."""
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

# backend/ folder (this file lives in backend/app/)
BACKEND_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    # ---- Model files ----
    model_path: Path = BACKEND_DIR / "models" / "best_model.keras"
    class_names_path: Path = BACKEND_DIR / "models" / "class_names.json"

    # ---- Upload limits ----
    max_upload_mb: int = 10
    allowed_content_types: tuple[str, ...] = ("image/jpeg", "image/png")
    allowed_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png")
    # A tiny PNG can decompress into a gigantic image, so the pixel count is capped too
    max_image_pixels: int = 25_000_000
    min_image_side: int = 64

    # ---- Preprocessing ----
    # How the image is shrunk to the model's input size. This MUST match training.
    #   tf_bilinear  = tf.image.resize (what keras image_dataset_from_directory uses)
    #   pil_bilinear = Pillow bilinear with anti-aliasing
    #   pil_nearest  = nearest neighbour (old keras load_img default)
    # Unsure which one was used? Run `python -m evaluation.run_evaluation --compare-resize`
    # on labeled images and pick the one with the best accuracy.
    resize_method: Literal["tf_bilinear", "pil_bilinear", "pil_nearest"] = "tf_bilinear"

    # ---- Image quality warnings (starting values - tune them on real images) ----
    blur_warning_threshold: float = 20.0  # variance of the Laplacian below this = blurry
    dark_warning_threshold: float = 25.0  # mean brightness (0-255) below this = too dark
    bright_warning_threshold: float = 235.0  # mean brightness above this = overexposed
    low_contrast_threshold: float = 10.0  # brightness std-dev below this = almost no detail

    # ---- "Uncertain result" rules (starting values - NOT calibrated yet) ----
    # Calibrate them with `python -m evaluation.run_evaluation` once you have labeled images.
    min_confidence: float = 0.50  # top probability below this = uncertain
    min_margin: float = 0.20  # gap between the top two classes below this = uncertain

    # ---- Security ----
    api_key: str = ""  # if set, every /api request must send it in the X-API-Key header
    rate_limit_per_minute: int = 60  # per client IP; 0 turns the limit off

    # Comma-separated list of frontend origins allowed to call this API (CORS)
    cors_origins: str = "http://localhost:8501"

    # model_path starts with "model_", which pydantic reserves by default, so allow it
    model_config = SettingsConfigDict(env_prefix="SKIN_", protected_namespaces=("settings_",))


settings = Settings()
