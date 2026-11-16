"""Response shapes returned by the API."""
from pydantic import BaseModel, Field


class PredictionResponse(BaseModel):
    success: bool = True
    predicted_class: str = Field(examples=["MEL"])
    confidence: float = Field(ge=0, le=1, examples=[0.94])
    predictions: dict[str, float] = Field(
        examples=[{"NV": 0.02, "MEL": 0.94, "BCC": 0.03, "BKL": 0.01}]
    )
    # True when the model's output looks unreliable (low confidence or two close classes)
    is_uncertain: bool = False
    uncertainty_reasons: list[str] = []
    # Image-quality problems and other notes the user should know about
    warnings: list[str] = []
    # Base64 PNG of a Grad-CAM heatmap. Only filled when ?explain=true is requested.
    explanation_png_base64: str | None = None
    # First 12 characters of the model file's SHA-256, so every result can be traced to a model
    model_version: str
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    classes: list[str] = []
    input_size: list[int] = []
    model_sha256: str = ""
    explain_available: bool = False
    resize_method: str = ""
    auth_required: bool = False


class ErrorResponse(BaseModel):
    success: bool = False
    detail: str
