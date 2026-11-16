"""Prediction endpoint: validates the upload and calls the prediction service."""
import logging
import os

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool

from app.config import settings
from app.schemas.prediction import ErrorResponse, PredictionResponse
from app.services.prediction_service import ModelLoadError
from app.utils.preprocessing import InvalidImageError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Prediction"])


async def read_validated_upload(file: UploadFile) -> bytes:
    """Check type and size, then return the raw bytes."""
    extension = os.path.splitext(file.filename or "")[1].lower()
    if file.content_type not in settings.allowed_content_types or extension not in settings.allowed_extensions:
        raise HTTPException(status_code=415, detail="Unsupported file type. Please upload a JPG, JPEG or PNG image.")

    # Read one byte more than the limit so we can detect oversized files without loading them all
    max_bytes = settings.max_upload_mb * 1024 * 1024
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {settings.max_upload_mb} MB.")
    if len(data) == 0:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return data


@router.post(
    "/predict",
    response_model=PredictionResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse},
               413: {"model": ErrorResponse}, 415: {"model": ErrorResponse},
               422: {"model": ErrorResponse}, 429: {"model": ErrorResponse},
               503: {"model": ErrorResponse}},
)
async def predict(request: Request, file: UploadFile, explain: bool = False):
    """Classify an uploaded dermoscopic image.

    Set `explain=true` to also receive a Grad-CAM heatmap (base64 PNG) showing which image
    regions most influenced the result. The heatmap is not a medical finding.
    """
    service = request.app.state.prediction_service
    if not service.is_ready:
        raise HTTPException(status_code=503, detail="Model is not available. Please try again later.")

    image_bytes = await read_validated_upload(file)
    try:
        # Model inference is CPU heavy, so run it off the event loop
        return await run_in_threadpool(service.predict, image_bytes, explain)
    except InvalidImageError as error:
        raise HTTPException(status_code=422, detail=str(error))
    except ModelLoadError as error:
        raise HTTPException(status_code=503, detail=str(error))
    except Exception:
        logger.exception("Unexpected error during prediction")
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error.")
