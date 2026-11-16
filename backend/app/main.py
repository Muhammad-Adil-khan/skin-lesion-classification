"""FastAPI application entry point."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.routes import prediction
from app.schemas.prediction import HealthResponse
from app.security import security_middleware
from app.services.prediction_service import ModelLoadError, PredictionService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts."""
    service = PredictionService(settings.model_path, settings.class_names_path)
    try:
        service.load()
    except ModelLoadError as error:
        # Keep the API running so /health can report the problem instead of crashing
        logger.error("Model failed to load: %s", error)
    app.state.prediction_service = service
    yield


app = FastAPI(
    title="Skin Disease Detection API",
    description="Educational decision-support prototype. Not a clinically validated diagnostic tool.",
    version="1.0.0",
    lifespan=lifespan,
)

# Added before CORS so that CORS is the outermost layer and also covers error responses
app.middleware("http")(security_middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in settings.cors_origins.split(",")],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(prediction.router)


@app.get("/")
def root():
    return {"message": "Welcome to the Skin Disease Detection API. See /docs for usage."}


@app.get("/health", response_model=HealthResponse)
def health():
    service: PredictionService = app.state.prediction_service
    return HealthResponse(
        status="ok" if service.is_ready else "degraded",
        model_loaded=service.is_ready,
        classes=service.class_names,
        input_size=list(service.input_size) if service.is_ready else [],
        model_sha256=service.model_sha256,
        explain_available=service.explain_available,
        resize_method=settings.resize_method,
        auth_required=bool(settings.api_key),
    )
