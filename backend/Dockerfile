# Backend image: FastAPI + TensorFlow (CPU). Build context is the backend/ folder.
FROM python:3.12-slim

# Unbuffered logs, no .pyc files, quieter TensorFlow
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Install dependencies first so Docker can cache this slow layer.
# requirements.lock pins every package (linux x86_64). On other platforms use requirements.txt.
COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY app ./app
COPY models ./models

# Run as a normal user instead of root
RUN useradd --create-home appuser
USER appuser

EXPOSE 8000

# Healthy only when the model is actually loaded (loading takes a few seconds)
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import json,urllib.request; d=json.load(urllib.request.urlopen('http://localhost:8000/health')); raise SystemExit(0 if d['model_loaded'] else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
