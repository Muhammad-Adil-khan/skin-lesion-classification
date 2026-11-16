# Frontend image: Streamlit only (no TensorFlow). Build context is the frontend/ folder.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY requirements.lock .
RUN pip install --no-cache-dir -r requirements.lock

COPY streamlit_app.py .
COPY .streamlit ./.streamlit

RUN useradd --create-home appuser
USER appuser

# Inside docker-compose the backend is reachable by its service name
ENV SKIN_API_BASE_URL=http://backend:8000

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
