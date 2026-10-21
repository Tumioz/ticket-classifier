FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/models/model.joblib

WORKDIR /app

# Install dependencies first so this layer is cached when only the model changes.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# Train locally first (`make train`); the trained model is baked into the image.
COPY models ./models

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "ticket_classifier.api:app", "--host", "0.0.0.0", "--port", "8000"]
