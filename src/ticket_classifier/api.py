"""FastAPI application that serves the trained classifier."""

import logging
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, StringConstraints

from ticket_classifier import __version__
from ticket_classifier.model import Prediction, TicketClassifier

logger = logging.getLogger("ticket_classifier.api")

DEFAULT_MODEL_PATH = "models/model.joblib"
STATIC_DIR = Path(__file__).parent / "static"

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]


class PredictRequest(BaseModel):
    text: QueryText
    top_k: int = Field(default=3, ge=1, le=10)


class BatchPredictRequest(BaseModel):
    texts: list[QueryText] = Field(min_length=1, max_length=100)
    top_k: int = Field(default=3, ge=1, le=10)


class ScoredLabel(BaseModel):
    label: str
    confidence: float


class PredictResponse(BaseModel):
    label: str
    confidence: float
    top_k: list[ScoredLabel]


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ModelInfoResponse(BaseModel):
    n_labels: int
    metadata: dict[str, Any]


def _to_response(prediction: Prediction) -> PredictResponse:
    return PredictResponse(
        label=prediction.label,
        confidence=prediction.confidence,
        top_k=[ScoredLabel(label=lab, confidence=conf) for lab, conf in prediction.top_k],
    )


def get_classifier(request: Request) -> TicketClassifier:
    classifier = request.app.state.classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run `make train` first.")
    return classifier


ClassifierDep = Annotated[TicketClassifier, Depends(get_classifier)]


def create_app(model_path: Path | str | None = None) -> FastAPI:
    """Application factory; the model path can be injected for tests."""
    path = Path(model_path or os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if path.exists():
            app.state.classifier = TicketClassifier.load(path)
            logger.info("Loaded model from %s", path)
        else:
            app.state.classifier = None
            logger.warning("No model found at %s; /predict will return 503", path)
        yield

    app = FastAPI(
        title="Ticket Classifier",
        description="Classifies customer-support queries into intents.",
        version=__version__,
        lifespan=lifespan,
    )

    @app.get("/", include_in_schema=False, response_model=None)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        return HealthResponse(status="ok", model_loaded=request.app.state.classifier is not None)

    @app.get("/model-info", response_model=ModelInfoResponse)
    def model_info(classifier: ClassifierDep) -> ModelInfoResponse:
        return ModelInfoResponse(n_labels=len(classifier.labels), metadata=classifier.metadata)

    # Plain `def` endpoints run in FastAPI's threadpool, so CPU-bound sklearn calls
    # don't block the event loop.
    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, classifier: ClassifierDep) -> PredictResponse:
        [prediction] = classifier.predict([payload.text], top_k=payload.top_k)
        return _to_response(prediction)

    @app.post("/predict/batch", response_model=BatchPredictResponse)
    def predict_batch(
        payload: BatchPredictRequest, classifier: ClassifierDep
    ) -> BatchPredictResponse:
        predictions = classifier.predict(payload.texts, top_k=payload.top_k)
        return BatchPredictResponse(predictions=[_to_response(p) for p in predictions])

    return app


app = create_app()
