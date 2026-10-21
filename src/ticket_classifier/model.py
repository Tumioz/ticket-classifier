"""Model definitions and the inference wrapper."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from ticket_classifier.data import clean_text

logger = logging.getLogger(__name__)

SEED = 42


def build_baseline(seed: int = SEED) -> Pipeline:
    """Simple, fast reference model: word TF-IDF + logistic regression."""
    return Pipeline(
        [
            ("tfidf", TfidfVectorizer(lowercase=True)),
            ("clf", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )


def build_improved(seed: int = SEED) -> Pipeline:
    """Word 1-2 grams plus character n-grams, which helps with typos and word variants.

    Hyperparameters were chosen a priori, not tuned on the test set.
    """
    features = FeatureUnion(
        [
            ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("features", features),
            ("clf", LogisticRegression(C=20.0, max_iter=2000, random_state=seed)),
        ]
    )


@dataclass(frozen=True)
class Prediction:
    """Top prediction plus the ranked top-k ``(label, confidence)`` pairs."""

    label: str
    confidence: float
    top_k: tuple[tuple[str, float], ...]


class TicketClassifier:
    """A fitted pipeline together with metadata, with save/load and a tidy predict API."""

    def __init__(self, pipeline: Pipeline, metadata: dict[str, Any] | None = None) -> None:
        self.pipeline = pipeline
        self.metadata: dict[str, Any] = dict(metadata or {})

    @property
    def labels(self) -> list[str]:
        return [str(label) for label in self.pipeline.classes_]

    def predict(self, texts: Sequence[str], top_k: int = 3) -> list[Prediction]:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        if len(texts) == 0:
            return []

        proba = self.pipeline.predict_proba([clean_text(text) for text in texts])
        classes = self.labels
        k = min(top_k, len(classes))
        order = np.argsort(-proba, axis=1, kind="stable")[:, :k]

        predictions = []
        for row, indices in zip(proba, order, strict=True):
            ranked = tuple((classes[i], float(row[i])) for i in indices)
            predictions.append(
                Prediction(label=ranked[0][0], confidence=ranked[0][1], top_k=ranked)
            )
        return predictions

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = {**self.metadata, "sklearn_version": sklearn.__version__}
        joblib.dump({"pipeline": self.pipeline, "metadata": metadata}, path)

    @classmethod
    def load(cls, path: Path | str) -> "TicketClassifier":
        """Load a saved classifier.

        joblib uses pickle under the hood, so only load files you trained yourself.
        """
        bundle = joblib.load(path)
        metadata = bundle.get("metadata", {})
        trained_with = metadata.get("sklearn_version")
        if trained_with and trained_with != sklearn.__version__:
            logger.warning(
                "Model trained with scikit-learn %s but running %s; predictions may differ.",
                trained_with,
                sklearn.__version__,
            )
        return cls(bundle["pipeline"], metadata)
