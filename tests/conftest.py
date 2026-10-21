import pandas as pd
import pytest

from ticket_classifier.model import TicketClassifier, build_baseline

SAMPLES = {
    "card_arrival": [
        "where is my new card",
        "my card has not arrived yet",
        "how long until my card arrives",
        "I am still waiting for my card to be delivered",
        "when will my new card get here",
        "my replacement card never showed up",
        "tracking my card delivery",
        "has my card been posted",
    ],
    "lost_or_stolen_card": [
        "I lost my card",
        "my card was stolen",
        "someone stole my wallet with my card",
        "I cannot find my card anywhere",
        "my card is missing please block it",
        "I think my card was stolen yesterday",
        "lost my debit card on the train",
        "report a stolen card",
    ],
    "exchange_rate": [
        "what exchange rate do you use",
        "what is the rate for euros",
        "how is the currency exchange rate calculated",
        "which exchange rate applies to dollars",
        "tell me the current exchange rate",
        "what rate will I get converting pounds",
        "is the exchange rate fair",
        "how do you set your exchange rates",
    ],
}


@pytest.fixture
def tiny_frame() -> pd.DataFrame:
    rows = [(text, label) for label, texts in SAMPLES.items() for text in texts]
    return pd.DataFrame(rows, columns=["text", "label"])


@pytest.fixture
def trained_classifier(tiny_frame: pd.DataFrame) -> TicketClassifier:
    pipeline = build_baseline().fit(tiny_frame["text"].tolist(), tiny_frame["label"].tolist())
    return TicketClassifier(pipeline, metadata={"model_name": "baseline"})


@pytest.fixture
def model_path(trained_classifier: TicketClassifier, tmp_path):
    path = tmp_path / "model.joblib"
    trained_classifier.save(path)
    return path
