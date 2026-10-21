import pytest

from ticket_classifier.model import TicketClassifier, build_improved


def test_predict_obvious_examples(trained_classifier):
    lost, rate = trained_classifier.predict(["my card was stolen", "what exchange rate do you use"])
    assert lost.label == "lost_or_stolen_card"
    assert rate.label == "exchange_rate"


def test_top_k_is_sorted_and_valid_probabilities(trained_classifier):
    [prediction] = trained_classifier.predict(["where is my card"], top_k=3)
    confidences = [conf for _, conf in prediction.top_k]

    assert confidences == sorted(confidences, reverse=True)
    assert prediction.label == prediction.top_k[0][0]
    assert prediction.confidence == confidences[0]
    assert 0.0 <= sum(confidences) <= 1.0 + 1e-9


def test_top_k_is_capped_by_number_of_classes(trained_classifier):
    [prediction] = trained_classifier.predict(["lost card"], top_k=10)
    assert len(prediction.top_k) == 3


def test_empty_input_returns_empty_list(trained_classifier):
    assert trained_classifier.predict([]) == []


def test_invalid_top_k_raises(trained_classifier):
    with pytest.raises(ValueError):
        trained_classifier.predict(["hello"], top_k=0)


def test_save_and_load_roundtrip(trained_classifier, model_path):
    loaded = TicketClassifier.load(model_path)
    queries = ["my card was stolen", "what is the rate for euros"]

    assert loaded.predict(queries) == trained_classifier.predict(queries)
    assert loaded.metadata["model_name"] == "baseline"
    assert "sklearn_version" in loaded.metadata


def test_improved_pipeline_fits_and_predicts(tiny_frame):
    pipeline = build_improved().fit(tiny_frame["text"].tolist(), tiny_frame["label"].tolist())
    classifier = TicketClassifier(pipeline)
    [prediction] = classifier.predict(["my card was stolen"])
    assert prediction.label == "lost_or_stolen_card"
