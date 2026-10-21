from fastapi.testclient import TestClient

from ticket_classifier.api import create_app


def test_health_reports_model_loaded(model_path):
    with TestClient(create_app(model_path)) as client:
        response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": True}


def test_predict_returns_label_and_ranked_top_k(model_path):
    with TestClient(create_app(model_path)) as client:
        response = client.post("/predict", json={"text": "my card was stolen", "top_k": 2})

    body = response.json()
    assert response.status_code == 200
    assert body["label"] == "lost_or_stolen_card"
    assert len(body["top_k"]) == 2
    assert body["top_k"][0]["label"] == body["label"]


def test_predict_rejects_blank_text(model_path):
    with TestClient(create_app(model_path)) as client:
        response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422


def test_predict_rejects_out_of_range_top_k(model_path):
    with TestClient(create_app(model_path)) as client:
        response = client.post("/predict", json={"text": "hello", "top_k": 0})
    assert response.status_code == 422


def test_batch_predict(model_path):
    payload = {"texts": ["my card was stolen", "what exchange rate do you use"]}
    with TestClient(create_app(model_path)) as client:
        response = client.post("/predict/batch", json=payload)

    labels = [p["label"] for p in response.json()["predictions"]]
    assert response.status_code == 200
    assert labels == ["lost_or_stolen_card", "exchange_rate"]


def test_batch_rejects_empty_list(model_path):
    with TestClient(create_app(model_path)) as client:
        response = client.post("/predict/batch", json={"texts": []})
    assert response.status_code == 422


def test_model_info(model_path):
    with TestClient(create_app(model_path)) as client:
        response = client.get("/model-info")
    assert response.status_code == 200
    assert response.json()["n_labels"] == 3
    assert response.json()["metadata"]["model_name"] == "baseline"


def test_predict_returns_503_when_model_missing(tmp_path):
    with TestClient(create_app(tmp_path / "missing.joblib")) as client:
        assert client.get("/health").json()["model_loaded"] is False
        response = client.post("/predict", json={"text": "hello"})
    assert response.status_code == 503


def test_index_serves_web_ui(model_path):
    with TestClient(create_app(model_path)) as client:
        response = client.get("/")
    assert response.status_code == 200
    assert "Ticket Classifier" in response.text
