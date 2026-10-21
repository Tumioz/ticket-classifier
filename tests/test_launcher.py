from ticket_classifier import launcher


def test_launcher_trains_first_when_model_is_missing(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(launcher.train, "main", lambda argv: calls.append("train") or 0)
    monkeypatch.setattr(launcher.uvicorn, "run", lambda app, **kwargs: calls.append("serve"))

    launcher.main(["--no-browser", "--model-path", str(tmp_path / "models" / "model.joblib")])

    assert calls == ["train", "serve"]


def test_launcher_skips_training_when_model_exists(tmp_path, monkeypatch):
    model = tmp_path / "model.joblib"
    model.write_bytes(b"")
    calls = []
    monkeypatch.setattr(launcher.train, "main", lambda argv: calls.append("train") or 0)
    monkeypatch.setattr(launcher.uvicorn, "run", lambda app, **kwargs: calls.append("serve"))

    launcher.main(["--no-browser", "--model-path", str(model)])

    assert calls == ["serve"]
