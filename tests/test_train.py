import json

from ticket_classifier import train


def test_run_experiments_scores_every_candidate(tiny_frame):
    experiments = train.run_experiments(tiny_frame, tiny_frame)

    assert {exp.name for exp in experiments} == set(train.CANDIDATES)
    for exp in experiments:
        assert set(exp.metrics) == {"accuracy", "macro_f1", "weighted_f1"}
        assert 0.0 <= exp.metrics["accuracy"] <= 1.0


def test_main_writes_model_and_reports(tiny_frame, tmp_path):
    csv_path = tmp_path / "tiny.csv"
    tiny_frame.rename(columns={"label": "category"}).to_csv(csv_path, index=False)

    exit_code = train.main(
        [
            "--train-csv", str(csv_path),
            "--test-csv", str(csv_path),
            "--models-dir", str(tmp_path / "models"),
            "--reports-dir", str(tmp_path / "reports"),
            "--save", "baseline",
        ]
    )

    assert exit_code == 0
    assert (tmp_path / "models" / "model.joblib").exists()
    metrics = json.loads((tmp_path / "reports" / "metrics.json").read_text())
    assert metrics["saved_model"] == "baseline"
    assert set(metrics["experiments"]) == {"baseline", "improved"}
    assert (tmp_path / "reports" / "classification_report.txt").exists()
