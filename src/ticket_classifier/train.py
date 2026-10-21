"""Train both candidate models, compare them on the test set, and save one for serving."""

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from ticket_classifier import __version__
from ticket_classifier.data import load_csv, load_split
from ticket_classifier.model import SEED, TicketClassifier, build_baseline, build_improved

CANDIDATES: dict[str, Callable[[int], Pipeline]] = {
    "baseline": build_baseline,
    "improved": build_improved,
}


@dataclass
class Experiment:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]
    train_seconds: float


def evaluate(pipeline: Pipeline, texts: list[str], labels: list[str]) -> dict[str, float]:
    predicted = pipeline.predict(texts)
    return {
        "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(f1_score(labels, predicted, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
    }


def run_experiments(
    train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int = SEED
) -> list[Experiment]:
    """Fit every candidate on the training split and score it on the test split."""
    x_train, y_train = train_df["text"].tolist(), train_df["label"].tolist()
    x_test, y_test = test_df["text"].tolist(), test_df["label"].tolist()

    experiments = []
    for name, build in CANDIDATES.items():
        pipeline = build(seed)
        started = time.perf_counter()
        pipeline.fit(x_train, y_train)
        elapsed = time.perf_counter() - started
        metrics = evaluate(pipeline, x_test, y_test)
        experiments.append(Experiment(name, pipeline, metrics, elapsed))
    return experiments


def format_table(experiments: list[Experiment]) -> str:
    lines = [f"{'model':<10} {'accuracy':>9} {'macro_f1':>9} {'weighted_f1':>12} {'train_s':>8}"]
    for exp in experiments:
        m = exp.metrics
        lines.append(
            f"{exp.name:<10} {m['accuracy']:>9.4f} {m['macro_f1']:>9.4f} "
            f"{m['weighted_f1']:>12.4f} {exp.train_seconds:>8.1f}"
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="download cache")
    parser.add_argument("--train-csv", type=Path, help="use a local training CSV instead")
    parser.add_argument("--test-csv", type=Path, help="use a local test CSV instead")
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument("--reports-dir", type=Path, default=Path("reports"))
    parser.add_argument("--save", choices=sorted(CANDIDATES), default="improved")
    parser.add_argument("--max-rows", type=int, help="subsample the training set (quick runs)")
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    train_df = load_csv(args.train_csv) if args.train_csv else load_split("train", args.data_dir)
    test_df = load_csv(args.test_csv) if args.test_csv else load_split("test", args.data_dir)
    if args.max_rows and args.max_rows < len(train_df):
        train_df = train_df.sample(n=args.max_rows, random_state=args.seed).reset_index(drop=True)

    print(f"train rows: {len(train_df)}  test rows: {len(test_df)}")
    print(f"labels: {train_df['label'].nunique()}")

    experiments = run_experiments(train_df, test_df, seed=args.seed)
    print(format_table(experiments))

    chosen = next(exp for exp in experiments if exp.name == args.save)
    metadata = {
        "model_name": chosen.name,
        "package_version": __version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "seed": args.seed,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "n_labels": int(train_df["label"].nunique()),
        "test_metrics": chosen.metrics,
    }
    model_path = args.models_dir / "model.joblib"
    TicketClassifier(chosen.pipeline, metadata).save(model_path)

    args.reports_dir.mkdir(parents=True, exist_ok=True)
    results = {
        "saved_model": chosen.name,
        "experiments": {
            exp.name: {**exp.metrics, "train_seconds": round(exp.train_seconds, 2)}
            for exp in experiments
        },
    }
    (args.reports_dir / "metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    report = classification_report(
        test_df["label"].tolist(),
        chosen.pipeline.predict(test_df["text"].tolist()),
        zero_division=0,
    )
    (args.reports_dir / "classification_report.txt").write_text(report)

    print(f"saved '{chosen.name}' to {model_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
