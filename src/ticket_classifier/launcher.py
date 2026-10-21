"""One-command launcher: train a model if needed, start the server, open the browser."""

import argparse
import threading
import time
import urllib.request
import webbrowser
from pathlib import Path

import uvicorn

from ticket_classifier import train
from ticket_classifier.api import create_app


def _open_browser_when_ready(url: str, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"{url}/health", timeout=1)
            break
        except OSError:
            time.sleep(0.3)
    webbrowser.open(url)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model-path", type=Path, default=Path("models/model.joblib"))
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args(argv)

    if not args.model_path.exists():
        print("No trained model found. Training one now; this downloads the dataset and can")
        print("take a few minutes. It only happens the first time.")
        try:
            train.main(["--models-dir", str(args.model_path.parent)])
        except OSError as exc:
            print(f"Could not download the dataset ({exc}). Check your internet connection.")
            return 1

    url = f"http://{args.host}:{args.port}"
    if not args.no_browser:
        threading.Thread(target=_open_browser_when_ready, args=(url,), daemon=True).start()
    print(f"Running at {url}  (press Ctrl+C to stop)")
    uvicorn.run(create_app(args.model_path), host=args.host, port=args.port)
    return 0
