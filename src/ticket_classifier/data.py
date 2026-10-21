"""Dataset loading and cleaning for the Banking77 support-query dataset."""

import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data"
SPLIT_URLS = {"train": f"{BASE_URL}/train.csv", "test": f"{BASE_URL}/test.csv"}
REQUIRED_COLUMNS = ("text", "label")


def clean_text(text: str) -> str:
    """Collapse runs of whitespace and strip both ends.

    Casing and punctuation are left alone; the vectorizer handles lowercasing.
    """
    return " ".join(text.split())


def validate_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return a cleaned copy with exactly the columns ``text`` and ``label``.

    Accepts the raw Banking77 layout (``category`` instead of ``label``), drops rows
    with missing or blank values, and raises ``ValueError`` if nothing usable remains.
    """
    if "category" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"category": "label"})

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")

    cleaned = df.loc[:, list(REQUIRED_COLUMNS)].dropna().copy()
    cleaned["text"] = cleaned["text"].astype(str).map(clean_text)
    cleaned["label"] = cleaned["label"].astype(str).str.strip()
    cleaned = cleaned[(cleaned["text"] != "") & (cleaned["label"] != "")]

    if cleaned.empty:
        raise ValueError("Dataset is empty after cleaning")
    return cleaned.reset_index(drop=True)


def _download(url: str, dest: Path, timeout: float = 30.0) -> None:
    """Download ``url`` to ``dest`` atomically (write to a temp file, then rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as fh:
        fh.write(response.read())
    partial.replace(dest)


def load_csv(path: Path | str) -> pd.DataFrame:
    """Load and validate a local CSV with ``text`` and ``label``/``category`` columns."""
    return validate_frame(pd.read_csv(path))


def load_split(split: str, data_dir: Path | str = "data") -> pd.DataFrame:
    """Load the ``train`` or ``test`` split, downloading and caching it on first use."""
    if split not in SPLIT_URLS:
        raise ValueError(f"Unknown split {split!r}; expected one of {sorted(SPLIT_URLS)}")

    path = Path(data_dir) / f"banking77_{split}.csv"
    if not path.exists():
        _download(SPLIT_URLS[split], path)
    return load_csv(path)
