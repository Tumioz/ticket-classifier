import pandas as pd
import pytest

from ticket_classifier import data


def test_clean_text_collapses_whitespace():
    assert data.clean_text("  my   card\tis \n lost  ") == "my card is lost"


def test_validate_frame_renames_category_column():
    raw = pd.DataFrame({"text": ["hello"], "category": ["greeting"]})
    assert list(data.validate_frame(raw).columns) == ["text", "label"]


def test_validate_frame_rejects_missing_columns():
    with pytest.raises(ValueError, match="Missing required column"):
        data.validate_frame(pd.DataFrame({"text": ["hello"]}))


def test_validate_frame_drops_blank_and_null_rows():
    raw = pd.DataFrame(
        {"text": ["ok query", "   ", None, "another"], "label": ["a", "a", "b", None]}
    )
    cleaned = data.validate_frame(raw)
    assert cleaned["text"].tolist() == ["ok query"]


def test_validate_frame_raises_when_nothing_left():
    with pytest.raises(ValueError, match="empty"):
        data.validate_frame(pd.DataFrame({"text": ["  "], "label": ["a"]}))


def test_load_split_rejects_unknown_split(tmp_path):
    with pytest.raises(ValueError, match="Unknown split"):
        data.load_split("validation", tmp_path)


def test_load_split_uses_cache_without_downloading(tmp_path, monkeypatch):
    pd.DataFrame({"text": ["hi there"], "category": ["greeting"]}).to_csv(
        tmp_path / "banking77_train.csv", index=False
    )

    def fail(*args, **kwargs):
        raise AssertionError("should not download when the cache file exists")

    monkeypatch.setattr(data, "_download", fail)
    assert data.load_split("train", tmp_path)["label"].tolist() == ["greeting"]


def test_load_split_downloads_when_cache_is_missing(tmp_path, monkeypatch):
    calls = []

    def fake_download(url, dest, timeout=30.0):
        calls.append(url)
        pd.DataFrame({"text": ["hi"], "category": ["greeting"]}).to_csv(dest, index=False)

    monkeypatch.setattr(data, "_download", fake_download)
    frame = data.load_split("test", tmp_path)

    assert calls == [data.SPLIT_URLS["test"]]
    assert len(frame) == 1
