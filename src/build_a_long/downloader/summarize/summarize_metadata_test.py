"""Tests for summarize_metadata."""

import json
import tempfile
from pathlib import Path

import pytest

from .summarize_metadata import summarize_metadata


@pytest.fixture
def temp_data_dir():
    """Create a temporary directory with some mock metadata files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)
        # Set 1 (2023)
        (data_dir / "set1").mkdir()
        with open(data_dir / "set1" / "metadata.json", "w") as f:
            json.dump(
                {
                    "set_id": "set1",
                    "year": 2023,
                },
                f,
            )

        # Set 2 (2023)
        (data_dir / "set2").mkdir()
        with open(data_dir / "set2" / "metadata.json", "w") as f:
            json.dump(
                {
                    "set_id": "set2",
                    "year": 2023,
                },
                f,
            )

        # Set 3 (2024)
        (data_dir / "set3").mkdir()
        with open(data_dir / "set3" / "metadata.json", "w") as f:
            json.dump(
                {
                    "set_id": "set3",
                    "year": 2024,
                },
                f,
            )

        # Corrupted JSON
        (data_dir / "set4").mkdir()
        with open(data_dir / "set4" / "metadata.json", "w") as f:
            f.write("{corrupted")

        yield data_dir


def test_summarize_metadata_success(temp_data_dir):
    """Test that summarize_metadata successfully creates index files."""
    output_dir = temp_data_dir / "indices"
    result = summarize_metadata(temp_data_dir, output_dir)

    assert result == 0
    assert (output_dir / "index-2023.json").exists()
    assert (output_dir / "index-2024.json").exists()
    assert (output_dir / "index.json").exists()

    with open(output_dir / "index-2023.json") as f:
        data_2023 = json.load(f)
        assert len(data_2023) == 2
        assert [d["set_id"] for d in data_2023] == ["set1", "set2"]

    with open(output_dir / "index-2024.json") as f:
        data_2024 = json.load(f)
        assert len(data_2024) == 1
        assert data_2024[0]["set_id"] == "set3"

    with open(output_dir / "index.json") as f:
        index_data = json.load(f)
        assert len(index_data) == 2
        summary_2023 = next(d for d in index_data if d["year"] == 2023)
        summary_2024 = next(d for d in index_data if d["year"] == 2024)

        assert summary_2023["count"] == 2
        assert summary_2023["filename"] == "index-2023.json"
        assert summary_2023["filesize"] > 0

        assert summary_2024["count"] == 1
        assert summary_2024["filename"] == "index-2024.json"
        assert summary_2024["filesize"] > 0


def test_summarize_metadata_no_files(tmp_path):
    """Test summarize_metadata when no metadata files are found."""
    output_dir = tmp_path / "indices"
    result = summarize_metadata(tmp_path, output_dir)
    assert result == 1


def test_summarize_metadata_empty_dir(tmp_path):
    """Test summarize_metadata with an empty data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "indices"
    result = summarize_metadata(data_dir, output_dir)
    assert result == 1
