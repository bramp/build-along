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
                    "set": "set1",
                    "locale": "en-us",
                    "year": 2023,
                },
                f,
            )

        # Set 2 (2023)
        (data_dir / "set2").mkdir()
        with open(data_dir / "set2" / "metadata.json", "w") as f:
            json.dump(
                {
                    "set": "set2",
                    "locale": "en-us",
                    "year": 2023,
                },
                f,
            )

        # Set 3 (2024)
        (data_dir / "set3").mkdir()
        with open(data_dir / "set3" / "metadata.json", "w") as f:
            json.dump(
                {
                    "set": "set3",
                    "locale": "en-us",
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
        assert [d["set"] for d in data_2023] == ["set1", "set2"]

    with open(output_dir / "index-2024.json") as f:
        data_2024 = json.load(f)
        assert len(data_2024) == 1
        assert data_2024[0]["set"] == "set3"

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


def test_summarize_metadata_with_urls(tmp_path):
    """Test that summarize_metadata correctly serializes URLs and other Pydantic types."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "indices"

    # Create metadata with URLs (AnyUrl) and other Pydantic types
    (data_dir / "10255").mkdir()
    with open(data_dir / "10255" / "metadata.json", "w") as f:
        json.dump(
            {
                "set": "10255",
                "locale": "en-us",
                "year": 2016,
                "name": "Assembly Square",
                "theme": "LEGO® Creator Expert",
                "age": "16+",
                "pieces": 4002,
                "set_image_url": "https://www.lego.com/cdn/cs/set/assets/image.jpg",
                "pdfs": [
                    {
                        "url": "https://www.lego.com/cdn/product-assets/instructions.pdf",
                        "filename": "10255_instructions.pdf",
                        "preview_url": "https://www.lego.com/cdn/preview.jpg",
                        "filesize": 1024000,
                        "filehash": "abc123",
                    }
                ],
            },
            f,
        )

    result = summarize_metadata(data_dir, output_dir)

    assert result == 0
    assert (output_dir / "index-2016.json").exists()

    # Verify the JSON is properly serialized (no AnyUrl objects)
    with open(output_dir / "index-2016.json") as f:
        data = json.load(f)  # Should not raise TypeError
        assert len(data) == 1
        assert data[0]["set"] == "10255"
        assert (
            data[0]["set_image_url"]
            == "https://www.lego.com/cdn/cs/set/assets/image.jpg"
        )
        assert isinstance(data[0]["set_image_url"], str)
        assert len(data[0]["pdfs"]) == 1
        assert (
            data[0]["pdfs"][0]["url"]
            == "https://www.lego.com/cdn/product-assets/instructions.pdf"
        )
        assert isinstance(data[0]["pdfs"][0]["url"], str)
