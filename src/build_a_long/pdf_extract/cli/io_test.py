"""Tests for io module."""

import bz2
import gzip
import json
import tempfile
from pathlib import Path

import pytest

from build_a_long.pdf_extract.cli.io import load_json, open_compressed


def test_open_compressed_with_uncompressed() -> None:
    """Test opening uncompressed file."""
    test_content = "Hello, World!"

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write(test_content)
        temp_path = Path(f.name)

    try:
        with open_compressed(temp_path, "rt", encoding="utf-8") as f:
            result = f.read()
        assert result == test_content
    finally:
        temp_path.unlink()


def test_open_compressed_with_bz2() -> None:
    """Test opening bz2 compressed file."""
    test_content = "Compressed with bz2!"

    with tempfile.NamedTemporaryFile(suffix=".txt.bz2", delete=False) as f:
        temp_path = Path(f.name)

    try:
        with bz2.open(temp_path, "wt", encoding="utf-8") as f:
            f.write(test_content)

        with open_compressed(temp_path, "rt", encoding="utf-8") as f:
            result = f.read()
        assert result == test_content
    finally:
        temp_path.unlink()


def test_open_compressed_with_gz() -> None:
    """Test opening gzip compressed file."""
    test_content = "Compressed with gzip!"

    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
        temp_path = Path(f.name)

    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as f:
            f.write(test_content)

        with open_compressed(temp_path, "rt", encoding="utf-8") as f:
            result = f.read()
        assert result == test_content
    finally:
        temp_path.unlink()


def test_open_compressed_binary_mode() -> None:
    """Test opening compressed file in binary mode."""
    test_content = b"Binary data\x00\x01\x02"

    with tempfile.NamedTemporaryFile(suffix=".bin.bz2", delete=False) as f:
        temp_path = Path(f.name)

    try:
        with bz2.open(temp_path, "wb") as f:
            f.write(test_content)

        with open_compressed(temp_path, "rb") as f:
            result = f.read()
        assert result == test_content
    finally:
        temp_path.unlink()


def test_open_compressed_write_mode() -> None:
    """Test writing to compressed file."""
    test_content = "Writing to compressed file!"

    with tempfile.NamedTemporaryFile(suffix=".txt.gz", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Write using open_compressed
        with open_compressed(temp_path, "wt", encoding="utf-8") as f:
            f.write(test_content)  # type: ignore[arg-type]

        # Read back to verify
        with gzip.open(temp_path, "rt", encoding="utf-8") as f:
            result = f.read()
        assert result == test_content
    finally:
        temp_path.unlink()


def test_load_json_with_valid_json() -> None:
    """Test loading valid JSON from uncompressed file."""
    data = {"key": "value", "number": 42}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(data, f)
        temp_path = Path(f.name)

    try:
        result = load_json(temp_path)
        assert result == data
    finally:
        temp_path.unlink()


def test_load_json_with_bz2() -> None:
    """Test loading valid JSON from bz2 compressed file."""
    data = {"compressed": True, "format": "bz2"}

    with tempfile.NamedTemporaryFile(suffix=".json.bz2", delete=False) as f:
        temp_path = Path(f.name)

    try:
        with bz2.open(temp_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_json(temp_path)
        assert result == data
    finally:
        temp_path.unlink()


def test_load_json_with_gz() -> None:
    """Test loading valid JSON from gzip compressed file."""
    data = {"compressed": True, "format": "gzip"}

    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
        temp_path = Path(f.name)

    try:
        with gzip.open(temp_path, "wt", encoding="utf-8") as f:
            json.dump(data, f)

        result = load_json(temp_path)
        assert result == data
    finally:
        temp_path.unlink()


def test_load_json_with_invalid_json() -> None:
    """Test that invalid JSON raises ValueError with helpful message."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("not valid json")
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError) as exc_info:
            load_json(temp_path)

        # Check that the error message contains the file path
        assert str(temp_path) in str(exc_info.value)
        # Check that the error message contains position information
        assert "line" in str(exc_info.value).lower()
        assert "column" in str(exc_info.value).lower()
    finally:
        temp_path.unlink()


def test_load_json_with_empty_file() -> None:
    """Test that empty file raises ValueError with helpful message."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        # Write nothing to create empty file
        temp_path = Path(f.name)

    try:
        with pytest.raises(ValueError) as exc_info:
            load_json(temp_path)

        # Check that the error message contains the file path
        assert str(temp_path) in str(exc_info.value)
        assert "Failed to parse JSON" in str(exc_info.value)
    finally:
        temp_path.unlink()


def test_load_json_with_corrupted_bz2() -> None:
    """Test that corrupted JSON in bz2 file raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".json.bz2", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Write corrupted JSON to bz2 file
        with bz2.open(temp_path, "wt", encoding="utf-8") as f:
            f.write("mv { invalid json }")

        with pytest.raises(ValueError) as exc_info:
            load_json(temp_path)

        # Check that the error message is helpful
        assert str(temp_path) in str(exc_info.value)
        assert "Failed to parse JSON" in str(exc_info.value)
        assert "line" in str(exc_info.value).lower()
    finally:
        temp_path.unlink()


def test_load_json_with_corrupted_gz() -> None:
    """Test that corrupted JSON in gzip file raises ValueError."""
    with tempfile.NamedTemporaryFile(suffix=".json.gz", delete=False) as f:
        temp_path = Path(f.name)

    try:
        # Write corrupted JSON to gzip file
        with gzip.open(temp_path, "wt", encoding="utf-8") as f:
            f.write("{ incomplete: json")

        with pytest.raises(ValueError) as exc_info:
            load_json(temp_path)

        # Check that the error message is helpful
        assert str(temp_path) in str(exc_info.value)
        assert "Failed to parse JSON" in str(exc_info.value)
    finally:
        temp_path.unlink()
