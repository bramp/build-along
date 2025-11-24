"""Tests for download command."""

import io
import sys
from datetime import timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

from pydantic import AnyUrl

from build_a_long.downloader.commands.download import (
    get_set_numbers_from_args,
    run_download,
)
from build_a_long.schemas import InstructionMetadata, PdfEntry


def test_get_set_numbers_from_args_single_arg():
    args = MagicMock()
    args.set_number = "12345"
    args.stdin = False

    result = get_set_numbers_from_args(args)
    assert result == ["12345"]


def test_get_set_numbers_from_args_invalid_arg(capsys):
    args = MagicMock()
    args.set_number = "invalid_id"
    args.stdin = False

    result = get_set_numbers_from_args(args)
    captured = capsys.readouterr()
    assert result == []
    assert "Invalid set ID" in captured.err


def test_get_set_numbers_from_args_stdin_valid(monkeypatch):
    args = MagicMock()
    args.set_number = None
    args.stdin = True

    monkeypatch.setattr(sys, "stdin", io.StringIO("12345\n67890\n"))
    result = get_set_numbers_from_args(args)
    assert result == ["12345", "67890"]


def test_get_set_numbers_from_args_stdin_with_invalid(monkeypatch, capsys):
    args = MagicMock()
    args.set_number = None
    args.stdin = True

    monkeypatch.setattr(sys, "stdin", io.StringIO("12345\ninvalid_id\n67890\n"))
    result = get_set_numbers_from_args(args)
    captured = capsys.readouterr()
    assert result == ["12345", "67890"]
    assert "Invalid set ID 'invalid_id'" in captured.err


@patch("build_a_long.downloader.commands.download.LegoInstructionDownloader")
def test_run_download_single_set(mock_downloader_class, capsys):
    """Test downloading a single set."""
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    args = MagicMock()
    args.set_number = "12345"
    args.stdin = False
    args.locale = "en-us"
    args.out_dir = None
    args.metadata = False
    args.overwrite_metadata = False
    args.overwrite_metadata_if_older_than = "1d"
    args.overwrite_download = False
    args.debug = False

    exit_code = run_download(args)

    assert exit_code == 0
    mock_downloader_class.assert_called_once()
    call_kwargs = mock_downloader_class.call_args[1]
    assert call_kwargs["locale"] == "en-us"
    assert call_kwargs["overwrite_metadata_if_older_than"] == timedelta(days=1)
    assert call_kwargs["overwrite_download"] is False

    mock_instance.process_sets.assert_called_once_with(["12345"])
    assert "All done" in capsys.readouterr().out


@patch("build_a_long.downloader.commands.download.LegoInstructionDownloader")
def test_run_download_multiple_sets_stdin(mock_downloader_class, monkeypatch):
    """Test downloading multiple sets from stdin."""
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(sys, "stdin", io.StringIO("12345\n67890\n"))

    args = MagicMock()
    args.set_number = None
    args.stdin = True
    args.locale = "en-us"
    args.out_dir = None
    args.metadata = False
    args.overwrite_metadata = False
    args.overwrite_metadata_if_older_than = "1d"
    args.overwrite_download = False
    args.debug = False

    exit_code = run_download(args)

    assert exit_code == 0
    mock_instance.process_sets.assert_called_once_with(["12345", "67890"])


@patch("build_a_long.downloader.commands.download.LegoInstructionDownloader")
def test_run_download_custom_locale_and_overwrites(mock_downloader_class):
    """Test download with custom locale and overwrite options."""
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    args = MagicMock()
    args.set_number = "12345"
    args.stdin = False
    args.locale = "de-de"
    args.out_dir = None
    args.metadata = False
    args.overwrite_metadata = True
    args.overwrite_metadata_if_older_than = "1d"
    args.overwrite_download = True
    args.debug = False

    exit_code = run_download(args)

    assert exit_code == 0
    call_kwargs = mock_downloader_class.call_args[1]
    assert call_kwargs["locale"] == "de-de"
    assert call_kwargs["overwrite_metadata_if_older_than"] == timedelta(seconds=0)
    assert call_kwargs["overwrite_download"] is True


@patch("build_a_long.downloader.commands.download.LegoInstructionDownloader")
@patch("build_a_long.downloader.commands.download.build_metadata")
def test_run_download_metadata_mode(mock_build_metadata, mock_downloader_class, capsys):
    """Test --metadata flag outputs JSON without downloading."""
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.fetch_instructions_page.return_value = "<html></html>"
    mock_downloader_class.return_value = mock_instance

    # Mock metadata return
    mock_meta = InstructionMetadata(
        set="12345",
        locale="en-us",
        name="Test Set",
        theme="LEGO® Theme",
        age="9+",
        pieces=100,
        year=2024,
        set_image_url=None,
        pdfs=[
            PdfEntry(
                url=AnyUrl("https://example.com/test.pdf"),
                filename="test.pdf",
                preview_url=None,
                filesize=None,
                filehash=None,
            )
        ],
    )
    mock_build_metadata.return_value = mock_meta

    args = MagicMock()
    args.set_number = "12345"
    args.stdin = False
    args.locale = "en-us"
    args.metadata = True
    args.debug = False

    exit_code = run_download(args)

    assert exit_code == 0
    # Should not call process_sets (download path)
    mock_instance.process_sets.assert_not_called()
    # Should fetch page and build metadata
    mock_instance.fetch_instructions_page.assert_called_once_with("12345")
    mock_build_metadata.assert_called_once()

    # Check JSON output
    output = capsys.readouterr().out
    assert '"set": "12345"' in output
    assert '"name": "Test Set"' in output
    assert '"theme": "LEGO® Theme"' in output
    assert '"age": "9+"' in output


@patch("build_a_long.downloader.commands.download.LegoInstructionDownloader")
def test_run_download_custom_out_dir(mock_downloader_class):
    """Test download with custom output directory."""
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    args = MagicMock()
    args.set_number = "12345"
    args.stdin = False
    args.locale = "en-us"
    args.out_dir = "/tmp/lego"
    args.metadata = False
    args.overwrite_metadata = False
    args.overwrite_metadata_if_older_than = "1d"
    args.overwrite_download = False
    args.debug = False

    exit_code = run_download(args)

    assert exit_code == 0
    call_kwargs = mock_downloader_class.call_args[1]
    assert call_kwargs["out_dir"] == Path("/tmp/lego")


@patch("build_a_long.downloader.commands.download.LegoInstructionDownloader")
def test_run_download_empty_stdin(mock_downloader_class, monkeypatch, capsys):
    """Test error when no set numbers provided from stdin."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n\n"))

    args = MagicMock()
    args.set_number = None
    args.stdin = True

    exit_code = run_download(args)

    assert exit_code == 1
    mock_downloader_class.assert_not_called()
    assert "Error: No LEGO set numbers provided." in capsys.readouterr().err


@patch("build_a_long.downloader.commands.download.LegoInstructionDownloader")
def test_run_download_invalid_set_ids_from_stdin(
    mock_downloader_class, monkeypatch, capsys
):
    """Test that invalid set IDs are skipped but valid ones are processed."""
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(sys, "stdin", io.StringIO("12345\ninvalid_id\n67890\n"))

    args = MagicMock()
    args.set_number = None
    args.stdin = True
    args.locale = "en-us"
    args.out_dir = None
    args.metadata = False
    args.overwrite_metadata = False
    args.overwrite_metadata_if_older_than = "1d"
    args.overwrite_download = False
    args.debug = False

    exit_code = run_download(args)

    assert exit_code == 0
    mock_instance.process_sets.assert_called_once_with(["12345", "67890"])
    assert (
        "Warning: Invalid set ID 'invalid_id' from stdin. Skipping."
        in capsys.readouterr().err
    )


def test_run_download_no_sets_provided(capsys):
    """Test error when no valid set numbers are provided."""
    args = MagicMock()
    args.set_number = None
    args.stdin = False

    exit_code = run_download(args)

    assert exit_code == 1
    assert "Error: No LEGO set numbers provided." in capsys.readouterr().err


def test_run_download_invalid_duration_string(capsys):
    """Test error handling for invalid duration string."""
    args = MagicMock()
    args.set_number = "12345"
    args.stdin = False
    args.locale = "en-us"
    args.out_dir = None
    args.metadata = False
    args.overwrite_metadata = False
    args.overwrite_metadata_if_older_than = "invalid_duration"
    args.overwrite_download = False
    args.debug = False

    exit_code = run_download(args)

    assert exit_code == 1
    assert "Error: Invalid duration string:" in capsys.readouterr().err
