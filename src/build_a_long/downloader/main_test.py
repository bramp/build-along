"""Tests for main.py - CLI entry point and argument parsing (pytest style)."""

import io
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from build_a_long.downloader.main import get_set_numbers_from_args, main


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


@patch("build_a_long.downloader.main.LegoInstructionDownloader")
def test_main_single_set_number_from_arg(mock_downloader_class, monkeypatch, capsys):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(sys, "argv", ["main.py", "12345"])
    exit_code = main()

    assert exit_code == 0
    # Verify downloader was created with correct args
    mock_downloader_class.assert_called_once()
    call_kwargs = mock_downloader_class.call_args[1]
    assert call_kwargs["locale"] == "en-us"
    assert call_kwargs["dry_run"] is False
    assert call_kwargs["overwrite"] is False

    # Verify process_sets was called with the set number
    mock_instance.process_sets.assert_called_once_with(["12345"])
    assert "All done" in capsys.readouterr().out


@patch("build_a_long.downloader.main.LegoInstructionDownloader")
def test_main_multiple_set_numbers_from_stdin(mock_downloader_class, monkeypatch):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(sys, "argv", ["main.py", "--stdin"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("12345\n67890\n"))
    exit_code = main()

    assert exit_code == 0
    mock_instance.process_sets.assert_called_once_with(["12345", "67890"])


@patch("build_a_long.downloader.main.LegoInstructionDownloader")
def test_main_custom_locale_and_force(mock_downloader_class, monkeypatch):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(
        sys, "argv", ["main.py", "--locale", "de-de", "--force", "12345"]
    )
    exit_code = main()

    assert exit_code == 0
    call_kwargs = mock_downloader_class.call_args[1]
    assert call_kwargs["locale"] == "de-de"
    assert call_kwargs["overwrite"] is True


@patch("build_a_long.downloader.main.LegoInstructionDownloader")
def test_main_dry_run(mock_downloader_class, monkeypatch):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(sys, "argv", ["main.py", "--dry-run", "12345"])
    exit_code = main()

    assert exit_code == 0
    call_kwargs = mock_downloader_class.call_args[1]
    assert call_kwargs["dry_run"] is True


@patch("build_a_long.downloader.main.LegoInstructionDownloader")
def test_main_custom_out_dir(mock_downloader_class, monkeypatch):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(sys, "argv", ["main.py", "--out-dir", "/tmp/lego", "12345"])
    exit_code = main()

    assert exit_code == 0
    call_kwargs = mock_downloader_class.call_args[1]
    assert call_kwargs["out_dir"] == Path("/tmp/lego")


@patch("build_a_long.downloader.main.LegoInstructionDownloader")
def test_main_empty_stdin(mock_downloader_class, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["main.py", "--stdin"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("\n\n"))
    exit_code = main()

    assert exit_code == 1  # Expect error exit code
    mock_downloader_class.assert_not_called()
    assert "Error: No LEGO set numbers provided." in capsys.readouterr().err


@patch("build_a_long.downloader.main.LegoInstructionDownloader")
def test_main_invalid_set_ids_from_stdin(mock_downloader_class, monkeypatch, capsys):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(sys, "argv", ["main.py", "--stdin"])
    monkeypatch.setattr(sys, "stdin", io.StringIO("12345\ninvalid_id\n67890\n"))
    exit_code = main()

    assert exit_code == 0
    # Should skip invalid ID
    mock_instance.process_sets.assert_called_once_with(["12345", "67890"])
    assert (
        "Warning: Invalid set ID 'invalid_id' from stdin. Skipping."
        in capsys.readouterr().err
    )
