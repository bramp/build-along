"""Tests for main.py - CLI routing and command integration."""

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

from build_a_long.downloader.main import main


@patch("build_a_long.downloader.commands.download.LegoInstructionDownloader")
def test_main_routes_to_download_command(mock_downloader_class, monkeypatch, capsys):
    """Test that main correctly routes to the download command."""
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.process_sets.return_value = 0
    mock_downloader_class.return_value = mock_instance

    monkeypatch.setattr(sys, "argv", ["main.py", "download", "12345"])
    exit_code = main()

    assert exit_code == 0
    mock_instance.process_sets.assert_called_once_with(["12345"])
    assert "All done" in capsys.readouterr().out


@patch("build_a_long.downloader.commands.summarize._summarize_metadata")
def test_main_routes_to_summarize_command(mock_summarize, monkeypatch):
    """Test that main correctly routes to the summarize command."""
    mock_summarize.return_value = 0
    monkeypatch.setattr(
        sys, "argv", ["main.py", "summarize", "--data-dir", "/tmp/data"]
    )
    exit_code = main()

    assert exit_code == 0
    mock_summarize.assert_called_once_with(Path("/tmp/data"), Path("data/indices"))


@patch("build_a_long.downloader.commands.verify._verify_data_integrity")
def test_main_routes_to_verify_command(mock_verify, monkeypatch):
    """Test that main correctly routes to the verify command."""
    mock_verify.return_value = 0
    monkeypatch.setattr(sys, "argv", ["main.py", "verify", "--data-dir", "/tmp/data"])
    exit_code = main()

    assert exit_code == 0
    mock_verify.assert_called_once_with(Path("/tmp/data"))


def test_main_no_command_specified(monkeypatch, capsys):
    """Test error when no command is specified."""
    # Mock argv to simulate running without a subcommand
    # argparse will exit, so we need to catch SystemExit
    monkeypatch.setattr(sys, "argv", ["main.py"])
    try:
        main()
    except SystemExit as e:
        # argparse exits with code 2 for usage errors
        assert e.code == 2
