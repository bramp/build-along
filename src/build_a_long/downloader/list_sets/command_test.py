"""Tests for list-sets command."""

import argparse
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

from build_a_long.downloader.list_sets.command import (
    add_list_sets_parser,
    run_list_sets,
)


def make_args(**kwargs):
    """Create a mock args object with default values."""
    defaults = {
        "source": "lego",
        "locale": "en-us",
        "data_dir": None,
        "no_cache": False,
        "cache_ttl": "1d",
        "min_year": None,
        "limit": None,
        "count": False,
        "rate_limit": 60,
        "debug": False,
    }
    defaults.update(kwargs)
    return MagicMock(**defaults)


@patch("build_a_long.downloader.list_sets.command.LegoInstructionDownloader")
def test_run_list_sets_outputs_sets(mock_downloader_class, capsys):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.get_set_list.return_value = ["10210", "10211", "75419"]
    mock_downloader_class.return_value = mock_instance

    args = make_args()
    exit_code = run_list_sets(args)

    assert exit_code == 0
    mock_instance.get_set_list.assert_called_once_with(
        source="lego",
        use_cache=True,
        cache_ttl=timedelta(days=1),
        min_year=None,
    )
    captured = capsys.readouterr()
    assert captured.out == "10210\n10211\n75419\n"


@patch("build_a_long.downloader.list_sets.command.LegoInstructionDownloader")
def test_run_list_sets_count_flag(mock_downloader_class, capsys):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.get_set_list.return_value = ["10210", "10211", "75419"]
    mock_downloader_class.return_value = mock_instance

    args = make_args(count=True)
    exit_code = run_list_sets(args)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == "3"


@patch("build_a_long.downloader.list_sets.command.LegoInstructionDownloader")
def test_run_list_sets_limit_flag(mock_downloader_class, capsys):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.get_set_list.return_value = ["10210", "10211", "75419"]
    mock_downloader_class.return_value = mock_instance

    args = make_args(limit=2)
    exit_code = run_list_sets(args)

    assert exit_code == 0
    captured = capsys.readouterr()
    assert captured.out == "10210\n10211\n"


@patch("build_a_long.downloader.list_sets.command.LegoInstructionDownloader")
def test_run_list_sets_rebrickable_source(mock_downloader_class):
    mock_instance = MagicMock()
    mock_instance.__enter__ = MagicMock(return_value=mock_instance)
    mock_instance.__exit__ = MagicMock(return_value=None)
    mock_instance.get_set_list.return_value = ["75419"]
    mock_downloader_class.return_value = mock_instance

    args = make_args(source="rebrickable", min_year=2024, no_cache=True)
    exit_code = run_list_sets(args)

    assert exit_code == 0
    mock_instance.get_set_list.assert_called_once_with(
        source="rebrickable",
        use_cache=False,
        cache_ttl=None,
        min_year=2024,
    )


def test_run_list_sets_invalid_cache_ttl(capsys):
    args = make_args(cache_ttl="invalid_duration")
    exit_code = run_list_sets(args)

    assert exit_code == 1
    assert "Invalid duration string for --cache-ttl" in capsys.readouterr().err


def test_parser_options():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    add_list_sets_parser(subparsers)

    args = parser.parse_args(["list-sets", "--source", "rebrickable", "--min-year", "2020", "--limit", "10"])
    assert args.command == "list-sets"
    assert args.source == "rebrickable"
    assert args.min_year == 2020
    assert args.limit == 10
    assert args.no_cache is False

    args_alias = parser.parse_args(["list", "--count"])
    assert args_alias.command == "list"
    assert args_alias.count is True
