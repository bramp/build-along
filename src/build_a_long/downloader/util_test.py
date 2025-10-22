"""Tests for util.py - pure utility functions (pytest style)."""

from build_a_long.downloader.util import is_valid_set_id


def test_is_valid_set_id_numeric():
    assert is_valid_set_id("12345")
    assert is_valid_set_id("75419")


def test_is_valid_set_id_invalid():
    assert not is_valid_set_id("abc123")
    assert not is_valid_set_id("invalid")
    assert not is_valid_set_id("")
    assert not is_valid_set_id("123-456")
