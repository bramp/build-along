"""Tests for util.py - pure utility functions (pytest style)."""

from pydantic import AnyUrl

from build_a_long.downloader.util import extract_filename_from_url, is_valid_set_id


def test_is_valid_set_id_numeric():
    assert is_valid_set_id("12345")
    assert is_valid_set_id("75419")


def test_is_valid_set_id_invalid():
    assert not is_valid_set_id("abc123")
    assert not is_valid_set_id("invalid")
    assert not is_valid_set_id("")
    assert not is_valid_set_id("123-456")


def test_extract_filename_from_url_simple():
    """Test basic filename extraction from URLs."""
    assert extract_filename_from_url("https://example.com/file.pdf") == "file.pdf"
    assert (
        extract_filename_from_url("https://example.com/path/to/file.pdf") == "file.pdf"
    )
    assert extract_filename_from_url("http://lego.com/6602000.pdf") == "6602000.pdf"


def test_extract_filename_from_url_with_query_params():
    """Test filename extraction when URL has query parameters."""
    assert (
        extract_filename_from_url("https://example.com/file.pdf?version=1")
        == "file.pdf"
    )
    assert (
        extract_filename_from_url("https://example.com/doc.pdf?param=value&other=123")
        == "doc.pdf"
    )


def test_extract_filename_from_url_with_fragment():
    """Test filename extraction when URL has fragments."""
    assert (
        extract_filename_from_url("https://example.com/file.pdf#section") == "file.pdf"
    )


def test_extract_filename_from_url_ambiguous_cases():
    """Test that ambiguous URLs return None."""
    # URL with trailing slash - no clear filename
    assert extract_filename_from_url("https://example.com/") is None
    assert extract_filename_from_url("https://example.com/path/") is None

    # URL with no path
    assert extract_filename_from_url("https://example.com") is None

    # URL with only root path
    assert extract_filename_from_url("https://example.com/") is None


def test_extract_filename_from_url_with_anyurl_object():
    """Test that function works with Pydantic AnyUrl objects."""
    url = AnyUrl("https://example.com/test.pdf")
    assert extract_filename_from_url(url) == "test.pdf"

    url_with_path = AnyUrl("https://example.com/downloads/manual.pdf")
    assert extract_filename_from_url(url_with_path) == "manual.pdf"


def test_extract_filename_from_url_special_characters():
    """Test filename extraction with special characters in filename."""
    assert (
        extract_filename_from_url("https://example.com/file%20with%20spaces.pdf")
        == "file with spaces.pdf"
    )
    assert (
        extract_filename_from_url("https://example.com/file-name_123.pdf")
        == "file-name_123.pdf"
    )


def test_extract_filename_from_url_lego_specific():
    """Test filename extraction for specific LEGO URLs with special / invalid
    characters. These get fixed up by _fix_url_encoding_issues."""
    url1 = "https://www.lego.com/cdn/product-assets/product.bi.additional.extra.pdf/8110_X_8110%20Snow%20Plow%20#1.pdf"
    url2 = "https://www.lego.com/cdn/product-assets/product.bi.additional.extra.pdf/8110_X_8110%20Snow%20Plow%20#2.pdf"

    filename1 = extract_filename_from_url(url1)
    filename2 = extract_filename_from_url(url2)

    assert filename1 == "8110_X_8110 Snow Plow "
    assert filename2 == "8110_X_8110 Snow Plow "
    assert filename1 == filename2
