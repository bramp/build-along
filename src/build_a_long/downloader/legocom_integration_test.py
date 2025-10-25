"""Integration tests for legocom.py - Tests against real LEGO.com (pytest style).

These tests make real HTTP requests to LEGO.com and are skipped by default.
Run with: ENABLE_INTEGRATION_TESTS=true pants test src/build_a_long/downloader::

Why integration tests:
- Verify our parsing works against real HTML (not just test fixtures)
- Detect when LEGO.com changes their page structure
- Ensure JSON fields we rely on still exist

When to run:
- Before releases
- Periodically in CI (e.g., nightly builds)
- When debugging production issues
- After updating parsing logic

Note: These tests may fail if:
- LEGO.com is down or slow
- Set numbers become invalid/retired
- Page structure changes (expected over time)
"""

import os

import pytest

from build_a_long.downloader.legocom import (
    build_instructions_url,
    parse_instruction_pdf_urls,
    parse_set_metadata,
)

# Skip all tests in this module unless explicitly enabled
# Set ENABLE_INTEGRATION_TESTS=true to run these tests
ENABLE_INTEGRATION_TESTS = (
    os.getenv("ENABLE_INTEGRATION_TESTS", "false").lower() == "true"
)

pytestmark = pytest.mark.skipif(
    not ENABLE_INTEGRATION_TESTS,
    reason="Integration tests disabled - set ENABLE_INTEGRATION_TESTS=true to enable",
)


@pytest.fixture
def http_client():
    """Provide an HTTP client for making requests."""
    import httpx

    with httpx.Client(follow_redirects=True, timeout=30) as client:
        yield client


def test_fetch_real_set_30708(http_client):
    """Test fetching and parsing a real small set (Millennium Falcon mini-build)."""
    set_number = "30708"
    url = build_instructions_url(set_number, "en-us")
    response = http_client.get(url)
    response.raise_for_status()
    html = response.text

    # Test PDF extraction
    pdfs = parse_instruction_pdf_urls(html)
    assert len(pdfs) >= 1, "Should find at least one instruction PDF"
    assert all("product.bi.core.pdf" in pdf.url for pdf in pdfs)
    assert all(pdf.url.startswith("https://") for pdf in pdfs)

    # Test metadata extraction
    meta = parse_set_metadata(html, set_number=set_number, locale="en-us")

    # Verify we got the expected metadata
    assert meta.name and "Millennium Falcon" in meta.name
    assert meta.theme and "Star Wars" in meta.theme
    assert meta.age == "6+"
    assert meta.pieces == 74
    assert meta.year and meta.year >= 2024


def test_fetch_real_set_75419(http_client):
    """Test fetching and parsing a real large set (Death Star)."""
    set_number = "75419"
    url = build_instructions_url(set_number, "en-us")
    response = http_client.get(url)
    response.raise_for_status()
    html = response.text

    # Test PDF extraction - large sets have multiple booklets
    pdfs = parse_instruction_pdf_urls(html)
    assert len(pdfs) >= 4, (
        f"Death Star should have multiple instruction PDFs, found {len(pdfs)}"
    )

    # Test metadata extraction
    meta = parse_set_metadata(html, set_number=set_number, locale="en-us")

    assert meta.name and "Death Star" in meta.name
    assert meta.theme and "Star Wars" in meta.theme
    assert meta.age == "18+"
    assert meta.pieces and meta.pieces > 9000
    assert meta.year and meta.year >= 2024


def test_json_fields_exist_in_real_pages(http_client):
    """Verify that the JSON fields we rely on still exist in LEGO.com pages.

    This test is intentionally more lenient - it just checks that our
    extraction methods find SOMETHING, to detect if LEGO.com changes
    their JSON structure completely.
    """
    set_number = "30708"
    url = build_instructions_url(set_number, "en-us")
    response = http_client.get(url)
    response.raise_for_status()
    html = response.text

    # Check that JSON extraction methods return values
    meta = parse_set_metadata(html, set_number=set_number, locale="en-us")
    assert meta.name
    assert meta.theme
    assert meta.age
    assert meta.pieces
    assert meta.year


def test_different_locale(http_client):
    """Test that different locales work (German example)."""
    set_number = "30708"
    url = build_instructions_url(set_number, "de-de")
    response = http_client.get(url)
    response.raise_for_status()
    html = response.text

    # Should still find PDFs regardless of locale
    pdfs = parse_instruction_pdf_urls(html)
    assert len(pdfs) >= 1

    # Metadata extraction should still work (JSON fields are often not localized)
    meta = parse_set_metadata(html, set_number=set_number, locale="de-de")
    assert meta.pieces == 74
    assert meta.theme


def test_invalid_set_number_handling(http_client):
    """Test that invalid/non-existent set numbers are handled gracefully.

    Our parsing should not crash regardless of what LEGO.com returns for
    invalid set numbers, even if the page structure is completely different.
    """
    from build_a_long.downloader.metadata import Metadata

    set_number = "99999999"  # Unlikely to exist
    locale = "en-us"
    url = build_instructions_url(set_number, locale)
    response = http_client.get(url)

    # Our code should not crash regardless of what LEGO.com returns
    html = response.text
    pdfs = parse_instruction_pdf_urls(html)
    meta = parse_set_metadata(html, set_number=set_number, locale=locale)

    # Should not crash, and should return proper types (likely with empty/minimal data)
    assert isinstance(pdfs, list)
    assert isinstance(meta, Metadata)
    assert meta.set == set_number
    assert meta.locale == locale
