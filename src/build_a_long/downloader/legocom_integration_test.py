"""Integration tests for legocom.py - Tests against real LEGO.com (pytest style).

These tests make real HTTP requests to LEGO.com and are skipped by default.
Run with: pants test --test-pytest-extra-args='--integration' src/build_a_long/downloader::

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

import pytest

from build_a_long.downloader.legocom import (
    build_instructions_url,
    parse_instruction_pdf_urls,
    parse_set_metadata,
)

# Skip all tests in this module by default
# TODO: Configure proper integration test runner
pytestmark = pytest.mark.skip(reason="Integration tests disabled - probe real LEGO.com")


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
    assert all("product.bi.core.pdf" in pdf for pdf in pdfs)
    assert all(pdf.startswith("https://") for pdf in pdfs)

    # Test metadata extraction
    meta = parse_set_metadata(html)

    # Verify we got the expected metadata
    assert "name" in meta, "Should extract set name"
    assert "Millennium Falcon" in meta["name"], (
        f"Expected Millennium Falcon in name, got: {meta.get('name')}"
    )

    assert "theme" in meta, "Should extract theme"
    assert "Star Wars" in meta["theme"], (
        f"Expected Star Wars theme, got: {meta.get('theme')}"
    )

    assert "age" in meta, "Should extract age rating"
    assert meta["age"] == "6+", f"Expected age 6+, got: {meta.get('age')}"

    assert "pieces" in meta, "Should extract piece count"
    assert meta["pieces"] == 74, f"Expected 74 pieces, got: {meta.get('pieces')}"

    assert "year" in meta, "Should extract year"
    assert meta["year"] >= 2024, f"Expected year >= 2024, got: {meta.get('year')}"


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
    meta = parse_set_metadata(html)

    assert "name" in meta
    assert "Death Star" in meta["name"], (
        f"Expected Death Star in name, got: {meta.get('name')}"
    )

    assert "theme" in meta
    assert "Star Wars" in meta["theme"]

    assert "age" in meta
    assert meta["age"] == "18+", f"Expected age 18+, got: {meta.get('age')}"

    assert "pieces" in meta
    assert meta["pieces"] > 9000, (
        f"Death Star should have >9000 pieces, got: {meta.get('pieces')}"
    )

    assert "year" in meta
    assert meta["year"] >= 2024


def test_json_fields_exist_in_real_pages(http_client):
    """Verify that the JSON fields we rely on still exist in LEGO.com pages.

    This test is intentionally more lenient - it just checks that our
    extraction methods find SOMETHING, to detect if LEGO.com changes
    their JSON structure completely.
    """
    from build_a_long.downloader.legocom import (
        _extract_age_from_json,
        _extract_name_from_json,
        _extract_pieces_from_json,
        _extract_theme_from_json,
        _extract_year_from_json,
    )

    set_number = "30708"
    url = build_instructions_url(set_number, "en-us")
    response = http_client.get(url)
    response.raise_for_status()
    html = response.text

    # Check that JSON extraction methods return values
    name = _extract_name_from_json(html)
    assert name is not None, (
        "JSON 'name' field not found - LEGO.com may have changed structure"
    )

    theme = _extract_theme_from_json(html)
    assert theme is not None, (
        "JSON 'themeName' field not found - LEGO.com may have changed structure"
    )

    age = _extract_age_from_json(html)
    assert age is not None, (
        "JSON 'ageRating' field not found - LEGO.com may have changed structure"
    )

    pieces = _extract_pieces_from_json(html)
    assert pieces is not None, (
        "JSON 'setPieceCount' field not found - LEGO.com may have changed structure"
    )

    year = _extract_year_from_json(html)
    assert year is not None, (
        "JSON 'year' field not found - LEGO.com may have changed structure"
    )


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
    meta = parse_set_metadata(html)
    assert "pieces" in meta
    assert meta["pieces"] == 74

    # Theme names might be localized or not - just check it exists
    assert "theme" in meta


@pytest.mark.skip(
    reason="LEGO.com handling of invalid sets varies - this test documents expected behavior but is not deterministic"
)
def test_invalid_set_number_handling(http_client):
    """Test that invalid/non-existent set numbers are handled gracefully.

    Note: This test is skipped because LEGO.com's behavior for invalid sets
    is not consistent - it may return 404, redirect, or show a search page.
    """
    set_number = "99999999"  # Unlikely to exist
    url = build_instructions_url(set_number, "en-us")
    response = http_client.get(url)

    # Our code should not crash regardless of what LEGO.com returns
    html = response.text
    pdfs = parse_instruction_pdf_urls(html)
    meta = parse_set_metadata(html)

    # Should not crash, but likely returns empty results
    assert isinstance(pdfs, list)
    assert isinstance(meta, dict)
