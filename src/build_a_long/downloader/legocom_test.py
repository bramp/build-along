"""Tests for legocom.py - LEGO.com website parsing (pytest style)."""

from build_a_long.downloader.legocom import (
    LEGO_BASE,
    _extract_age_from_json,
    _extract_age_from_text,
    _extract_name_from_html,
    _extract_name_from_json,
    _extract_pieces_from_json,
    _extract_pieces_from_text,
    _extract_theme_from_json,
    _extract_year_from_json,
    _extract_year_from_text,
    build_instructions_url,
    parse_instruction_pdf_urls,
    parse_set_metadata,
)
from bs4 import BeautifulSoup


def test_build_instructions_url():
    url = build_instructions_url("75419", "en-us")
    assert url == "https://www.lego.com/en-us/service/building-instructions/75419"


def test_build_instructions_url_different_locale():
    url = build_instructions_url("12345", "de-de")
    assert url == "https://www.lego.com/de-de/service/building-instructions/12345"


def test_parse_instruction_pdf_urls_absolute_and_relative():
    html = (
        '<a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf">Download</a>'
        '<a href="/cdn/product-assets/product.bi.core.pdf/6602645.pdf">Download</a>'
        '<a href="/cdn/x/notpdf.txt">Ignore</a>'
    )
    urls = parse_instruction_pdf_urls(html, base=LEGO_BASE)
    assert urls == [
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf",
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602645.pdf",
    ]


def test_parse_instruction_pdf_urls_deduplicates():
    html = (
        '<a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/dup.pdf">A</a>'
        '<a href="/cdn/product-assets/product.bi.core.pdf/dup.pdf">B</a>'
    )
    urls = parse_instruction_pdf_urls(html, base=LEGO_BASE)
    assert urls == [
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/dup.pdf"
    ]


def test_parse_instruction_pdf_urls_filters_non_instructions():
    html = (
        '<a href="https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf">Instruction</a>'
        '<a href="https://www.lego.com/cdn/cs/aboutus/assets/blt1a02e1065ccb2f31/LEGOGroup_ModernSlaveryTransparencyStatement_2024.pdf">Non-Instruction</a>'
    )
    urls = parse_instruction_pdf_urls(html, base=LEGO_BASE)
    assert urls == [
        "https://www.lego.com/cdn/product-assets/product.bi.core.pdf/6602644.pdf"
    ]


# Tests for individual metadata extraction functions


def test_extract_name_from_json():
    html = '"name":"Millennium Falcon™ Mini-Build","setNumber":"30708"'
    assert _extract_name_from_json(html) == "Millennium Falcon™ Mini-Build"


def test_extract_name_from_json_with_unicode_escapes():
    html = r'"name":"Rock \u0026 Roll Band","setNumber":"12345"'
    assert _extract_name_from_json(html) == "Rock & Roll Band"


def test_extract_name_from_json_not_found():
    html = "<div>No JSON here</div>"
    assert _extract_name_from_json(html) is None


def test_extract_name_from_html_og_title():
    html = '<meta property="og:title" content="Death Star™" />'
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_name_from_html(soup) == "Death Star™"


def test_extract_name_from_html_h1():
    html = "<h1>  Star   Destroyer  </h1>"
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_name_from_html(soup) == "Star Destroyer"


def test_extract_name_from_html_h2():
    html = "<h2>AT-AT Walker</h2>"
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_name_from_html(soup) == "AT-AT Walker"


def test_extract_name_from_html_not_found():
    html = "<div>No title here</div>"
    soup = BeautifulSoup(html, "html.parser")
    assert _extract_name_from_html(soup) is None


def test_extract_theme_from_json():
    html = '"themeName":"LEGO® Star Wars™"'
    assert _extract_theme_from_json(html) == "LEGO® Star Wars™"


def test_extract_theme_from_json_with_unicode_escapes():
    html = r'"themeName":"LEGO® Friends \u0026 Family"'
    assert _extract_theme_from_json(html) == "LEGO® Friends & Family"


def test_extract_theme_from_json_not_found():
    html = "<div>No JSON here</div>"
    assert _extract_theme_from_json(html) is None


def test_extract_age_from_json():
    html = '"ageRating":"6+"'
    assert _extract_age_from_json(html) == "6+"


def test_extract_age_from_json_with_decimal():
    html = '"ageRating":"1.5+"'
    assert _extract_age_from_json(html) == "1.5+"


def test_extract_age_from_json_not_found():
    html = "<div>No JSON here</div>"
    assert _extract_age_from_json(html) is None


def test_extract_age_from_text_simple():
    text = "Ages 9+ years"
    assert _extract_age_from_text(text) == "9+"


def test_extract_age_from_text_no_label():
    text = "Recommended for 6+ builders"
    assert _extract_age_from_text(text) == "6+"


def test_extract_age_from_text_decimal():
    text = "Ages 1.5+"
    assert _extract_age_from_text(text) == "1.5+"


def test_extract_age_from_text_first_occurrence():
    text = "Ages 6+ recommended, but 18+ prefer it"
    assert _extract_age_from_text(text) == "6+"


def test_extract_age_from_text_not_found():
    text = "No age information"
    assert _extract_age_from_text(text) is None


def test_extract_pieces_from_json():
    html = '"setPieceCount":"74"'
    assert _extract_pieces_from_json(html) == 74


def test_extract_pieces_from_json_large_number():
    html = '"setPieceCount":"9023"'
    assert _extract_pieces_from_json(html) == 9023


def test_extract_pieces_from_json_not_found():
    html = "<div>No JSON here</div>"
    assert _extract_pieces_from_json(html) is None


def test_extract_pieces_from_text():
    text = "This set contains 1,083 pieces and is great"
    assert _extract_pieces_from_text(text) == 1083


def test_extract_pieces_from_text_pcs():
    text = "74 pcs"
    assert _extract_pieces_from_text(text) == 74


def test_extract_pieces_from_text_no_comma():
    text = "9023 pieces"
    assert _extract_pieces_from_text(text) == 9023


def test_extract_pieces_from_text_not_found():
    text = "No piece information"
    assert _extract_pieces_from_text(text) is None


def test_extract_year_from_json():
    html = '"year":"2025"'
    assert _extract_year_from_json(html) == 2025


def test_extract_year_from_json_not_found():
    html = "<div>No JSON here</div>"
    assert _extract_year_from_json(html) is None


def test_extract_year_from_text_with_label():
    text = "Released in Year: 2024"
    assert _extract_year_from_text(text) == 2024


def test_extract_year_from_text_no_label():
    text = "Released in 2023"
    assert _extract_year_from_text(text) == 2023


def test_extract_year_from_text_nineteen_hundreds():
    text = "Classic set from 1999"
    assert _extract_year_from_text(text) == 1999


def test_extract_year_from_text_out_of_range():
    text = "In the year 2150"
    assert _extract_year_from_text(text) is None


def test_extract_year_from_text_not_found():
    text = "No year information"
    assert _extract_year_from_text(text) is None


# Integration tests for parse_set_metadata


def test_parse_set_metadata_with_json():
    """Test parsing when JSON data is available."""
    html = """
    <html>
    <script>
    {"name":"Millennium Falcon™ Mini-Build","setNumber":"30708","year":"2025","ageRating":"6+","setPieceCount":"74"}
    {"themeName":"LEGO® Star Wars™"}
    </script>
    <div>999 pieces</div>
    </html>
    """
    meta = parse_set_metadata(html)
    assert meta["name"] == "Millennium Falcon™ Mini-Build"
    assert meta["theme"] == "LEGO® Star Wars™"
    assert meta["age"] == "6+"
    assert meta["pieces"] == 74  # From JSON, not text (999)
    assert meta["year"] == 2025


def test_parse_set_metadata_fallback_to_html():
    """Test parsing when JSON is not available, using HTML fallbacks."""
    html = """
    <html>
    <meta property="og:title" content="Death Star™" />
    <div>Ages 18+ · 9,023 pieces · Year: 2025</div>
    </html>
    """
    meta = parse_set_metadata(html)
    assert meta["name"] == "Death Star™"
    assert meta["age"] == "18+"
    assert meta["pieces"] == 9023
    assert meta["year"] == 2025
    assert "theme" not in meta  # No theme in HTML fallback


def test_parse_set_metadata_mixed_sources():
    """Test parsing with some fields from JSON, others from HTML."""
    html = """
    <html>
    <script>{"name":"X-Wing","setNumber":"12345","ageRating":"9+"}</script>
    <meta property="og:title" content="Ignored because JSON has name" />
    <div>500 pieces · 2024</div>
    </html>
    """
    meta = parse_set_metadata(html)
    assert meta["name"] == "X-Wing"  # From JSON
    assert meta["age"] == "9+"  # From JSON
    assert meta["pieces"] == 500  # From text
    assert meta["year"] == 2024  # From text


def test_parse_set_metadata_minimal():
    """Test parsing with minimal information available."""
    html = "<html><h1>Simple Set</h1></html>"
    meta = parse_set_metadata(html)
    assert meta["name"] == "Simple Set"
    assert "age" not in meta
    assert "pieces" not in meta
    assert "year" not in meta
    assert "theme" not in meta


def test_parse_set_metadata_empty():
    """Test parsing with no recognizable metadata."""
    html = "<html><body>Nothing useful</body></html>"
    meta = parse_set_metadata(html)
    assert meta == {}
