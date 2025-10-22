"""Tests for util.py - pure utility functions (pytest style)."""

from build_a_long.downloader.util import (
    LEGO_BASE,
    build_instructions_url,
    is_valid_set_id,
    parse_instruction_pdf_urls,
)


def test_is_valid_set_id_numeric():
    assert is_valid_set_id("12345")
    assert is_valid_set_id("75419")


def test_is_valid_set_id_invalid():
    assert not is_valid_set_id("abc123")
    assert not is_valid_set_id("invalid")
    assert not is_valid_set_id("")
    assert not is_valid_set_id("123-456")


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
