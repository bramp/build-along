"""Tests for set_sources module."""

import gzip
import io

import httpx
import pytest

from build_a_long.downloader.set_sources import (
    build_product_sitemap_url,
    fetch_lego_sitemap_sets,
    fetch_rebrickable_sets,
    format_locale_for_sitemap,
)


def test_format_locale_for_sitemap():
    assert format_locale_for_sitemap("en-us") == "en-US"
    assert format_locale_for_sitemap("en-gb") == "en-GB"
    assert format_locale_for_sitemap("de-de") == "de-DE"
    assert format_locale_for_sitemap("custom") == "custom"


def test_build_product_sitemap_url():
    url = build_product_sitemap_url("en-us", base="https://www.lego.com")
    assert url == "https://www.lego.com/sitemap-productPage-en-US0.xml"


def test_fetch_lego_sitemap_sets():
    xml_content = """<?xml version="1.0" encoding="UTF-8"?>
    <urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
      <url><loc>https://www.lego.com/en-us/product/arctic-animals-science-kit-45203</loc></url>
      <url><loc>https://www.lego.com/en-us/product/5006794</loc></url>
      <url><loc>https://www.lego.com/en-us/product/grogu-with-hover-pram-75403</loc></url>
      <url><loc>https://www.lego.com/en-us/product/non-numeric-slug</loc></url>
    </urlset>
    """

    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://www.lego.com/sitemap-productPage-en-US0.xml"
        return httpx.Response(200, text=xml_content)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    sets = fetch_lego_sitemap_sets(client, locale="en-us")

    assert "45203" in sets
    assert "5006794" in sets
    assert "75403" in sets
    assert "non-numeric-slug" not in sets
    assert len(sets) == 3


def test_fetch_rebrickable_sets():
    csv_data = """set_num,name,year,theme_id,num_parts,img_url
75419-1,Death Star,2024,1,100,https://example.com/75419.jpg
10210-1,Imperial Flagship,2010,2,1664,https://example.com/10210.jpg
invalid_id-1,Invalid,2020,3,10,https://example.com/inv.jpg
"""
    compressed = gzip.compress(csv_data.encode("utf-8"))

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=compressed)

    client = httpx.Client(transport=httpx.MockTransport(handler))

    # All sets
    all_sets = fetch_rebrickable_sets(client)
    assert all_sets == ["10210", "75419"]

    # Filtered by min_year
    recent_sets = fetch_rebrickable_sets(client, min_year=2020)
    assert recent_sets == ["75419"]
