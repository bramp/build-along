"""Set list sources for LEGO set discovery (LEGO.com sitemap and Rebrickable)."""

from __future__ import annotations

import csv
import gzip
import io
import re
import sys
from typing import Final

import httpx

from build_a_long.downloader.util import is_valid_set_id

LEGO_BASE: Final[str] = "https://www.lego.com"
REBRICKABLE_SETS_URL: Final[str] = (
    "https://cdn.rebrickable.com/media/downloads/sets.csv.gz"
)

# Matches product page URLs like:
# https://www.lego.com/en-us/product/arctic-animals-science-kit-45203
# https://www.lego.com/en-us/product/5006794
# Capturing the numeric set ID at the end.
_PRODUCT_URL_SET_REGEX = re.compile(
    r"<loc>(?:https?://[^<]*/product/)?(?:[a-zA-Z0-9_-]+-)?(\d+)</loc>"
)


def format_locale_for_sitemap(locale: str) -> str:
    """Format locale into the sitemap file naming convention (e.g., 'en-us' -> 'en-US')."""
    parts = locale.split("-")
    if len(parts) == 2:
        return f"{parts[0].lower()}-{parts[1].upper()}"
    return locale


def build_product_sitemap_url(locale: str = "en-us", base: str = LEGO_BASE) -> str:
    """Build the URL for the LEGO.com product sitemap for a given locale."""
    locale_tag = format_locale_for_sitemap(locale)
    return f"{base.rstrip('/')}/sitemap-productPage-{locale_tag}0.xml"


def fetch_lego_sitemap_sets(
    client: httpx.Client,
    locale: str = "en-us",
    base: str = LEGO_BASE,
) -> list[str]:
    """Fetch all LEGO set IDs from the official LEGO.com product sitemap.

    This makes a single HTTP GET request to the sitemap XML and extracts set IDs.

    Args:
        client: The HTTP client to use.
        locale: The LEGO locale (e.g., 'en-us').
        base: The base LEGO URL.

    Returns:
        Sorted list of unique LEGO set numbers found in the sitemap.
    """
    sitemap_url = build_product_sitemap_url(locale=locale, base=base)
    response = client.get(sitemap_url)
    response.raise_for_status()

    set_ids: set[str] = set()
    for match in _PRODUCT_URL_SET_REGEX.finditer(response.text):
        candidate = match.group(1)
        if is_valid_set_id(candidate):
            set_ids.add(candidate)

    return sorted(set_ids, key=lambda x: (len(x), x))


def fetch_rebrickable_sets(
    client: httpx.Client,
    min_year: int | None = None,
    url: str = REBRICKABLE_SETS_URL,
) -> list[str]:
    """Fetch all LEGO set IDs from Rebrickable's open database export (sets.csv.gz).

    Args:
        client: The HTTP client to use.
        min_year: Optional minimum release year to filter by.
        url: URL of the gzipped CSV export.

    Returns:
        Sorted list of unique LEGO set numbers found in the export.
    """
    response = client.get(url)
    response.raise_for_status()

    decompressed = gzip.decompress(response.content).decode("utf-8")
    reader = csv.DictReader(io.StringIO(decompressed))

    set_ids: set[str] = set()
    for row in reader:
        set_num = row.get("set_num", "")
        # Rebrickable set numbers have variant suffix e.g. "75419-1"
        clean_id = set_num.split("-")[0].strip()
        if not is_valid_set_id(clean_id):
            continue

        if min_year is not None:
            try:
                row_year = int(row.get("year", "0"))
                if row_year < min_year:
                    continue
            except (ValueError, TypeError):
                continue

        set_ids.add(clean_id)

    return sorted(set_ids, key=lambda x: (len(x), x))
