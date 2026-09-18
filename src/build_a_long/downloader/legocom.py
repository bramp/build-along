"""Unified LEGO.com metadata retrieval with GraphQL primary and HTML fallback."""

from __future__ import annotations

import logging
from typing import Any

import httpx

from build_a_long.downloader.legocom_graphql import (
    fetch_set_data_graphql,
    format_locale_for_header,
    get_graphql_auth_token,
    parse_metadata_from_graphql,
)
from build_a_long.downloader.legocom_html import (
    LEGO_BASE,
    _extract_next_data,
    _fix_url_encoding_issues,
    _get_apollo_state,
    _get_building_instruction_data,
    build_instructions_url,
    build_metadata_from_html,
    parse_instruction_pdf_urls,
    parse_instruction_pdf_urls_apollo,
    parse_instruction_pdf_urls_fallback,
    parse_set_metadata,
)
from build_a_long.downloader.models import (
    DownloadUrl,
    ImageEntry,
    InstructionMetadata,
    PdfEntry,
    VideoEntry,
)

log = logging.getLogger(__name__)

__all__ = [
    "LEGO_BASE",
    "DownloadUrl",
    "ImageEntry",
    "InstructionMetadata",
    "PdfEntry",
    "VideoEntry",
    "_extract_next_data",
    "_fix_url_encoding_issues",
    "_get_apollo_state",
    "_get_building_instruction_data",
    "build_instructions_url",
    "build_metadata",
    "build_metadata_from_html",
    "fetch_metadata",
    "fetch_set_data_graphql",
    "format_locale_for_header",
    "get_graphql_auth_token",
    "parse_instruction_pdf_urls",
    "parse_instruction_pdf_urls_apollo",
    "parse_instruction_pdf_urls_fallback",
    "parse_metadata_from_graphql",
    "parse_set_metadata",
]


def build_metadata(
    html: str,
    set_number: str,
    locale: str,
    base: str = LEGO_BASE,
    debug: bool = False,
) -> InstructionMetadata:
    """Build metadata from HTML content (backward compatibility alias)."""
    return build_metadata_from_html(
        html=html,
        set_number=set_number,
        locale=locale,
        base=base,
        debug=debug,
    )


def fetch_metadata(
    client: httpx.Client,
    set_number: str,
    locale: str = "en-us",
    base: str = LEGO_BASE,
    debug: bool = False,
) -> InstructionMetadata | None:
    """Fetch complete metadata for a LEGO set.

    First attempts the fast GraphQL API (which returns instructions + full descriptions,
    gallery images, videos, and categories). If GraphQL fails or returns no data,
    falls back to fetching the building instructions HTML page and parsing __NEXT_DATA__.

    Args:
        client: HTTP client.
        set_number: The LEGO set number.
        locale: The locale to use.
        base: The LEGO.com base URL.
        debug: Whether to print debug information.

    Returns:
        The InstructionMetadata object, or None if the set does not exist.
    """
    # 1. Try GraphQL first
    try:
        gql_data = fetch_set_data_graphql(
            client=client,
            set_number=set_number,
            locale=locale,
            base=base,
        )
        if gql_data:
            meta = parse_metadata_from_graphql(
                gql_data,
                set_number=set_number,
                locale=locale,
                base=base,
            )
            if meta is not None:
                if debug:
                    log.debug(
                        "Successfully fetched metadata via GraphQL for %s", set_number
                    )
                return meta
    except Exception as e:
        log.debug(
            "GraphQL fetch failed for set %s, falling back to HTML: %s", set_number, e
        )

    # 2. Fallback to HTML
    try:
        url = build_instructions_url(set_number, locale=locale)
        resp = client.get(url)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        meta = build_metadata_from_html(
            html=resp.text,
            set_number=set_number,
            locale=locale,
            base=base,
            debug=debug,
        )
        if meta and meta.name:
            return meta
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 404:
            return None
        raise
    except Exception as e:
        log.debug("HTML fallback failed for set %s: %s", set_number, e)

    return None
