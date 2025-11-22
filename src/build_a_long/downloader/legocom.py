"""LEGO.com website parsing utilities.

This module contains logic specific to parsing LEGO.com instruction pages,
including URL construction, PDF extraction, and metadata parsing.
"""

import json
import logging
import re
from contextlib import suppress
from typing import Any
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from pydantic import AnyUrl

from build_a_long.downloader.models import DownloadUrl
from build_a_long.downloader.util import extract_filename_from_url
from build_a_long.schemas import (
    InstructionMetadata,
    PdfEntry,
)

LEGO_BASE = "https://www.lego.com"

log = logging.getLogger(__name__)


def build_instructions_url(set_number: str, locale: str = "en-us") -> str:
    """Build the LEGO instructions page URL for a given set and locale."""
    return f"{LEGO_BASE}/{locale}/service/building-instructions/{set_number}"


def _extract_next_data(html: str, debug: bool = False) -> dict[str, Any] | None:
    """Extracts the __NEXT_DATA__ JSON blob from the HTML."""
    soup = BeautifulSoup(html, "html.parser")
    script_tag = soup.find("script", id="__NEXT_DATA__")
    if script_tag and script_tag.string:
        try:
            data = json.loads(script_tag.string)
            if debug:
                print(json.dumps(data, indent=2))
            return data
        except json.JSONDecodeError as e:
            log.error(
                "Failed to parse __NEXT_DATA__ JSON: %s at line %d column %d",
                e.msg,
                e.lineno,
                e.colno,
            )
            pass
    return None


def _get_apollo_state(next_data: dict[str, Any]) -> dict[str, Any]:
    """Extract the Apollo state dictionary from the Next.js data."""
    return next_data.get("props", {}).get("pageProps", {}).get("__APOLLO_STATE__", {})


def _get_building_instruction_data(
    apollo_state: dict[str, Any],
) -> dict[str, Any] | None:
    # Find the main building instruction data entry
    for value in apollo_state.values():
        if (
            isinstance(value, dict)
            and value.get("__typename") == "CS_BuildingInstructionData"
        ):
            return value

    return None


def parse_set_metadata(
    html: str,
    set_number: str = "",
    locale: str = "",
    base: str = LEGO_BASE,
    debug: bool = False,
) -> InstructionMetadata:
    """Parse a LEGO instructions HTML page and extract set metadata.

    Args:
        html: The HTML content to parse
        set_number: Optional set number to include in metadata
        locale: Optional locale to include in metadata
        base: Base URL for resolving relative URLs
        debug: If True, enable debug output.
    """
    next_data = _extract_next_data(html, debug=debug)
    if not next_data:
        return InstructionMetadata(set=set_number, locale=locale)

    apollo_state = _get_apollo_state(next_data)
    if not apollo_state:
        return InstructionMetadata(set=set_number, locale=locale)

    bi_data = _get_building_instruction_data(apollo_state)
    if not bi_data or not bi_data.get("name"):  # Added check for bi_data.get("name")
        return InstructionMetadata(set=set_number, locale=locale)

    # Extract name
    name = bi_data.get("name")

    # Extract theme
    theme = None
    if bi_data.get("theme"):
        theme_ref = bi_data["theme"].get("id")
        if theme_ref and theme_ref in apollo_state:
            theme = apollo_state[theme_ref].get("themeName")

    # Extract age rating
    age = bi_data.get("ageRating")

    # Extract piece count
    pieces = None
    if bi_data.get("setPieceCount"):
        with suppress(ValueError, TypeError):
            pieces = int(bi_data["setPieceCount"])

    # Extract year
    year = None
    if bi_data.get("year"):
        with suppress(ValueError, TypeError):
            year = int(bi_data["year"])

    # Extract set image URL and resolve to absolute
    set_image_url = None
    if bi_data.get("setImage"):
        image_ref = bi_data["setImage"].get("id")
        if image_ref and image_ref in apollo_state:
            relative_url = apollo_state[image_ref].get("src")
            if relative_url:
                set_image_url = urljoin(base, relative_url)

    return InstructionMetadata(
        set=set_number,
        locale=locale,
        name=name,
        theme=theme,
        age=age,
        pieces=pieces,
        year=year,
        set_image_url=AnyUrl(set_image_url) if set_image_url else None,
    )


def _apollo_resolve(apollo_state: dict[str, Any], item_or_ref: Any) -> Any:
    """Resolve an Apollo reference to its concrete object.

    The Apollo cache often stores references like {"type": "id", "id": "some.key"}.
    This utility follows those references recursively until a non-reference value is
    reached, or a resolution cannot be performed. Non-dict values (e.g., strings)
    are returned unchanged.

    This guards against cycles by tracking visited ids.
    """
    if item_or_ref is None or not isinstance(item_or_ref, dict):
        return item_or_ref

    visited: set[str] = set()
    current: Any = item_or_ref
    while (
        isinstance(current, dict)
        and current.get("type", "id") == "id"
        and "id" in current
    ):
        ref_id = current["id"]
        if ref_id in visited:
            # Cycle detected; abort resolution
            return current
        visited.add(ref_id)
        resolved = apollo_state.get(ref_id)
        if resolved is None:
            return current
        current = resolved
    return current


def parse_instruction_pdf_urls_apollo(
    html: str, base: str = LEGO_BASE, debug: bool = False
) -> list[DownloadUrl]:
    """Parse instruction PDFs using the Apollo (__NEXT_DATA__) approach only.

    Returns an empty list if the expected Apollo structures are not present.
    Resolves relative URLs to absolute URLs using the provided base.
    """
    next_data = _extract_next_data(html, debug=debug)
    if not next_data:
        return []

    apollo_state = _get_apollo_state(next_data)
    if not apollo_state:
        return []

    bi_data = _get_building_instruction_data(apollo_state)
    if not bi_data:
        return []

    results: list[DownloadUrl] = []
    for item_or_ref in bi_data.get("buildingInstructions", []) or []:
        item = _apollo_resolve(apollo_state, item_or_ref)
        if not isinstance(item, dict):
            continue

        sequence_number: int | None = None
        sequence_total: int | None = None
        if sequence_ref := item.get("sequence"):
            sequence_data = _apollo_resolve(apollo_state, sequence_ref)
            if isinstance(sequence_data, dict):
                if "element" in sequence_data:
                    with suppress(ValueError, TypeError):
                        sequence_number = int(sequence_data["element"])
                if "total" in sequence_data:
                    with suppress(ValueError, TypeError):
                        sequence_total = int(sequence_data["total"])

        is_additional_info_booklet: bool | None = None
        if "isAdditionalInfoBooklet" in item:
            with suppress(ValueError, TypeError):
                is_additional_info_booklet = bool(item["isAdditionalInfoBooklet"])

        pdf = _apollo_resolve(apollo_state, item.get("pdf"))
        if not isinstance(pdf, dict):
            log.debug(
                "Skipping building instructions with invalid pdf data: %s\n%s",
                pdf,
                item,
            )
            continue

        pdf_url = _apollo_resolve(apollo_state, pdf.get("pdfUrl"))
        if not isinstance(pdf_url, str):
            log.debug(
                "Skipping building instructions with invalid pdf url: %s\n%s",
                pdf_url,
                pdf,
            )
            continue

        # Resolve relative URLs to absolute
        absolute_pdf_url = urljoin(base, pdf_url)

        cover_image = _apollo_resolve(apollo_state, pdf.get("coverImage"))
        preview_url = cover_image.get("src") if isinstance(cover_image, dict) else None

        # Resolve preview URL if present
        absolute_preview_url = urljoin(base, preview_url) if preview_url else None

        results.append(
            DownloadUrl(
                url=AnyUrl(absolute_pdf_url),
                sequence_number=sequence_number,
                sequence_total=sequence_total,
                preview_url=(
                    AnyUrl(absolute_preview_url) if absolute_preview_url else None
                ),
                is_additional_info_booklet=is_additional_info_booklet,
            )
        )

    return results


def parse_instruction_pdf_urls_fallback(html: str) -> list[DownloadUrl]:
    """Fallback parser that extracts LEGO instruction PDFs via regex.

    This is used when Apollo data is missing or incomplete. Preview images are not
    available via this mechanism, so preview_url will be None.
    """
    pattern = re.compile(
        r"https://www\.lego\.com/cdn/product-assets/product\.bi\.core\.pdf/\w+\.pdf"
    )
    urls_in_order: list[str] = []
    seen: set[str] = set()
    for m in pattern.finditer(html):
        u = m.group(0)
        if u not in seen:
            seen.add(u)
            urls_in_order.append(u)

    return [DownloadUrl(url=AnyUrl(u), preview_url=None) for u in urls_in_order]


def parse_instruction_pdf_urls(
    html: str, base: str = LEGO_BASE, debug: bool = False
) -> list[DownloadUrl]:
    """Parse an instructions HTML page and return instruction PDF URLs.

    Prefers Apollo/Next.js state when present. Falls back to regex scanning otherwise.
    """
    results = parse_instruction_pdf_urls_apollo(html, base=base, debug=debug)
    if results:
        return results
    return parse_instruction_pdf_urls_fallback(html)


def build_metadata(
    html: str,
    set_number: str,
    locale: str,
    base: str = LEGO_BASE,
    debug: bool = False,
) -> InstructionMetadata:
    """Construct a InstructionMetadata dataclass from the instructions HTML.

    Parses both the set fields and the ordered list of instruction PDFs.
    """
    metadata = parse_set_metadata(
        html, set_number=set_number, locale=locale, base=base, debug=debug
    )

    # If no name was found, it's a "not found" set, so don't look for PDFs.
    if not metadata.name:
        return metadata

    pdf_infos = parse_instruction_pdf_urls(html, base=base, debug=debug)

    # Add PDFs to the metadata
    metadata.pdfs = []
    for info in pdf_infos:
        filename = extract_filename_from_url(info.url)
        if filename is None:
            log.warning(
                "Could not extract filename from URL: %s. Skipping PDF.", info.url
            )
            continue

        metadata.pdfs.append(
            PdfEntry(
                url=info.url,
                sequence_number=info.sequence_number,
                sequence_total=info.sequence_total,
                filename=filename,
                preview_url=info.preview_url,
                is_additional_info_booklet=info.is_additional_info_booklet,
            )
        )

    return metadata
