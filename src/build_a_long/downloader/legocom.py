"""LEGO.com website parsing utilities.

This module contains logic specific to parsing LEGO.com instruction pages,
including URL construction, PDF extraction, and metadata parsing.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Set

from bs4 import BeautifulSoup

from build_a_long.downloader.metadata import DownloadUrl, Metadata, PdfEntry

LEGO_BASE = "https://www.lego.com"

log = logging.getLogger(__name__)


def build_instructions_url(set_number: str, locale: str = "en-us") -> str:
    """Build the LEGO instructions page URL for a given set and locale."""
    return f"{LEGO_BASE}/{locale}/service/building-instructions/{set_number}"


def _extract_next_data(html: str) -> Optional[Dict[str, Any]]:
    """Extracts the __NEXT_DATA__ JSON blob from the HTML."""
    soup = BeautifulSoup(html, "html.parser")
    script_tag = soup.find("script", id="__NEXT_DATA__")
    if script_tag and script_tag.string:
        try:
            return json.loads(script_tag.string)
        except json.JSONDecodeError:
            pass
    return None


def _get_apollo_state(next_data: Dict[str, Any]) -> Dict[str, Any]:
    return next_data.get("props", {}).get("pageProps", {}).get("__APOLLO_STATE__", {})


def _get_building_instruction_data(
    apollo_state: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    # Find the main building instruction data entry
    for value in apollo_state.values():
        if (
            isinstance(value, dict)
            and value.get("__typename") == "CS_BuildingInstructionData"
        ):
            return value

    return None


def parse_set_metadata(html: str, set_number: str = "", locale: str = "") -> Metadata:
    """Parse a LEGO instructions HTML page and extract set metadata.

    Args:
        html: The HTML content to parse
        set_number: Optional set number to include in metadata
        locale: Optional locale to include in metadata

    Returns:
        Metadata object with extracted fields, or minimal Metadata if parsing fails
    """
    next_data = _extract_next_data(html)
    if not next_data:
        return Metadata(set=set_number, locale=locale)

    apollo_state = _get_apollo_state(next_data)
    if not apollo_state:
        return Metadata(set=set_number, locale=locale)

    bi_data = _get_building_instruction_data(apollo_state)
    if not bi_data:
        return Metadata(set=set_number, locale=locale)

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
        try:
            pieces = int(bi_data["setPieceCount"])
        except (ValueError, TypeError):
            pass

    # Extract year
    year = None
    if bi_data.get("year"):
        try:
            year = int(bi_data["year"])
        except (ValueError, TypeError):
            pass

    # Extract set image URL
    set_image_url = None
    if bi_data.get("setImage"):
        image_ref = bi_data["setImage"].get("id")
        if image_ref and image_ref in apollo_state:
            set_image_url = apollo_state[image_ref].get("src")

    return Metadata(
        set=set_number,
        locale=locale,
        name=name,
        theme=theme,
        age=age,
        pieces=pieces,
        year=year,
        set_image_url=set_image_url,
    )


def _apollo_resolve(apollo_state: Dict[str, Any], item_or_ref: Any) -> Any:
    """Resolve an Apollo reference to its concrete object.

    The Apollo cache often stores references like {"type": "id", "id": "some.key"}.
    This utility follows those references recursively until a non-reference value is
    reached, or a resolution cannot be performed. Non-dict values (e.g., strings)
    are returned unchanged.

    This guards against cycles by tracking visited ids.
    """
    if item_or_ref is None or not isinstance(item_or_ref, dict):
        return item_or_ref

    visited: Set[str] = set()
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


def parse_instruction_pdf_urls_apollo(html: str) -> List[DownloadUrl]:
    """Parse instruction PDFs using the Apollo (__NEXT_DATA__) approach only.

    Returns an empty list if the expected Apollo structures are not present.
    """
    next_data = _extract_next_data(html)
    if not next_data:
        return []

    apollo_state = _get_apollo_state(next_data)
    if not apollo_state:
        return []

    bi_data = _get_building_instruction_data(apollo_state)
    if not bi_data:
        return []

    results: List[DownloadUrl] = []
    for item_or_ref in bi_data.get("buildingInstructions", []) or []:
        item = _apollo_resolve(apollo_state, item_or_ref)
        if not isinstance(item, dict):
            continue

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

        cover_image = _apollo_resolve(apollo_state, pdf.get("coverImage"))
        preview_url = cover_image.get("src") if isinstance(cover_image, dict) else None

        results.append(DownloadUrl(url=pdf_url, preview_url=preview_url))

    return results


def parse_instruction_pdf_urls_fallback(html: str) -> List[DownloadUrl]:
    """Fallback parser that extracts LEGO instruction PDFs via regex.

    This is used when Apollo data is missing or incomplete. Preview images are not
    available via this mechanism, so preview_url will be None.
    """
    pattern = re.compile(
        r"https://www\.lego\.com/cdn/product-assets/product\.bi\.core\.pdf/\w+\.pdf"
    )
    urls_in_order: List[str] = []
    seen: Set[str] = set()
    for m in pattern.finditer(html):
        u = m.group(0)
        if u not in seen:
            seen.add(u)
            urls_in_order.append(u)

    return [DownloadUrl(url=u, preview_url=None) for u in urls_in_order]


def parse_instruction_pdf_urls(html: str, base: str = LEGO_BASE) -> List[DownloadUrl]:
    """Parse an instructions HTML page and return instruction PDF URLs.

    Prefers Apollo/Next.js state when present. Falls back to regex scanning otherwise.
    """
    results = parse_instruction_pdf_urls_apollo(html)
    if results:
        return results
    return parse_instruction_pdf_urls_fallback(html)


def build_metadata(
    html: str, set_number: str, locale: str, base: str = LEGO_BASE
) -> Metadata:
    """Construct a Metadata dataclass from the instructions HTML.

    Parses both the set fields and the ordered list of instruction PDFs.
    """
    pdf_infos = parse_instruction_pdf_urls(html, base=base)
    metadata = parse_set_metadata(html, set_number=set_number, locale=locale)

    # Add PDFs to the metadata
    metadata.pdfs = [
        PdfEntry(
            url=info.url,
            filename=info.url.split("/")[-1],
            preview_url=info.preview_url,
        )
        for info in pdf_infos
    ]

    return metadata
