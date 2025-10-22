"""LEGO.com website parsing utilities.

This module contains logic specific to parsing LEGO.com instruction pages,
including URL construction, PDF extraction, and metadata parsing.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from bs4 import BeautifulSoup

LEGO_BASE = "https://www.lego.com"


def build_instructions_url(set_number: str, locale: str = "en-us") -> str:
    """Build the LEGO instructions page URL for a given set and locale."""
    return f"{LEGO_BASE}/{locale}/service/building-instructions/{set_number}"


def parse_instruction_pdf_urls(html: str, base: str = LEGO_BASE) -> List[str]:
    """Parse an instructions HTML page and return absolute instruction PDF URLs.

    - Collects links ending with .pdf
    - Normalizes relative links to absolute using `base`
    - Filters to instruction manuals (product.bi.core.pdf)
    - Deduplicates while preserving order
    """
    soup = BeautifulSoup(html, "html.parser")

    pdfs: List[str] = []
    for a in soup.find_all("a", href=True):
        href = a.get("href")
        if href and isinstance(href, str) and href.lower().endswith(".pdf"):
            if href.startswith("/"):
                href = f"{base}{href}"
            pdfs.append(href)

    seen = set()
    unique_pdfs: List[str] = []
    for u in pdfs:
        if u not in seen and "product.bi.core.pdf" in u:
            seen.add(u)
            unique_pdfs.append(u)
    return unique_pdfs


def _clean_text(text: str) -> str:
    """Remove extra whitespace from text."""
    return re.sub(r"\s+", " ", text).strip()


def _extract_name_from_json(html: str) -> Optional[str]:
    """Extract set name from JSON data in HTML.

    Looks for: "name":"Millennium Falcon™ Mini-Build","setNumber":"..."
    """
    match = re.search(r'"name"\s*:\s*"([^"]+)"\s*,\s*"setNumber"', html)
    return match.group(1) if match else None


def _extract_name_from_html(soup: BeautifulSoup) -> Optional[str]:
    """Extract set name from HTML meta tags or headers.

    Tries og:title meta tag first, then falls back to H1/H2 elements.
    """
    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title is not None:
        content = og_title.get("content")
        if isinstance(content, str):
            title = content.strip()
            if title:
                return title

    h1 = soup.find(["h1", "h2"])
    if h1 and h1.get_text():
        return _clean_text(h1.get_text())

    return None


def _extract_theme_from_json(html: str) -> Optional[str]:
    """Extract theme name from JSON data in HTML.

    Looks for: "themeName":"LEGO® Star Wars™"
    """
    match = re.search(r'"themeName"\s*:\s*"([^"]+)"', html)
    return match.group(1) if match else None


def _extract_age_from_json(html: str) -> Optional[str]:
    """Extract age rating from JSON data in HTML.

    Looks for: "ageRating":"6+"
    """
    match = re.search(r'"ageRating"\s*:\s*"([^"]+)"', html)
    return match.group(1) if match else None


def _extract_age_from_text(text: str) -> Optional[str]:
    """Extract age rating from visible text.

    Looks for patterns like "Ages 9+", "9+ years", "6+".
    Returns the first occurrence found.
    """
    candidates = re.findall(
        r"(?:Ages?\s*)?(\d{1,2}\.?\d*\+)(?:\s*years?)?", text, re.IGNORECASE
    )
    return candidates[0] if candidates else None


def _extract_pieces_from_json(html: str) -> Optional[int]:
    """Extract piece count from JSON data in HTML.

    Looks for: "setPieceCount":"74"
    """
    match = re.search(r'"setPieceCount"\s*:\s*"(\d+)"', html)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def _extract_pieces_from_text(text: str) -> Optional[int]:
    """Extract piece count from visible text.

    Looks for patterns like "1,083 pieces", "74 pcs".
    Returns the integer piece count.
    """
    match = re.search(
        r"(\d{1,5}(?:,\d{3})*)(?:\s*)(?:pcs|pieces)\b", text, re.IGNORECASE
    )
    if match:
        try:
            return int(match.group(1).replace(",", ""))
        except ValueError:
            pass
    return None


def _extract_year_from_json(html: str) -> Optional[int]:
    """Extract year from JSON data in HTML.

    Looks for: "year":"2025"
    """
    match = re.search(r'"year"\s*:\s*"(\d{4})"', html)
    if match:
        try:
            return int(match.group(1))
        except ValueError:
            pass
    return None


def _extract_year_from_text(text: str) -> Optional[int]:
    """Extract year from visible text.

    Prefers years near the label "Year:", otherwise finds any plausible year
    between 1970 and 2099.
    """
    # Try to find year near "Year" label first
    year_label_match = re.search(
        r"Year[^\d]{0,10}(20\d{2}|19\d{2})", text, re.IGNORECASE
    )
    year_match: Optional[re.Match[str]] = year_label_match or re.search(
        r"\b(20\d{2}|19\d{2})\b", text
    )
    if year_match:
        try:
            year = int(year_match.group(1))
            if 1970 <= year <= 2099:
                return year
        except ValueError:
            pass
    return None


def parse_set_metadata(html: str) -> Dict[str, Any]:
    """Parse a LEGO instructions HTML page and extract set metadata.

    Attempts to extract commonly available fields and is resilient to
    missing data and layout changes.

    Extracted fields when available:
    - name: Name/title of the set (from JSON or og:title meta or first H1)
    - theme: Theme name (e.g., "LEGO® Star Wars™")
    - age: Displayed age range (e.g., "9+")
    - pieces: Integer piece count
    - year: Integer year (e.g., 2024)

    Returns a dict with any fields found; absent fields are omitted.
    """
    soup = BeautifulSoup(html, "html.parser")
    meta: Dict[str, Any] = {}

    # Extract name: try JSON first, then HTML fallback
    name = _extract_name_from_json(html)
    if not name:
        name = _extract_name_from_html(soup)
    if name:
        meta["name"] = name

    # Extract theme: JSON only (no fallback available)
    theme = _extract_theme_from_json(html)
    if theme:
        meta["theme"] = theme

    # Extract age: try JSON first, then text fallback
    age = _extract_age_from_json(html)
    if not age:
        full_text = _clean_text(soup.get_text(" "))
        age = _extract_age_from_text(full_text)
    if age:
        meta["age"] = age

    # Extract pieces: try JSON first, then text fallback
    pieces = _extract_pieces_from_json(html)
    if pieces is None:
        full_text = _clean_text(soup.get_text(" "))
        pieces = _extract_pieces_from_text(full_text)
    if pieces is not None:
        meta["pieces"] = pieces

    # Extract year: try JSON first, then text fallback
    year = _extract_year_from_json(html)
    if not year:
        full_text = _clean_text(soup.get_text(" "))
        year = _extract_year_from_text(full_text)
    if year is not None:
        meta["year"] = year

    return meta


@dataclass
class PdfEntry:
    """Represents a single instruction PDF file."""

    url: str
    filename: str


@dataclass
class Metadata:
    """Complete metadata for a LEGO set's instructions."""

    set: str
    locale: str
    name: Optional[str] = None
    theme: Optional[str] = None
    age: Optional[str] = None
    pieces: Optional[int] = None
    year: Optional[int] = None
    pdfs: List[PdfEntry] = field(default_factory=list)


def build_metadata(
    html: str, set_number: str, locale: str, base: str = LEGO_BASE
) -> Metadata:
    """Construct a Metadata dataclass from the instructions HTML.

    Parses both the set fields and the ordered list of instruction PDFs.
    """
    pdf_urls = parse_instruction_pdf_urls(html, base=base)
    fields = parse_set_metadata(html)
    return Metadata(
        set=set_number,
        locale=locale,
        name=fields.get("name"),
        theme=fields.get("theme"),
        age=fields.get("age"),
        pieces=fields.get("pieces"),
        year=fields.get("year"),
        pdfs=[PdfEntry(url=u, filename=u.split("/")[-1]) for u in pdf_urls],
    )
