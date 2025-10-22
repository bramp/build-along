from typing import List

from bs4 import BeautifulSoup


LEGO_BASE = "https://www.lego.com"


def is_valid_set_id(set_id: str) -> bool:
    """
    Validates if a string is a valid LEGO set ID (only digits).
    """
    return set_id.isdigit()


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

    # TODO With each URL extracted, try and save extra metadata, specifically
    # For a single set, there will be multiple PDFs. Store this information
    # in a metadata json file, in the same directory as the PDFs.

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
