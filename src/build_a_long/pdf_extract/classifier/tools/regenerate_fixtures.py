#!/usr/bin/env python3
"""Regenerate all raw fixture files from source PDFs.

This script extracts pages from the source PDFs to create the raw fixture files
used by tests. Run this when you need to update fixtures after changes to the
extraction logic.

Fixtures are defined in fixtures.json5 (JSON5 format to allow comments).

Usage:
    pants run src/build_a_long/pdf_extract/classifier/tools/regenerate_fixtures.py
"""

import bz2
import logging
import sys
from pathlib import Path
from typing import cast

import json5
import pymupdf
from pydantic import BaseModel

from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.extractor.extractor import Extractor
from build_a_long.pdf_extract.parser import parse_page_ranges

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)


class Fixture(BaseModel):
    """Definition of a fixture to generate."""

    pdf: str
    """Path to the source PDF file."""

    description: str
    """Human-readable description of the fixture."""

    pages: str | None = None
    """Page range to extract (e.g., '10-17,180'). None means all pages."""

    compress: bool = False
    """Whether to compress the output with bz2."""


def load_fixtures(fixtures_path: Path) -> list[Fixture]:
    """Load fixture definitions from JSON5 file.

    Args:
        fixtures_path: Path to fixtures.json5

    Returns:
        List of Fixture objects
    """
    data = cast(dict, json5.loads(fixtures_path.read_text()))
    return [Fixture.model_validate(f) for f in data["fixtures"]]


def extract_pages(pdf_path: Path, page_numbers: list[int] | None) -> list:
    """Extract page data from a PDF.

    Args:
        pdf_path: Path to PDF file
        page_numbers: 1-indexed page numbers to extract, or None for all pages

    Returns:
        List of PageData objects
    """
    pages = []
    with pymupdf.open(str(pdf_path)) as doc:
        if page_numbers is None:
            page_numbers = list(range(1, len(doc) + 1))

        for page_num in page_numbers:
            page_index = page_num - 1
            if page_index < 0 or page_index >= len(doc):
                log.warning(f"  Skipping page {page_num} (out of range)")
                continue

            page = doc[page_index]
            extractor = Extractor(page=page, page_num=page_num, include_metadata=True)
            page_data = extractor.extract_page_data()
            pages.append(page_data)

    return pages


def save_fixture(
    pages: list,
    output_dir: Path,
    pdf_path: Path,
    *,
    compress: bool,
    per_page: bool,
) -> None:
    """Save extracted pages as fixture files.

    Args:
        pages: List of PageData objects
        output_dir: Directory to save fixtures
        pdf_path: Original PDF path (for naming)
        compress: Whether to compress with bz2
        per_page: Whether to save one file per page
    """
    element_id = pdf_path.stem

    if per_page:
        # Save each page as a separate file
        for page in pages:
            extraction = ExtractionResult(pages=[page])
            json_str = extraction.to_json()

            filename = f"{element_id}_page_{page.page_number:03d}_raw.json"
            output_path = output_dir / filename
            output_path.write_text(json_str)
            log.info(f"  Wrote {output_path}")
    else:
        # Save all pages in a single file
        extraction = ExtractionResult(pages=pages)
        json_str = extraction.to_json()

        if compress:
            filename = f"{element_id}_raw.json.bz2"
            output_path = output_dir / filename
            output_path.write_bytes(bz2.compress(json_str.encode("utf-8")))
        else:
            filename = f"{element_id}_raw.json"
            output_path = output_dir / filename
            output_path.write_text(json_str)

        log.info(f"  Wrote {output_path}")


def main() -> int:
    """Regenerate all fixture files."""
    # Locate fixtures.json5 relative to this script
    script_dir = Path(__file__).parent
    fixtures_json_path = script_dir / "fixtures.json5"

    if not fixtures_json_path.exists():
        log.error(f"Fixtures config not found: {fixtures_json_path}")
        return 1

    # TODO maybe be explicit with the full path of the directory
    fixtures_dir = Path("src/build_a_long/pdf_extract/fixtures")

    if not fixtures_dir.exists():
        log.error(f"Fixtures directory not found: {fixtures_dir}")
        return 1

    # Load fixture definitions
    fixtures = load_fixtures(fixtures_json_path)
    log.info(f"Loaded {len(fixtures)} fixture definitions from {fixtures_json_path}")
    log.info("")

    errors: list[str] = []

    for fixture in fixtures:
        log.info(f"=== {fixture.description} ===")

        pdf_path = Path(fixture.pdf)
        if not pdf_path.exists():
            log.error(f"  PDF not found: {pdf_path}")
            errors.append(str(pdf_path))
            continue

        # Parse page numbers using the existing parser
        page_numbers: list[int] | None = None
        if fixture.pages:
            page_ranges = parse_page_ranges(fixture.pages)
            # We need a document length for page_numbers(), but we don't have
            # it yet. Use a large number and filter later in extract_pages.
            page_numbers = list(page_ranges.page_numbers(10000))
            log.info(f"  Extracting pages: {fixture.pages}")
        else:
            log.info("  Extracting all pages")

        # Extract pages
        pages = extract_pages(pdf_path, page_numbers)
        log.info(f"  Extracted {len(pages)} page(s)")

        # Save fixture
        per_page = fixture.pages is not None
        save_fixture(
            pages,
            fixtures_dir,
            pdf_path,
            compress=fixture.compress,
            per_page=per_page,
        )

        log.info("")

    if errors:
        log.error(f"✗ Failed to process {len(errors)} PDF(s)")
        return 1

    log.info("✓ All raw fixtures regenerated successfully!")
    log.info("")
    log.info("Next steps:")
    log.info(
        "  1. Run: pants run src/build_a_long/pdf_extract/classifier/"
        "tools/generate_golden_hints.py"
    )
    log.info(
        "  2. Run: pants run src/build_a_long/pdf_extract/classifier/"
        "tools/generate_golden_files.py"
    )
    log.info("  3. Run: pants test ::")

    return 0


if __name__ == "__main__":
    sys.exit(main())
