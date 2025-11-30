#!/usr/bin/env python3
"""Generate golden files for FontSizeHints and PageHintCollection tests.

This script generates hint files for all element IDs that have *_raw.json fixtures.
It prefers using *_raw.json.bz2 files when available, falling back to PDFs in
data/$set_id/*.pdf.

Usage:
    pants run src/build_a_long/pdf_extract/classifier/tools/generate_golden_hints.py
"""

import json
import logging
import re
import sys
from pathlib import Path

import pymupdf

from build_a_long.pdf_extract.classifier.pages import PageHintCollection
from build_a_long.pdf_extract.classifier.text import FontSizeHints
from build_a_long.pdf_extract.cli.io import load_json
from build_a_long.pdf_extract.extractor import ExtractionResult, extract_page_data

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def extract_element_id(filename: str) -> str | None:
    """Extract element ID from a fixture filename.

    An element ID identifies a LEGO element (manual, piece, sticker sheet, etc.).

    Args:
        filename: Fixture filename like '6509377_page_013_raw.json'

    Returns:
        The element ID (e.g., '6509377') or None if not found
    """
    match = re.match(r"^(\d+)", filename)
    return match.group(1) if match else None


def find_pdf_for_element(element_id: str, data_dir: Path) -> Path | None:
    """Find a PDF file for the given element ID in the data directory.

    Searches data/$set_id/$element_id.pdf for matching PDFs.

    Args:
        element_id: The element ID (e.g., "6433200")
        data_dir: Root data directory

    Returns:
        Path to the PDF if found, None otherwise
    """
    # Search all subdirectories of data/ for matching PDF
    for set_dir in data_dir.iterdir():
        if not set_dir.is_dir():
            continue
        pdf_path = set_dir / f"{element_id}.pdf"
        if pdf_path.exists():
            return pdf_path
    return None


def find_all_required_element_ids(fixtures_dir: Path) -> set[str]:
    """Find all element IDs that need hints.

    This scans for:
    1. *_raw.json files (individual page fixtures used by generate-golden-files)
    2. *_raw.json.bz2 files (full document fixtures used by font_size_hints_test)

    Args:
        fixtures_dir: Path to fixtures directory

    Returns:
        Set of all element IDs that require hints
    """
    required_ids: set[str] = set()

    # Find element IDs from _raw.json files (what generate-golden-files uses)
    for fixture_file in fixtures_dir.glob("*_raw.json"):
        element_id = extract_element_id(fixture_file.name)
        if element_id:
            required_ids.add(element_id)

    # Also find element IDs from _raw.json.bz2 files (full document fixtures)
    for fixture_file in fixtures_dir.glob("*_raw.json.bz2"):
        element_id = extract_element_id(fixture_file.name)
        if element_id:
            required_ids.add(element_id)

    return required_ids


def find_bz2_fixtures(fixtures_dir: Path) -> dict[str, Path]:
    """Find all *_raw.json.bz2 fixtures and map element IDs to paths.

    Args:
        fixtures_dir: Path to fixtures directory

    Returns:
        Dict mapping element ID to bz2 fixture path
    """
    bz2_map: dict[str, Path] = {}
    for bz2_file in fixtures_dir.glob("*_raw.json.bz2"):
        element_id = extract_element_id(bz2_file.name)
        if element_id:
            bz2_map[element_id] = bz2_file
    return bz2_map


def write_hints(
    fixtures_dir: Path,
    element_id: str,
    font_hints: FontSizeHints,
    page_hints: PageHintCollection,
) -> None:
    """Write font and page hints to fixture files.

    Args:
        fixtures_dir: Path to fixtures directory
        element_id: The element ID
        font_hints: FontSizeHints to write
        page_hints: PageHintCollection to write
    """
    font_hints_path = fixtures_dir / f"{element_id}_font_hints_expected.json"
    page_hints_path = fixtures_dir / f"{element_id}_page_hints_expected.json"

    font_hints_data = font_hints.model_dump()
    font_hints_path.write_text(
        json.dumps(font_hints_data, indent=2, sort_keys=True) + "\n"
    )
    log.info(f"    Wrote {font_hints_path.name}")

    page_hints_data = page_hints.model_dump(mode="json")
    page_hints_path.write_text(
        json.dumps(page_hints_data, indent=2, sort_keys=True) + "\n"
    )
    log.info(f"    Wrote {page_hints_path.name}")


def generate_hints_from_bz2(fixtures_dir: Path, bz2_path: Path) -> bool:
    """Generate hints from a *_raw.json.bz2 fixture.

    Args:
        fixtures_dir: Path to fixtures directory
        bz2_path: Path to the bz2 fixture file

    Returns:
        True if successful, False otherwise
    """
    element_id = extract_element_id(bz2_path.name)
    if not element_id:
        return False

    log.info(f"  {bz2_path.name} (from bz2)...")

    # Load the input fixture
    json_data = load_json(bz2_path)
    extraction: ExtractionResult = ExtractionResult.model_validate_json(
        json.dumps(json_data)
    )  # type: ignore[assignment]

    # Generate hints
    font_hints = FontSizeHints.from_pages(extraction.pages)
    page_hints = PageHintCollection.from_pages(extraction.pages)

    # Write hints
    write_hints(fixtures_dir, element_id, font_hints, page_hints)
    return True


def generate_hints_from_pdf(
    fixtures_dir: Path, element_id: str, pdf_path: Path
) -> bool:
    """Generate hints from a PDF file.

    Args:
        fixtures_dir: Path to fixtures directory
        element_id: The element ID
        pdf_path: Path to the PDF file

    Returns:
        True if successful, False otherwise
    """
    log.info(f"  {pdf_path.name} (from {pdf_path.parent.name}/)...")

    # Extract pages from PDF
    doc = pymupdf.open(pdf_path)
    pages = extract_page_data(doc)
    doc.close()

    # Generate hints
    font_hints = FontSizeHints.from_pages(pages)
    page_hints = PageHintCollection.from_pages(pages)

    # Write hints
    write_hints(fixtures_dir, element_id, font_hints, page_hints)
    return True


def main() -> None:
    """Generate golden files for FontSizeHints and PageHintCollection."""

    fixtures_dir = Path("src/build_a_long/pdf_extract/fixtures")
    data_dir = Path("data")

    if not fixtures_dir.exists():
        log.error(f"Fixtures directory not found: {fixtures_dir}")
        sys.exit(1)

    log.info(f"Writing golden files to: {fixtures_dir.absolute()}")

    # Find all element IDs that generate-golden-files will need
    required_ids = find_all_required_element_ids(fixtures_dir)
    log.info(f"Found {len(required_ids)} element IDs required by generate-golden-files")

    # Find available bz2 fixtures
    bz2_fixtures = find_bz2_fixtures(fixtures_dir)

    # Generate hints for all required element IDs
    log.info("Generating hints...")
    success_count = 0
    missing_sources: list[str] = []

    for element_id in sorted(required_ids):
        # Prefer bz2 fixture if available
        if element_id in bz2_fixtures and generate_hints_from_bz2(
            fixtures_dir, bz2_fixtures[element_id]
        ):
            success_count += 1
            continue

        # Fall back to PDF
        pdf_path = find_pdf_for_element(element_id, data_dir)
        if pdf_path and generate_hints_from_pdf(fixtures_dir, element_id, pdf_path):
            success_count += 1
            continue

        # No source available
        missing_sources.append(element_id)

    if missing_sources:
        log.error("")
        log.error("=" * 70)
        log.error("ERROR: Cannot generate hints - missing sources")
        log.error("")
        log.error("The following element IDs have *_raw.json fixtures but")
        log.error("no corresponding *_raw.json.bz2 or PDF file:")
        log.error("")
        for element_id in sorted(missing_sources):
            log.error(f"  - {element_id}")
        log.error("")
        log.error("To fix this, either:")
        log.error("")
        log.error("  Option 1: Create a bz2 fixture")
        log.error("    pants run src/build_a_long/pdf_extract:main -- \\")
        log.error("      <pdf_path> --output-dir <fixtures_dir> \\")
        log.error("      --debug-json --debug-extra-json --compress-json")
        log.error("")
        log.error("  Option 2: Download the PDF")
        log.error("    pants run src/build_a_long/downloader:main -- <set_id>")
        log.error("    (PDF should be saved to: data/<set_id>/<element_id>.pdf)")
        log.error("")
        log.error("=" * 70)
        sys.exit(1)

    log.info(
        f"✓ Generated {success_count * 2} golden files ({success_count} element IDs)"
    )


if __name__ == "__main__":
    main()
