#!/usr/bin/env python3
"""Regenerate all raw fixture files from source PDFs.

This script extracts pages from the source PDFs to create the raw fixture files
used by tests. Run this when you need to update fixtures after changes to the
extraction logic.

Fixtures are defined in index.json5 (JSON5 format to allow comments).

Usage:
    pants run src/build_a_long/pdf_extract/classifier/tools/regenerate_fixtures.py
"""

import bz2
import logging
import sys
from pathlib import Path

from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.tests.fixture_utils import (
    FixtureDefinition,
    extract_pages_from_pdf,
    load_fixture_definitions,
)

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Output directory for fixtures (relative to repo root when running outside sandbox)
FIXTURES_DIR = Path("src/build_a_long/pdf_extract/fixtures")


def save_fixture(
    fixture_def: FixtureDefinition,
    extracted_pages: dict[int, PageData],
    output_dir: Path,
) -> None:
    """Save extracted pages as fixture files.

    Args:
        fixture_def: The fixture definition
        extracted_pages: Dict mapping page number to PageData
        output_dir: Directory to save fixtures
    """
    if fixture_def.is_per_page:
        # Save each page as a separate file
        for page_num, page_data in sorted(extracted_pages.items()):
            extraction = ExtractionResult(pages=[page_data])
            json_str = extraction.to_json()

            filename = fixture_def.get_fixture_filename(page_num)
            output_path = output_dir / filename
            output_path.write_text(json_str)
            log.info(f"  Wrote {output_path}")
    else:
        # Save all pages in a single file
        pages = [extracted_pages[pn] for pn in sorted(extracted_pages.keys())]
        extraction = ExtractionResult(pages=pages)
        json_str = extraction.to_json()

        filename = fixture_def.get_fixture_filename()
        output_path = output_dir / filename

        if fixture_def.compress:
            output_path.write_bytes(bz2.compress(json_str.encode("utf-8")))
        else:
            output_path.write_text(json_str)

        log.info(f"  Wrote {output_path}")


def main() -> int:
    """Regenerate all fixture files."""
    if not FIXTURES_DIR.exists():
        log.error(f"Fixtures directory not found: {FIXTURES_DIR}")
        return 1

    # Load fixture definitions
    index_path = FIXTURES_DIR / "index.json5"
    if not index_path.exists():
        log.error(f"Index file not found: {index_path}")
        return 1

    fixtures = load_fixture_definitions(index_path)
    log.info(f"Loaded {len(fixtures)} fixture definitions from {index_path}")
    log.info("")

    errors: list[str] = []

    for fixture_def in fixtures:
        log.info(f"=== {fixture_def.description} ===")

        pdf_path = fixture_def.pdf_path
        if not pdf_path.exists():
            log.error(f"  PDF not found: {pdf_path}")
            errors.append(str(pdf_path))
            continue

        if fixture_def.pages:
            log.info(f"  Extracting pages: {fixture_def.pages}")
        else:
            log.info("  Extracting all pages")

        # Extract all needed pages in one pass (also resolves page ranges)
        extraction = extract_pages_from_pdf(pdf_path, fixture_def.pages)
        log.info(f"  Extracted {len(extraction.pages)} page(s)")

        # Save fixture(s)
        save_fixture(fixture_def, extraction.pages, FIXTURES_DIR)

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
