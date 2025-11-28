#!/usr/bin/env python3
"""Generate golden files for FontSizeHints tests.

This script runs FontSizeHints.from_pages on all bz2 fixtures and generates
golden output files.

Usage:
    pants run \\
        src/build_a_long/pdf_extract/classifier/tools:generate-font-hints-golden
"""

import json
import logging
import sys
from pathlib import Path

from build_a_long.pdf_extract.classifier.text import FontSizeHints
from build_a_long.pdf_extract.cli.io import load_json
from build_a_long.pdf_extract.extractor import ExtractionResult

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    """Generate golden files for FontSizeHints on all bz2 fixtures."""

    fixtures_dir = Path("src/build_a_long/pdf_extract/fixtures")

    if not fixtures_dir.exists():
        log.error(f"Fixtures directory not found: {fixtures_dir}")
        sys.exit(1)

    raw_fixtures = list(fixtures_dir.glob("*_raw.json.bz2"))

    if not raw_fixtures:
        log.error(f"No *_raw.json.bz2 fixtures found in {fixtures_dir}")
        sys.exit(1)

    log.info(f"Found {len(raw_fixtures)} raw bz2 fixtures")
    log.info(f"Writing golden files to: {fixtures_dir.absolute()}")

    for fixture_path in sorted(raw_fixtures):
        golden_path = fixture_path.with_name(
            fixture_path.name.replace("_raw.json.bz2", "_font_hints_expected.json")
        )

        log.info(f"Processing {fixture_path.name}...")

        # Load the input fixture
        json_data = load_json(fixture_path)
        extraction: ExtractionResult = ExtractionResult.model_validate_json(
            json.dumps(json_data)
        )  # type: ignore[assignment]

        # Run FontSizeHints.from_pages
        hints = FontSizeHints.from_pages(extraction.pages)

        # Serialize the hints to dict using built-in model_dump()
        golden_data = hints.model_dump()

        # Write golden file
        golden_path.write_text(json.dumps(golden_data, indent=2, sort_keys=True) + "\n")
        log.info(f"  Wrote {golden_path.name}")

    log.info(f"✓ Generated {len(raw_fixtures)} golden files")


if __name__ == "__main__":
    main()
