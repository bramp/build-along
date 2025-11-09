#!/usr/bin/env python3
"""Generate golden files for classifier tests.

This script runs the classifier on all fixtures and generates golden output files.

Usage:
    pants run src/build_a_long/pdf_extract/classifier/tools:generate-golden-files
"""

import json
import logging
import sys
from pathlib import Path

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import ExtractionResult, PageData

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    """Generate golden files for all fixtures."""

    # TODO maybe be explict with the full path of the directory
    fixtures_dir = Path("src/build_a_long/pdf_extract/fixtures")

    if not fixtures_dir.exists():
        log.error(f"Fixtures directory not found: {fixtures_dir}")
        sys.exit(1)

    raw_fixtures = list(fixtures_dir.glob("*_raw.json"))

    if not raw_fixtures:
        log.error(f"No *_raw.json fixtures found in {fixtures_dir}")
        sys.exit(1)

    log.info(f"Found {len(raw_fixtures)} raw fixtures")
    log.info(f"Writing golden files to: {fixtures_dir.absolute()}")

    for fixture_path in sorted(raw_fixtures):
        golden_path = fixture_path.with_name(
            fixture_path.name.replace("_raw.json", "_expected.json")
        )

        log.info(f"Processing {fixture_path.name}...")

        # Load the input fixture (which is an ExtractionResult with pages)
        extraction_result: ExtractionResult = ExtractionResult.from_json(
            fixture_path.read_text()
        )  # type: ignore[assignment]

        # Get the first (and usually only) page
        if not extraction_result.pages:
            log.warning(f"  Skipping {fixture_path.name} - no pages found")
            continue

        page: PageData = extraction_result.pages[0]

        # Run classification
        result = classify_elements(page)

        # Serialize the results using to_dict()
        golden_data = result.to_dict()

        # Write golden file
        golden_path.write_text(json.dumps(golden_data, indent=2) + "\n")
        log.info(f"  Wrote {golden_path.absolute()}")

    log.info(f"✓ Generated {len(raw_fixtures)} golden files")


if __name__ == "__main__":
    main()
