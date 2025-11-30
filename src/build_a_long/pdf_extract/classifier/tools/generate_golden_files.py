#!/usr/bin/env python3
"""Generate golden files for classifier tests.

This script runs the classifier on all fixtures and generates golden output files.
It uses hint fixtures (font_hints and page_hints) when available for consistent
classification.

Usage:
    pants run src/build_a_long/pdf_extract/classifier/tools:generate-golden-files
"""

import logging
import re
import sys
from pathlib import Path

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.extractor import ExtractionResult, PageData
from build_a_long.pdf_extract.fixtures import load_classifier_config

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

    # Track which instruction IDs are missing hints
    missing_hints: list[str] = []

    for fixture_path in sorted(raw_fixtures):
        golden_path = fixture_path.with_name(
            fixture_path.name.replace("_raw.json", "_expected.json")
        )

        log.info(f"Processing {fixture_path.name}...")

        # Load the input fixture (which is an ExtractionResult with pages)
        extraction_result: ExtractionResult = ExtractionResult.model_validate_json(
            fixture_path.read_text()
        )  # type: ignore[assignment]

        # Get the first (and usually only) page
        if not extraction_result.pages:
            log.warning(f"  Skipping {fixture_path.name} - no pages found")
            continue

        page: PageData = extraction_result.pages[0]

        # Try to load hints for this element ID
        element_id = extract_element_id(fixture_path.name)
        config: ClassifierConfig | None = None

        if element_id:
            try:
                config = load_classifier_config(element_id)
                log.info(f"  Using hints for {element_id}")
            except FileNotFoundError:
                log.warning(f"  No hints found for {element_id}")
                missing_hints.append(element_id)

        # Run classification
        result = classify_elements(page, config)

        # Build the Page from classification results
        page_element = result.page

        # Serialize with by_alias=True to use __tag__ instead of tag
        # Use Pydantic's JSON encoder for consistent serialization
        assert page_element is not None, "Page element should not be None"
        golden_json = page_element.to_json(indent=2) + "\n"

        # Write golden file
        golden_path.write_text(golden_json)
        log.info(f"  Wrote {golden_path.absolute()}")

    log.info(f"✓ Generated {len(raw_fixtures)} golden files")

    if missing_hints:
        unique_missing = sorted(set(missing_hints))
        log.warning("")
        log.warning("=" * 70)
        log.warning("WARNING: The following element IDs had no hints:")
        for elem_id in unique_missing:
            log.warning(f"  - {elem_id}")
        log.warning("")
        log.warning("Run the following to generate hints:")
        log.warning(
            "  pants run src/build_a_long/pdf_extract/classifier/"
            "tools:generate-golden-hints"
        )
        log.warning("=" * 70)


if __name__ == "__main__":
    main()
