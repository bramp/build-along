"""Centralized fixture file discovery for tests.

This module provides constants for different types of fixture files used across
the test suite, eliminating duplication of glob patterns.
"""

import json
import re
from pathlib import Path

from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.pages import PageHintCollection
from build_a_long.pdf_extract.classifier.text import FontSizeHints

# Base directory containing all fixture files
FIXTURES_DIR = Path(__file__).parent / "fixtures"

# Raw input fixture files (input to extractor/classifier)
RAW_FIXTURE_FILES = sorted(f.name for f in FIXTURES_DIR.glob("*_raw.json"))

# Expected output fixture files (golden files for classifier output)
EXPECTED_FIXTURE_FILES = sorted(f.name for f in FIXTURES_DIR.glob("*_expected.json"))

# Font hints fixture files (golden files for font size hints)
FONT_HINTS_FIXTURE_FILES = sorted(
    f.name for f in FIXTURES_DIR.glob("*_font_hints_expected.json")
)

# Page hints fixture files (golden files for page hints)
PAGE_HINTS_FIXTURE_FILES = sorted(
    f.name for f in FIXTURES_DIR.glob("*_page_hints_expected.json")
)

# All fixture files
ALL_FIXTURE_FILES = sorted(f.name for f in FIXTURES_DIR.glob("*.json"))


def extract_element_id(fixture_file: str) -> str:
    """Extract element ID from a fixture filename.

    An element ID identifies a LEGO element (manual, piece, sticker sheet, etc.).

    Args:
        fixture_file: Fixture filename like '6509377_page_013_raw.json'

    Returns:
        The element ID (e.g., '6509377')

    Raises:
        ValueError: If the element ID cannot be extracted
    """
    # Match the leading digits before the first underscore
    match = re.match(r"^(\d+)", fixture_file)
    if not match:
        raise ValueError(f"Cannot extract element ID from: {fixture_file}")
    return match.group(1)


def load_font_hints(element_id: str) -> FontSizeHints:
    """Load font size hints from a fixture file.

    Args:
        element_id: The element ID (e.g., "6509377"). An element ID identifies
            a LEGO element (manual, piece, sticker sheet, etc.). This should
            match the prefix of an existing *_font_hints_expected.json file.

    Returns:
        FontSizeHints loaded from the fixture file.

    Raises:
        FileNotFoundError: If the hint fixture file doesn't exist.
    """
    hint_file = FIXTURES_DIR / f"{element_id}_font_hints_expected.json"
    if not hint_file.exists():
        raise FileNotFoundError(
            f"Font hints fixture not found: {hint_file}\n"
            f"Run: pants run src/build_a_long/pdf_extract/classifier/"
            f"tools:generate-golden-hints"
        )

    with open(hint_file) as f:
        data = json.load(f)

    return FontSizeHints.model_validate(data)


def load_page_hints(element_id: str) -> PageHintCollection:
    """Load page hints from a fixture file.

    Args:
        element_id: The element ID (e.g., "6509377"). An element ID identifies
            a LEGO element (manual, piece, sticker sheet, etc.). This should
            match the prefix of an existing *_page_hints_expected.json file.

    Returns:
        PageHintCollection loaded from the fixture file.

    Raises:
        FileNotFoundError: If the hint fixture file doesn't exist.
    """
    hint_file = FIXTURES_DIR / f"{element_id}_page_hints_expected.json"
    if not hint_file.exists():
        raise FileNotFoundError(
            f"Page hints fixture not found: {hint_file}\n"
            f"Run: pants run src/build_a_long/pdf_extract/classifier/"
            f"tools:generate-golden-hints"
        )

    with open(hint_file) as f:
        data = json.load(f)

    return PageHintCollection.model_validate(data)


def load_classifier_config(element_id: str) -> ClassifierConfig:
    """Load classifier config with hints from fixture files.

    This is the recommended way to load hints for tests - it creates a
    ClassifierConfig with both font_size_hints and page_hints loaded
    from the golden fixture files.

    Args:
        element_id: The element ID (e.g., "6509377"). An element ID identifies
            a LEGO element (manual, piece, sticker sheet, etc.). This should
            match the prefix of existing hint fixture files.

    Returns:
        ClassifierConfig with font_size_hints and page_hints populated.

    Raises:
        FileNotFoundError: If any hint fixture file doesn't exist.
    """
    font_hints = load_font_hints(element_id)
    page_hints = load_page_hints(element_id)

    return ClassifierConfig(font_size_hints=font_hints, page_hints=page_hints)
