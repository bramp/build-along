"""Centralized fixture file discovery for tests.

This module provides constants for different types of fixture files used across
the test suite, eliminating duplication of glob patterns.
"""

from pathlib import Path

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

# All fixture files
ALL_FIXTURE_FILES = sorted(f.name for f in FIXTURES_DIR.glob("*.json"))
