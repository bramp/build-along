"""Fixture loading utilities for tests.

This module provides shared code for loading fixture definitions from index.json5
and extracting pages from PDFs. It is used by both the test suite and the
regenerate_fixtures.py script.
"""

import difflib
from pathlib import Path

import json5
import pymupdf
from pydantic import BaseModel

from build_a_long.pdf_extract.extractor.extractor import Extractor, PageData
from build_a_long.pdf_extract.fixtures import FIXTURES_DIR
from build_a_long.pdf_extract.parser import parse_page_ranges

# Path to index.json5
INDEX_PATH = FIXTURES_DIR / "index.json5"


class FixtureDefinition(BaseModel):
    """Definition of a fixture from index.json5."""

    pdf: str
    """Path to the source PDF file (relative to repo root)."""

    description: str
    """Human-readable description of the fixture."""

    pages: str | None = None
    """Page range to extract (e.g., '10-17,180'). None means all pages."""

    compress: bool = False
    """Whether to compress the output with bz2."""

    @property
    def element_id(self) -> str:
        """Get the element ID from the PDF path."""
        return Path(self.pdf).stem

    @property
    def pdf_path(self) -> Path:
        """Get the PDF path as a Path object."""
        return Path(self.pdf)

    def get_page_numbers(self, total_pages: int) -> list[int]:
        """Get the list of 1-indexed page numbers to extract.

        Args:
            total_pages: Total number of pages in the PDF

        Returns:
            List of 1-indexed page numbers
        """
        if self.pages is None:
            return list(range(1, total_pages + 1))
        page_ranges = parse_page_ranges(self.pages)
        return list(page_ranges.page_numbers(total_pages))

    def get_fixture_filename(self, page_num: int | None = None) -> str:
        """Get the fixture filename for this definition.

        Args:
            page_num: Page number for per-page fixtures, or None for whole-doc

        Returns:
            Filename like '6509377_page_013_raw.json' or '6509377_raw.json.bz2'
        """
        if page_num is not None:
            return f"{self.element_id}_page_{page_num:03d}_raw.json"
        elif self.compress:
            return f"{self.element_id}_raw.json.bz2"
        else:
            return f"{self.element_id}_raw.json"

    @property
    def is_per_page(self) -> bool:
        """Whether this fixture generates per-page files."""
        return self.pages is not None


def compare_json(expected_json: str, actual_json: str, fixture_name: str) -> str | None:
    """Compare two JSON strings and return diff if different.

    Args:
        expected_json: Expected JSON content
        actual_json: Actual JSON content
        fixture_name: Name for diff output

    Returns:
        Diff string if different, None if identical
    """
    if expected_json == actual_json:
        return None

    # Only split into lines when we need to generate a diff
    diff_lines = list(
        difflib.unified_diff(
            expected_json.splitlines(keepends=True),
            actual_json.splitlines(keepends=True),
            fromfile=f"{fixture_name} (expected)",
            tofile=f"{fixture_name} (actual)",
            lineterm="",
        )
    )

    # Limit diff to first 100 lines
    if len(diff_lines) > 100:
        diff_lines = diff_lines[:100] + ["\n... (diff truncated) ...\n"]

    return "".join(diff_lines)


def load_fixture_definitions(index_path: Path | None = None) -> list[FixtureDefinition]:
    """Load fixture definitions from index.json5.

    Args:
        index_path: Path to index.json5 file. Defaults to INDEX_PATH.

    Returns:
        List of FixtureDefinition objects
    """
    if index_path is None:
        index_path = INDEX_PATH

    if not index_path.exists():
        return []

    data: dict = json5.loads(index_path.read_text())  # type: ignore[assignment]
    fixtures = data.get("fixtures", [])
    if not isinstance(fixtures, list):
        return []
    return [FixtureDefinition.model_validate(f) for f in fixtures]


class ExtractionResult:
    """Result of extracting pages from a PDF.

    Attributes:
        pages: Dict mapping page number to PageData
        total_pages: Total number of pages in the PDF
        page_numbers: List of page numbers that were extracted
    """

    def __init__(
        self,
        pages: dict[int, PageData],
        total_pages: int,
        page_numbers: list[int],
    ) -> None:
        self.pages = pages
        self.total_pages = total_pages
        self.page_numbers = page_numbers


def extract_pages_from_pdf(
    pdf_path: Path, page_range: str | None = None
) -> ExtractionResult:
    """Extract pages from a PDF.

    Opens the PDF once and extracts the specified pages.

    Args:
        pdf_path: Path to PDF file
        page_range: Page range string (e.g., '10-17,180'). None means all pages.

    Returns:
        ExtractionResult with pages dict, total page count, and extracted page numbers
    """
    result: dict[int, PageData] = {}

    with pymupdf.open(str(pdf_path)) as doc:
        total_pages = len(doc)

        # Resolve page range to actual page numbers
        if page_range is None:
            page_numbers = list(range(1, total_pages + 1))
        else:
            page_ranges = parse_page_ranges(page_range)
            page_numbers = list(page_ranges.page_numbers(total_pages))

        for page_num in page_numbers:
            page_index = page_num - 1
            if page_index < 0 or page_index >= total_pages:
                continue

            page = doc[page_index]
            extractor = Extractor(page=page, page_num=page_num, include_metadata=True)
            page_data = extractor.extract_page_data()
            result[page_num] = page_data

    return ExtractionResult(result, total_pages, page_numbers)
