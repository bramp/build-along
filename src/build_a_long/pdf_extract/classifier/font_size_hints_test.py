"""Tests for font_size_hints module."""

import json
from pathlib import Path

import pytest

from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.cli.io import load_json_auto
from build_a_long.pdf_extract.extractor import ExtractionResult, PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Text


def test_from_pages_with_all_sizes() -> None:
    """Test hint extraction when we have all 3 top sizes."""
    # Create mock pages with text blocks
    # Use multiple pages to simulate realistic instruction booklet
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="2x", font_size=10.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="3x", font_size=10.0, id=2),
            Text(bbox=BBox(10, 50, 20, 60), text="4x", font_size=10.0, id=3),
            Text(bbox=BBox(10, 70, 20, 80), text="5x", font_size=12.0, id=4),
            Text(
                bbox=BBox(30, 10, 40, 20), text="99", font_size=16.0, id=5
            ),  # Integer not matching page ±1
        ],
    )
    page2 = PageData(
        page_number=2,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="1x", font_size=12.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="2x", font_size=12.0, id=2),
            Text(bbox=BBox(10, 50, 20, 60), text="3x", font_size=14.0, id=3),
            Text(bbox=BBox(10, 60, 20, 70), text="4x", font_size=14.0, id=4),
            Text(bbox=BBox(10, 70, 20, 80), text="5x", font_size=14.0, id=5),
        ],
    )
    page3 = PageData(
        page_number=3,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="1x", font_size=14.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="2x", font_size=14.0, id=2),
        ],
    )

    hints = FontSizeHints.from_pages([page1, page2, page3])

    assert (
        hints.part_count_size == 10.0
    )  # Most common in instruction pages (3 occurrences)
    assert hints.step_number_size == 12.0  # 2nd most common (3 occurrences)
    assert hints.step_repeat_size == 14.0  # 3rd most common (3 occurrences)

    # Remaining should only have "other integers" (not part counts or page numbers)
    assert hints.remaining_font_sizes == {"16.0": 1}


def test_from_pages_with_two_sizes() -> None:
    """Test hint extraction when we only have 2 part count sizes."""
    # Create multiple pages to meet MIN_SAMPLES requirement
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="2x", font_size=10.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="3x", font_size=10.0, id=2),
            Text(bbox=BBox(10, 50, 20, 60), text="4x", font_size=10.0, id=3),
        ],
    )
    page2 = PageData(
        page_number=2,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="1x", font_size=12.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="2x", font_size=12.0, id=2),
            Text(bbox=BBox(10, 50, 20, 60), text="3x", font_size=12.0, id=3),
        ],
    )
    page3 = PageData(
        page_number=3,
        bbox=BBox(0, 0, 100, 100),
        blocks=[],
    )

    hints = FontSizeHints.from_pages([page1, page2, page3])

    assert hints.part_count_size == 10.0
    assert hints.step_number_size == 12.0
    assert hints.step_repeat_size is None  # Not enough data

    # Remaining should be empty since both sizes are known
    assert hints.remaining_font_sizes == {}


def test_from_pages_with_one_size() -> None:
    """Test hint extraction when we only have 1 part count size."""
    # Create multiple pages to meet MIN_SAMPLES requirement
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="2x", font_size=10.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="3x", font_size=10.0, id=2),
            Text(bbox=BBox(10, 50, 20, 60), text="4x", font_size=10.0, id=3),
        ],
    )
    page2 = PageData(
        page_number=2,
        bbox=BBox(0, 0, 100, 100),
        blocks=[],
    )
    page3 = PageData(
        page_number=3,
        bbox=BBox(0, 0, 100, 100),
        blocks=[],
    )

    hints = FontSizeHints.from_pages([page1, page2, page3])

    assert hints.part_count_size == 10.0
    assert hints.catalog_part_count_size is None
    assert hints.step_number_size is None

    # Remaining should be empty since the only size is known
    assert hints.remaining_font_sizes == {}


def test_from_pages_with_no_part_counts() -> None:
    """Test hint extraction when there are no part count patterns."""
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(
                bbox=BBox(10, 10, 20, 20), text="10", font_size=10.0, id=1
            ),  # Integer not matching page ±1
            Text(
                bbox=BBox(10, 30, 20, 40), text="20", font_size=12.0, id=2
            ),  # Integer not matching page ±1
        ],
    )

    hints = FontSizeHints.from_pages([page1])

    assert hints.part_count_size is None
    assert hints.catalog_part_count_size is None
    assert hints.step_number_size is None

    # All "other integer" sizes should remain
    assert hints.remaining_font_sizes == {"10.0": 1, "12.0": 1}


def test_remaining_font_sizes_preserves_counts() -> None:
    """Test that remaining font sizes preserve their original counts."""
    # Need multiple pages to ensure data goes to instruction section
    page1 = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[
            Text(bbox=BBox(10, 10, 20, 20), text="2x", font_size=10.0, id=1),
            Text(bbox=BBox(10, 30, 20, 40), text="3x", font_size=12.0, id=2),
            Text(bbox=BBox(10, 50, 20, 60), text="50", font_size=14.0, id=3),  # Integer
            Text(bbox=BBox(10, 70, 20, 80), text="60", font_size=14.0, id=4),  # Integer
            Text(bbox=BBox(30, 10, 40, 20), text="70", font_size=16.0, id=5),  # Integer
        ],
    )
    page2 = PageData(
        page_number=2,
        bbox=BBox(0, 0, 100, 100),
        blocks=[],
    )
    page3 = PageData(
        page_number=3,
        bbox=BBox(0, 0, 100, 100),
        blocks=[],
    )

    hints = FontSizeHints.from_pages([page1, page2, page3])

    # Only "other integer" sizes should remain (not part counts)
    assert hints.remaining_font_sizes == {"14.0": 2, "16.0": 1}


def test_catalog_section_separation() -> None:
    """Test that catalog pages are analyzed separately from instruction pages."""
    # Create 9 pages: first 6 are instructions (2/3), last 3 are catalog (1/3)
    instruction_pages = [
        PageData(
            page_number=i,
            bbox=BBox(0, 0, 100, 100),
            blocks=[
                Text(bbox=BBox(10, 10, 20, 20), text="1x", font_size=12.0, id=1),
                Text(bbox=BBox(10, 30, 20, 40), text="2x", font_size=12.0, id=2),
                Text(bbox=BBox(10, 50, 20, 60), text="3x", font_size=12.0, id=3),
            ],
        )
        for i in range(1, 7)
    ]

    # Catalog pages with smaller font and higher frequency
    catalog_pages = [
        PageData(
            page_number=i,
            bbox=BBox(0, 0, 100, 100),
            blocks=[
                Text(
                    bbox=BBox(10, j * 10, 20, j * 10 + 10),
                    text=f"{j}x",
                    font_size=8.0,
                    id=j,
                )
                for j in range(1, 21)  # 20 part counts per catalog page
            ],
        )
        for i in range(7, 10)
    ]

    all_pages = instruction_pages + catalog_pages
    hints = FontSizeHints.from_pages(all_pages)

    # Despite catalog having more total occurrences (60 vs 18),
    # part_count_size should come from instruction pages
    assert hints.part_count_size == 12.0
    # Catalog part count should be the smaller size from catalog section
    assert hints.catalog_part_count_size == 8.0


def test_catalog_size_validation() -> None:
    """Test that catalog sizes larger than instruction sizes are rejected."""
    # Instruction pages with smaller font
    instruction_pages = [
        PageData(
            page_number=i,
            bbox=BBox(0, 0, 100, 100),
            blocks=[
                Text(bbox=BBox(10, 10, 20, 20), text="1x", font_size=8.0, id=1),
                Text(bbox=BBox(10, 30, 20, 40), text="2x", font_size=8.0, id=2),
                Text(bbox=BBox(10, 50, 20, 60), text="3x", font_size=8.0, id=3),
            ],
        )
        for i in range(1, 7)
    ]

    # Catalog pages with LARGER font (unusual, should be rejected)
    catalog_pages = [
        PageData(
            page_number=i,
            bbox=BBox(0, 0, 100, 100),
            blocks=[
                Text(
                    bbox=BBox(10, j * 10, 20, j * 10 + 10),
                    text=f"{j}x",
                    font_size=12.0,
                    id=j,
                )
                for j in range(1, 11)
            ],
        )
        for i in range(7, 10)
    ]

    all_pages = instruction_pages + catalog_pages
    hints = FontSizeHints.from_pages(all_pages)

    assert hints.part_count_size == 8.0
    # Catalog size should be None because 12.0 > 8.0 (invalid)
    assert hints.catalog_part_count_size is None


class TestFontSizeHintsGolden:
    """Golden file tests for FontSizeHints.from_pages."""

    @pytest.mark.parametrize(
        "fixture_file",
        [
            "6055741_raw.json.bz2",
            "6509377_raw.json.bz2",
            "6580053_raw.json.bz2",
        ],
    )
    def test_from_pages_matches_golden(self, fixture_file: str) -> None:
        """Test that FontSizeHints.from_pages matches the golden file.

        This test:
        1. Loads a raw page fixture (compressed)
        2. Runs FontSizeHints.from_pages
        3. Compares against golden file

        To generate golden files, run:
            pants run \\
                src/build_a_long/pdf_extract/classifier/ \\
                tools:generate-font-hints-golden
        """
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        fixture_path = fixtures_dir / fixture_file

        # Determine golden file path
        golden_file = fixture_file.replace("_raw.json.bz2", "_font_hints_expected.json")
        golden_path = fixtures_dir / golden_file

        # Load the input fixture
        json_data = load_json_auto(fixture_path)
        extraction: ExtractionResult = ExtractionResult.from_json(json.dumps(json_data))  # type: ignore[assignment]

        # Run FontSizeHints.from_pages
        hints = FontSizeHints.from_pages(extraction.pages)

        # Serialize the hints to dict using built-in to_dict()
        actual = hints.to_dict()

        # Check if golden file exists
        if not golden_path.exists():
            pytest.skip(
                f"Golden file not found: {golden_file}\n"
                "Run: pants run "
                "src/build_a_long/pdf_extract/classifier/"
                "tools:generate-font-hints-golden"
            )

        # Load expected results
        with open(golden_path) as f:
            expected = json.load(f)

        # Compare results
        assert actual == expected, (
            f"FontSizeHints mismatch for {fixture_file}.\n"
            f"Expected: {json.dumps(expected, indent=2, sort_keys=True)}\n"
            f"Actual: {json.dumps(actual, indent=2, sort_keys=True)}"
        )
