"""Tests for generate_golden_files script.

These tests ensure that the golden file generation process remains functional.
"""

import json
from pathlib import Path

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.classifier.lego_page_builder import build_page
from build_a_long.pdf_extract.classifier.tools.generate_golden_files import main
from build_a_long.pdf_extract.extractor import PageData


def test_page_serialization() -> None:
    """Test that Page can be serialized to JSON.

    This is a regression test for the generate_golden_files script.
    """
    # Load a simple fixture
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    raw_fixture = fixtures_dir / "6509377_page_010_raw.json"

    if not raw_fixture.exists():
        # Skip test if fixture doesn't exist
        return

    # Load the page data
    page: PageData = PageData.model_validate_json(raw_fixture.read_text())

    # Run classification
    result = classify_elements(page)

    # Build the Page
    page_element = build_page(result)

    # This should not raise an error
    golden_data = page_element.model_dump(by_alias=True)

    # Verify it's valid JSON
    json_str = json.dumps(golden_data)
    assert json_str is not None

    # Verify we can parse it back
    parsed = json.loads(json_str)
    assert isinstance(parsed, dict)
    assert "__tag__" in parsed
    assert parsed["__tag__"] == "Page"


def test_generate_golden_files_main() -> None:
    """Test that the generate_golden_files.main() function works end-to-end.

    This is a regression test to ensure the script doesn't break due to
    serialization issues (e.g., TYPE_CHECKING imports making types unavailable
    at runtime).
    """
    fixtures_dir = Path(__file__).parent.parent / "fixtures"
    expected_files = list(fixtures_dir.glob("*_raw.json"))

    if not expected_files:
        # Skip test if no fixtures are available
        return

    # Run the main function - this should not raise any errors
    main()

    # Verify that golden files were generated for each raw fixture
    for raw_fixture in expected_files:
        expected_path = raw_fixture.with_name(
            raw_fixture.name.replace("_raw.json", "_expected.json")
        )
        assert expected_path.exists(), (
            f"Expected golden file not found: {expected_path}"
        )

        # Verify the golden file is valid JSON
        golden_data = json.loads(expected_path.read_text())
        assert isinstance(golden_data, dict)
        assert "__tag__" in golden_data
        assert golden_data["__tag__"] == "Page"
