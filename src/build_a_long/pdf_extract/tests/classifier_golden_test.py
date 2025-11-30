"""Golden file tests for the classifier.

We use "golden files" to validate that the final Page output matches expected results.
Golden files contain the expected serialized Page output for known inputs.

How it works:
1. Test loads a raw fixture (PageData from real PDF extraction)
2. Loads classifier config with hints from the PDF's hint fixtures
3. Runs the classifier to produce a ClassificationResult
4. Builds a Page from the classification result
5. Serializes the Page using model_dump()
6. Compares against the expected golden file

Note: We compare the serialized Page.model_dump() output,
not the object directly, to ensure JSON round-tripping works correctly.

To update golden files:
    pants run src/build_a_long/pdf_extract/classifier/tools/generate_golden_files.py
"""

import difflib
import logging

import pytest

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.fixtures import (
    FIXTURES_DIR,
    RAW_FIXTURE_FILES,
    extract_element_id,
    load_classifier_config,
)

log = logging.getLogger(__name__)


def _compare_json(
    actual_json: str,
    expected_json: str,
    fixture_name: str,
) -> list[str]:
    """Compare actual and expected JSON outputs.

    Returns a list of error messages (empty list if they match).

    This performs a comparison of the JSON strings and shows
    a unified diff when they don't match.
    """
    errors: list[str] = []

    # Compare the JSON strings
    if actual_json != expected_json:
        # Generate a unified diff showing the differences
        diff_lines = list(
            difflib.unified_diff(
                expected_json.splitlines(keepends=True),
                actual_json.splitlines(keepends=True),
                fromfile=f"{fixture_name} (expected)",
                tofile=f"{fixture_name} (actual)",
                lineterm="",
            )
        )

        # Limit diff to first 100 lines to avoid overwhelming output
        if len(diff_lines) > 100:
            diff_lines = diff_lines[:100] + ["\n... (diff truncated) ...\n"]

        diff_output = "".join(diff_lines)
        errors.append(f"JSON outputs don't match:\n{diff_output}")

    return errors


class TestClassifierGolden:
    """Golden file tests validating Page output."""

    @pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
    def test_page_output_matches_golden(self, fixture_file: str) -> None:
        """Test that the Page output matches the golden file.

        This test:
        1. Loads a raw page fixture (PageData)
        2. Runs classification
        3. Builds a Page from the classification result
        4. Compares the Page against the golden file

        To update golden files, use:
            pants run src/build_a_long/pdf_extract/classifier/
            tools:generate-golden-files
        """
        fixture_path = FIXTURES_DIR / fixture_file

        # Determine golden file path
        golden_file = fixture_file.replace("_raw.json", "_expected.json")
        golden_path = FIXTURES_DIR / golden_file

        # Load the input fixture
        extraction: ExtractionResult = ExtractionResult.model_validate_json(
            fixture_path.read_text()
        )  # type: ignore[assignment]

        # Check that golden file exists
        if not golden_path.exists():
            pytest.skip(
                f"Golden file not found: {golden_file}\n"
                "Run: pants run src/build_a_long/pdf_extract/classifier/"
                "tools:generate-golden-files"
            )

        # Run classification on first page (same as generate_golden_files.py)
        if not extraction.pages:
            pytest.skip(f"No pages found in {fixture_file}")

        # Load classifier config with hints
        element_id = extract_element_id(fixture_file)
        config = load_classifier_config(element_id)

        page = extraction.pages[0]
        result = classify_elements(page, config)
        page_element = result.page

        # Serialize the Page using Pydantic's JSON encoder
        assert page_element is not None, "Page element should not be None"
        actual_json = page_element.to_json(indent=2) + "\n"

        # Load expected golden data
        expected_json = golden_path.read_text()

        # Compare the JSON strings
        comparison_errors = _compare_json(actual_json, expected_json, fixture_file)

        # Report errors if any
        if comparison_errors:
            pytest.fail(
                f"Page test failed for {fixture_file}:\n"
                + "\n".join(f"  - {e}" for e in comparison_errors)
                + "\n\nTo update golden files, run: "
                "pants run src/build_a_long/pdf_extract/classifier/"
                "tools:generate-golden-files",
                pytrace=False,
            )

        # Log success
        log.info(f"✓ {fixture_file}: {page_element}")
