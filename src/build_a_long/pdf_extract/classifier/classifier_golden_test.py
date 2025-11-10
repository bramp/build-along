"""Golden file tests for the classifier.

We use "golden files" to validate that the final Page output matches expected results.
Golden files contain the expected serialized Page output for known inputs.

How it works:
1. Test loads a raw fixture (PageData from real PDF extraction)
2. Runs the classifier to produce a ClassificationResult
3. Builds a Page from the classification result
4. Serializes the Page using model_dump()
5. Compares against the expected golden file

Note: We compare the serialized Page.model_dump() output,
not the object directly, to ensure JSON round-tripping works correctly.

To update golden files:
    pants run src/build_a_long/pdf_extract/classifier/tools:generate-golden-files
"""

import json
import logging
from pathlib import Path

import pytest

from build_a_long.pdf_extract.classifier.classifier import classify_pages
from build_a_long.pdf_extract.classifier.lego_page_builder import build_page
from build_a_long.pdf_extract.extractor import ExtractionResult

log = logging.getLogger(__name__)


def _compare_pages(
    actual: dict,
    expected: dict,
    fixture_name: str,
) -> list[str]:
    """Compare actual and expected Page outputs.

    Returns a list of error messages (empty list if they match).

    This performs a deep comparison of the Page structure.
    """
    errors: list[str] = []

    # Simple deep comparison - if they don't match, report the differences
    if actual != expected:
        # Try to give more specific error messages for common differences
        if actual.get("__tag__") != expected.get("__tag__"):
            errors.append(
                f"Tag mismatch: expected {expected.get('__tag__')}, "
                f"got {actual.get('__tag__')}"
            )

        # Compare page number
        actual_page_num = actual.get("page_number")
        expected_page_num = expected.get("page_number")
        if actual_page_num != expected_page_num:
            errors.append(
                f"Page number mismatch: expected {expected_page_num}, "
                f"got {actual_page_num}"
            )

        # Compare steps count
        actual_steps = actual.get("steps", [])
        expected_steps = expected.get("steps", [])
        if len(actual_steps) != len(expected_steps):
            errors.append(
                f"Steps count mismatch: expected {len(expected_steps)} steps, "
                f"got {len(actual_steps)} steps"
            )

        # Compare parts_lists count
        actual_parts_lists = actual.get("parts_lists", [])
        expected_parts_lists = expected.get("parts_lists", [])
        if len(actual_parts_lists) != len(expected_parts_lists):
            errors.append(
                f"Parts lists count mismatch: expected {len(expected_parts_lists)}, "
                f"got {len(actual_parts_lists)}"
            )

        # If we didn't find specific differences, just report general mismatch
        if not errors:
            errors.append(
                "Page structures don't match "
                "(run generate-golden-files to see full diff)"
            )

    return errors


class TestClassifierGolden:
    """Golden file tests validating Page output."""

    @pytest.mark.parametrize(
        "fixture_file",
        [
            f.name
            for f in (Path(__file__).parent.parent / "fixtures").glob("*_raw.json")
        ],
    )
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
        fixtures_dir = Path(__file__).parent.parent / "fixtures"
        fixture_path = fixtures_dir / fixture_file

        # Determine golden file path
        golden_file = fixture_file.replace("_raw.json", "_expected.json")
        golden_path = fixtures_dir / golden_file

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

        # Run classification and build pages
        batch_result = classify_pages(extraction.pages)
        pages = [build_page(result) for result in batch_result.results]

        # Serialize the Page(s) using model_dump with by_alias=True
        actual = [page.model_dump(by_alias=True) for page in pages]

        # Load expected golden data
        expected = json.loads(golden_path.read_text())

        # Compare results (handle both single page and multi-page)
        actual_to_compare = actual[0] if len(actual) == 1 else actual

        # Normalize expected to same format (single dict or list)
        if isinstance(actual_to_compare, dict) and isinstance(expected, dict):
            comparison_errors = _compare_pages(
                actual_to_compare, expected, fixture_file
            )
        elif isinstance(actual_to_compare, list) and isinstance(expected, list):
            # Multi-page comparison
            comparison_errors = []
            if len(actual_to_compare) != len(expected):
                comparison_errors.append(
                    f"Page count mismatch: expected {len(expected)} pages, "
                    f"got {len(actual_to_compare)} pages"
                )
            else:
                for i, (actual_page, expected_page) in enumerate(
                    zip(actual_to_compare, expected, strict=True), start=1
                ):
                    errors = _compare_pages(actual_page, expected_page, fixture_file)
                    comparison_errors.extend(f"Page {i}: {e}" for e in errors)
        else:
            comparison_errors = [
                f"Format mismatch: actual is {type(actual_to_compare).__name__}, "
                f"expected is {type(expected).__name__}"
            ]

        # Report errors if any
        if comparison_errors:
            pytest.fail(
                f"Page test failed for {fixture_file}:\n"
                + "\n".join(f"  - {e}" for e in comparison_errors)
                + "\n\nTo update golden files, run: "
                "pants run src/build_a_long/pdf_extract/classifier/"
                "tools:generate-golden-files"
            )

        # Log success
        if pages:
            log.info(f"✓ {fixture_file}: {pages[0]}")
