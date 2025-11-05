"""Golden file tests for the PDF element classifier.

This test suite validates that classification output matches expected golden files.
Golden files are JSON snapshots of classification results that represent known-good
output. When classification logic changes intentionally, golden files can be updated
using the generate-golden-files script.

Golden files are stored in fixtures/ with suffix '_expected.json'.
Input fixtures are in fixtures/ with suffix '_raw.json'.

These tests also run all invariant checks from classifier_rules_test to ensure
both correctness of output and adherence to fundamental rules.
"""

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from build_a_long.pdf_extract.classifier import classify_elements
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.classifier_rules_test import ClassifiedPage
from build_a_long.pdf_extract.extractor import PageData

log = logging.getLogger(__name__)


def _compare_classification_results(
    actual: dict[str, Any],
    expected: dict[str, Any],
    fixture_name: str,
) -> list[str]:
    """Compare actual and expected classification results.

    Returns a list of error messages (empty list if they match).

    Note: We compare the serialized ClassificationResult.to_dict() output,
    which includes fields like:
    - _warnings: List of warning messages
    - _removal_reasons: Dict mapping element IDs to removal reasons
    - _constructed_elements: Dict mapping element IDs to LegoPageElements
    - _candidates: Dict mapping labels to lists of candidates
    """
    errors: list[str] = []

    # Compare warnings
    actual_warnings = actual.get("_warnings", [])
    expected_warnings = expected.get("_warnings", [])
    if actual_warnings != expected_warnings:
        errors.append(
            f"Warnings mismatch: expected {expected_warnings}, got {actual_warnings}"
        )

    # Compare removal reasons
    actual_removed = actual.get("_removal_reasons", {})
    expected_removed = expected.get("_removal_reasons", {})
    if actual_removed != expected_removed:
        all_removed_ids = set(str(k) for k in actual_removed) | set(
            str(k) for k in expected_removed
        )
        for elem_id in sorted(all_removed_ids):
            str_id = str(elem_id)
            if str_id not in expected_removed:
                errors.append(f"Element {elem_id}: unexpectedly removed")
            elif str_id not in actual_removed:
                errors.append(f"Element {elem_id}: expected to be removed but wasn't")

    # Compare constructed elements (check that types match)
    actual_constructed = actual.get("_constructed_elements", {})
    expected_constructed = expected.get("_constructed_elements", {})

    all_constructed_ids = set(str(k) for k in actual_constructed) | set(
        str(k) for k in expected_constructed
    )
    for elem_id in sorted(all_constructed_ids):
        str_id = str(elem_id)
        actual_elem = actual_constructed.get(str_id)
        expected_elem = expected_constructed.get(str_id)

        if actual_elem is None and expected_elem is not None:
            errors.append(
                f"Element {elem_id}: expected constructed type "
                f"'{expected_elem.get('__tag__')}', got None"
            )
        elif actual_elem is not None and expected_elem is None:
            errors.append(
                f"Element {elem_id}: expected None, got constructed type "
                f"'{actual_elem.get('__tag__')}'"
            )
        elif actual_elem is not None and expected_elem is not None:
            actual_type = actual_elem.get("__tag__")
            expected_type = expected_elem.get("__tag__")
            if actual_type != expected_type:
                errors.append(
                    f"Element {elem_id}: expected type '{expected_type}', "
                    f"got '{actual_type}'"
                )

    # Compare candidates (check winners for each label)
    actual_candidates = actual.get("_candidates", {})
    expected_candidates = expected.get("_candidates", {})

    all_labels = set(actual_candidates.keys()) | set(expected_candidates.keys())
    for label in sorted(all_labels):
        actual_label_candidates = actual_candidates.get(label, [])
        expected_label_candidates = expected_candidates.get(label, [])

        # Extract winners
        actual_winners = [c for c in actual_label_candidates if c.get("is_winner")]
        expected_winners = [c for c in expected_label_candidates if c.get("is_winner")]

        # Compare winner element IDs
        actual_winner_ids = {
            c.get("source_element", {}).get("id")
            for c in actual_winners
            if c.get("source_element")
        }
        expected_winner_ids = {
            c.get("source_element", {}).get("id")
            for c in expected_winners
            if c.get("source_element")
        }

        if actual_winner_ids != expected_winner_ids:
            errors.append(
                f"Label '{label}': expected winner IDs {expected_winner_ids}, "
                f"got {actual_winner_ids}"
            )

    return errors


def _run_invariant_checks(
    page: PageData, result: ClassificationResult, fixture_name: str
) -> list[str]:
    """Run all invariant checks from classifier_rules_test.

    Returns a list of error messages (empty list if all checks pass).
    """
    errors: list[str] = []
    classified = ClassifiedPage(page, result)

    parts_lists = classified.parts_lists()
    part_images = classified.part_images()

    # Check 1: Every parts list contains at least one part image
    for parts_list in parts_lists:
        part_images_inside = classified.children_of(parts_list, label="part_image")
        if len(part_images_inside) < 1:
            errors.append(
                f"PartsList {parts_list.id} contains no part_images (expected >= 1)"
            )

        # Check that no part_images inside are deleted
        all_part_images_inside = [
            elem
            for elem in page.elements
            if result.get_label(elem) == "part_image"
            and elem.bbox.fully_inside(parts_list.bbox)
        ]
        deleted_images = [
            img for img in all_part_images_inside if result.is_removed(img)
        ]
        if deleted_images:
            errors.append(
                f"PartsList {parts_list.id} has {len(deleted_images)} deleted part_images inside"
            )

        # Check that part_counts match part_images
        part_counts_inside = classified.children_of(parts_list, label="part_count")
        if len(part_counts_inside) != len(part_images_inside):
            errors.append(
                f"PartsList {parts_list.id} has {len(part_images_inside)} part_images "
                f"but {len(part_counts_inside)} part_counts (should match)"
            )

    # Check 2: Parts lists don't overlap
    for i, parts_list_a in enumerate(parts_lists):
        for parts_list_b in parts_lists[i + 1 :]:
            if parts_list_a.bbox.overlaps(parts_list_b.bbox):
                errors.append(
                    f"PartsLists {parts_list_a.id} and {parts_list_b.id} overlap"
                )

    # Check 3: Each part image is inside a parts list
    for part_image in part_images:
        inside_any = any(part_image.bbox.fully_inside(pl.bbox) for pl in parts_lists)
        if not inside_any:
            errors.append(f"PartImage {part_image.id} is not inside any parts_list")

    # Check 4: No labeled element is deleted
    for elem in page.elements:
        if result.get_label(elem) is not None and result.is_removed(elem):
            errors.append(
                f"Element {elem.id} has label '{result.get_label(elem)}' but is marked deleted"
            )

    return errors


@pytest.mark.skip(
    reason=(
        "Classifier has known issues with deleting labeled elements. "
        "Enable this test once classifier_rules_test passes."
    )
)
class TestClassifierGolden:
    """Golden file tests validating exact classification output."""

    @pytest.mark.parametrize(
        "fixture_file",
        [f.name for f in (Path(__file__).with_name("fixtures")).glob("*_raw.json")],
    )
    def test_classification_matches_golden(self, fixture_file: str) -> None:
        """Test that classification output matches the golden file.

        This test:
        1. Loads a raw page fixture
        2. Runs classification
        3. Compares against golden file
        4. Runs all invariant checks

        To update golden files, use:
            pants run src/build_a_long/pdf_extract/classifier:generate-golden-files
        """
        fixtures_dir = Path(__file__).with_name("fixtures")
        fixture_path = fixtures_dir / fixture_file

        # Determine golden file path
        golden_file = fixture_file.replace("_raw.json", "_expected.json")
        golden_path = fixtures_dir / golden_file

        # Load the input fixture
        page: PageData = PageData.from_json(fixture_path.read_text())  # type: ignore[assignment]

        # Run classification
        result = classify_elements(page)

        # Serialize the results using to_dict()
        actual = result.to_dict()

        # Check that golden file exists
        if not golden_path.exists():
            pytest.fail(
                f"Golden file not found: {golden_file}\n"
                "Run: pants run "
                "src/build_a_long/pdf_extract/classifier:generate-golden-files"
            )

        expected = json.loads(golden_path.read_text())

        # Compare results
        comparison_errors = _compare_classification_results(
            actual, expected, fixture_file
        )

        # Run invariant checks
        invariant_errors = _run_invariant_checks(
            page, result, fixture_name=fixture_file
        )

        # Combine all errors
        all_errors = []
        if comparison_errors:
            all_errors.append("Golden file comparison failures:")
            all_errors.extend(f"  - {e}" for e in comparison_errors)
        if invariant_errors:
            all_errors.append("Invariant check failures:")
            all_errors.extend(f"  - {e}" for e in invariant_errors)

        if all_errors:
            pytest.fail(
                f"Classification test failed for {fixture_file}:\n"
                + "\n".join(all_errors)
                + "\n\nTo update golden files, run: "
                "pants run src/build_a_long/pdf_extract/classifier:generate-golden-files"
            )

        # Log success
        classified = ClassifiedPage(page, result)
        classified.print_summary(log)
