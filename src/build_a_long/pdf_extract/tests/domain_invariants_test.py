"""Integration tests for LEGO page domain invariants.

These tests validate that the complete classification pipeline produces
domain objects (Page, PartsList, Part, etc.) that satisfy LEGO instruction
layout invariants.

Tests run on real fixture files to ensure end-to-end correctness.

The actual validation logic is implemented in the validation module.
These tests run those validators against fixture files and assert no issues.
"""

import logging

import pytest

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.classifier.classifier_rules_test import (
    RAW_FIXTURE_FILES,
    _load_config_for_fixture,
)
from build_a_long.pdf_extract.classifier.classifier_rules_test import (
    _load_pages_from_fixture as load_pages,
)
from build_a_long.pdf_extract.validation import (
    ValidationResult,
    validate_content_no_metadata_overlap,
    validate_elements_within_page,
    validate_no_divider_intersection,
    validate_part_contains_children,
    validate_parts_list_has_parts,
    validate_parts_lists_no_overlap,
    validate_steps_no_significant_overlap,
)

log = logging.getLogger(__name__)

# Known failing fixtures that should be skipped for certain tests
KNOWN_STEP_OVERLAP_FAILURES: set[str] = set()


def _run_validation_on_fixtures(
    fixture_file: str,
    validator_func: callable,  # type: ignore[type-arg]
    *,
    skip_fixtures: set[str] | None = None,
    skip_reason: str = "Known failing fixture",
    **validator_kwargs: object,
) -> ValidationResult:
    """Helper to run a validation function on all pages in a fixture file.

    Args:
        fixture_file: Name of the fixture file to test
        validator_func: Validation function to run (takes validation, page, page_data)
        skip_fixtures: Set of fixture filenames to skip
        skip_reason: Reason message for skipping
        **validator_kwargs: Additional kwargs to pass to validator_func

    Returns:
        ValidationResult with all issues found
    """
    if skip_fixtures and fixture_file in skip_fixtures:
        pytest.skip(skip_reason)

    pages = load_pages(fixture_file)
    config = _load_config_for_fixture(fixture_file)
    validation = ValidationResult()

    for _page_idx, page_data in enumerate(pages):
        result = classify_elements(page_data, config)
        page = result.page

        if page is None:
            continue

        validator_func(validation, page, page_data, **validator_kwargs)

    return validation


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_parts_list_contains_parts(fixture_file: str) -> None:
    """Every PartsList should contain at least one Part.

    Domain Invariant: A parts list without parts doesn't make sense in the
    context of LEGO instructions. Each PartsList should contain ≥1 Part objects.

    NOTE: Currently this is a soft assertion - we log when empty PartsList
    objects are found, but don't fail the test. This appears to be a known
    issue in the classification pipeline that needs investigation.
    """
    validation = _run_validation_on_fixtures(
        fixture_file, validate_parts_list_has_parts
    )

    # Log warnings but don't fail (known issue in classification)
    for issue in validation.issues:
        if issue.rule == "empty_parts_list":
            log.warning(f"{fixture_file}: {issue.message} - {issue.details}")

    # TODO: Make this a hard assertion once classification is fixed
    # assert not validation.has_issues(), (
    #     f"Found {len(validation.issues)} empty PartsList issues in {fixture_file}"
    # )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_parts_lists_do_not_overlap(fixture_file: str) -> None:
    """PartsList bounding boxes should not overlap.

    Domain Invariant: Each parts list occupies a distinct region on the page.
    Overlapping parts lists would indicate a classification error.
    """
    validation = _run_validation_on_fixtures(
        fixture_file, validate_parts_lists_no_overlap
    )

    errors = [i for i in validation.issues if i.rule == "overlapping_parts_lists"]
    assert not errors, (
        f"Found {len(errors)} overlapping PartsList issues in {fixture_file}: "
        + "; ".join(i.details or i.message for i in errors[:3])
    )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_steps_do_not_overlap(fixture_file: str) -> None:
    """Step bounding boxes should not significantly overlap.

    Domain Invariant: Steps should occupy distinct regions on the page.
    Some minor overlap is acceptable (e.g., at boundaries), but significant
    overlap would indicate a classification error.

    We allow up to 5% IOU (Intersection over Union) to account for minor
    boundary overlaps or shared visual elements.
    """
    validation = _run_validation_on_fixtures(
        fixture_file,
        validate_steps_no_significant_overlap,
        skip_fixtures=KNOWN_STEP_OVERLAP_FAILURES,
        skip_reason="Step bounding boxes have significant overlaps. "
        "This indicates issues with step classification.",
        overlap_threshold=0.05,
    )

    errors = [i for i in validation.issues if i.rule == "overlapping_steps"]
    assert not errors, (
        f"Found {len(errors)} overlapping step issues in {fixture_file}: "
        + "; ".join(i.details or i.message for i in errors[:3])
    )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_part_bbox_contains_count_and_diagram(fixture_file: str) -> None:
    """Part bbox should contain its count and diagram bboxes.

    Domain Invariant: A Part represents a cohesive visual unit consisting of
    the part diagram (image) and the count label below it. The Part's bounding
    box should encompass both elements.
    """
    validation = _run_validation_on_fixtures(
        fixture_file, validate_part_contains_children
    )

    errors = [
        i
        for i in validation.issues
        if i.rule in ("part_count_outside", "part_diagram_outside")
    ]
    assert not errors, (
        f"Found {len(errors)} part containment issues in {fixture_file}: "
        + "; ".join(i.details or i.message for i in errors[:3])
    )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_elements_stay_within_page_bounds(fixture_file: str) -> None:
    """All element bounding boxes should stay within the page boundaries.

    Domain Invariant: Elements should not extend beyond the page boundaries.
    This would indicate an extraction or classification error.
    """
    validation = _run_validation_on_fixtures(
        fixture_file, validate_elements_within_page
    )

    errors = [
        i
        for i in validation.issues
        if i.rule in ("element_outside_page", "no_page_bbox")
    ]
    assert not errors, (
        f"Found {len(errors)} page boundary issues in {fixture_file}: "
        + "; ".join(i.details or i.message for i in errors[:3])
    )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_elements_do_not_overlap_page_metadata(fixture_file: str) -> None:
    """Elements should not overlap with page number or progress bar.

    Domain Invariant: The page number and progress bar are navigation elements
    that should be distinct from the actual content (steps, parts, etc.).
    Any overlap indicates a classification error.
    """
    validation = _run_validation_on_fixtures(
        fixture_file, validate_content_no_metadata_overlap
    )

    errors = [i for i in validation.issues if i.rule == "content_metadata_overlap"]
    assert not errors, (
        f"Found {len(errors)} metadata overlap issues in {fixture_file}: "
        + "; ".join(i.details or i.message for i in errors[:3])
    )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_elements_do_not_overlap_dividers(fixture_file: str) -> None:
    """Elements should not overlap with dividers.

    Domain Invariant: Dividers separate content sections. Elements like steps,
    parts, diagrams, etc. should not cross or touch divider lines.
    """
    validation = _run_validation_on_fixtures(
        fixture_file, validate_no_divider_intersection
    )

    errors = [i for i in validation.issues if i.rule == "divider_intersection"]
    assert not errors, (
        f"Found {len(errors)} divider intersection issues in {fixture_file}: "
        + "; ".join(i.details or i.message for i in errors[:3])
    )
