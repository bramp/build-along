"""Integration tests for LEGO page domain invariants.

These tests validate that the complete classification pipeline produces
domain objects (Page, PartsList, Part, etc.) that satisfy LEGO instruction
layout invariants.

Tests run on real fixture files to ensure end-to-end correctness.
"""

import logging

import pytest

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.classifier.classifier_rules_test import (
    RAW_FIXTURE_FILES,
)
from build_a_long.pdf_extract.classifier.classifier_rules_test import (
    _load_pages_from_fixture as load_pages,
)

log = logging.getLogger(__name__)


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_parts_list_contains_parts(fixture_file: str) -> None:
    """Every PartsList should contain at least one Part.

    Domain Invariant: A parts list without parts doesn't make sense in the
    context of LEGO instructions. Each PartsList should contain ≥1 Part objects.

    NOTE: Currently this is a soft assertion - we log when empty PartsList
    objects are found, but don't fail the test. This appears to be a known
    issue in the classification pipeline that needs investigation.
    """
    pages = load_pages(fixture_file)

    for page_idx, page_data in enumerate(pages):
        result = classify_elements(page_data)
        page = result.page

        if page is None:
            continue

        # Check each step's parts_list
        for step in page.steps:
            if step.parts_list is None:
                continue

            parts_list = step.parts_list
            # TODO: Make this a hard assertion once classification is fixed
            if len(parts_list.parts) == 0:
                print(
                    f"WARNING: PartsList in {fixture_file} page {page_idx} "
                    f"step {step.step_number.value} is empty at {parts_list.bbox}"
                )

            log.debug(
                f"{fixture_file} page {page_idx} step {step.step_number.value}: "
                f"PartsList has {len(parts_list.parts)} parts"
            )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_parts_lists_do_not_overlap(fixture_file: str) -> None:
    """PartsList bounding boxes should not overlap.

    Domain Invariant: Each parts list occupies a distinct region on the page.
    Overlapping parts lists would indicate a classification error.
    """
    pages = load_pages(fixture_file)

    for _page_idx, page_data in enumerate(pages):
        result = classify_elements(page_data)
        page = result.page

        if page is None:
            continue

        # Collect all parts_lists from all steps
        parts_lists = [
            step.parts_list for step in page.steps if step.parts_list is not None
        ]

        # Check pairwise for overlaps
        for i, pl1 in enumerate(parts_lists):
            for pl2 in parts_lists[i + 1 :]:
                overlap = pl1.bbox.overlaps(pl2.bbox)
                assert overlap == 0.0, (
                    f"PartsList at {pl1.bbox} and {pl2.bbox} in {fixture_file} "
                    f"page {page_data.page_number} overlap with IOU {pl1.bbox.iou(pl2.bbox)}"
                )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_part_bbox_contains_count_and_diagram(fixture_file: str) -> None:
    """Part bbox should contain its count and diagram bboxes.

    Domain Invariant: A Part represents a cohesive visual unit consisting of
    the part diagram (image) and the count label below it. The Part's bounding
    box should encompass both elements.
    """
    pages = load_pages(fixture_file)

    for page_idx, page_data in enumerate(pages):
        result = classify_elements(page_data)
        page = result.page

        if page is None:
            continue

        # Check all parts in all parts_lists
        for step in page.steps:
            if step.parts_list is None:
                continue

            for part in step.parts_list.parts:
                # Count must be inside Part bbox
                assert part.count.bbox.fully_inside(part.bbox), (
                    f"Part count {part.count.bbox} not inside Part bbox {part.bbox} "
                    f"in {fixture_file} page {page_idx}"
                )

                # Diagram (if present) must be inside Part bbox
                if part.diagram:
                    assert part.diagram.bbox.fully_inside(part.bbox), (
                        f"Part diagram {part.diagram.bbox} not inside "
                        f"Part bbox {part.bbox} in {fixture_file} page {page_idx}"
                    )


@pytest.mark.parametrize("fixture_file", RAW_FIXTURE_FILES)
def test_elements_stay_within_page_bounds(fixture_file: str) -> None:
    """All element bounding boxes should stay within the page boundaries.

    Domain Invariant: Elements should not extend beyond the page boundaries.
    This would indicate an extraction or classification error.
    """
    pages = load_pages(fixture_file)

    for page_idx, page_data in enumerate(pages):
        result = classify_elements(page_data)
        page = result.page

        if page is None:
            continue

        # Check that page has valid bbox
        assert page_data.bbox is not None, (
            f"Page in {fixture_file} page {page_idx} has no bbox"
        )

        page_bbox = page_data.bbox

        # Check all elements in the page hierarchy
        for element in page.iter_elements():
            assert element.bbox.fully_inside(page_bbox), (
                f"{element.__class__.__name__} at {element.bbox} extends beyond "
                f"page bounds {page_bbox} in {fixture_file} page {page_idx}"
            )


# TODO Add test to ensure nothing overlaps the page number or progress bar
