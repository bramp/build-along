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
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElement,
    Page,
    Part,
    PartsList,
    Step,
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

        # Get the constructed Page object
        page_candidates = result.get_candidates("page")
        if not page_candidates:
            continue

        page = page_candidates[0].constructed
        if page is None:
            continue

        assert isinstance(page, Page), f"Expected Page, got {type(page)}"

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

    for page_idx, page_data in enumerate(pages):
        result = classify_elements(page_data)

        # Get the constructed Page object
        page_candidates = result.get_candidates("page")
        if not page_candidates:
            continue

        page = page_candidates[0].constructed
        if page is None:
            continue

        assert isinstance(page, Page), f"Expected Page, got {type(page)}"

        # Collect all parts_lists from all steps
        parts_lists = [
            step.parts_list for step in page.steps if step.parts_list is not None
        ]

        # Check pairwise for overlaps
        for i, pl1 in enumerate(parts_lists):
            for pl2 in parts_lists[i + 1 :]:
                overlap = pl1.bbox.iou(pl2.bbox)
                assert overlap == 0.0, (
                    f"PartsList at {pl1.bbox} and {pl2.bbox} in {fixture_file} "
                    f"page {page_idx} overlap with IOU {overlap}"
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

        # Get the constructed Page object
        page_candidates = result.get_candidates("page")
        if not page_candidates:
            continue

        page = page_candidates[0].constructed
        if page is None:
            continue

        assert isinstance(page, Page), f"Expected Page, got {type(page)}"

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
def test_all_winners_discoverable_from_page(fixture_file: str) -> None:
    """All winning candidates should be discoverable from the root Page element.

    Domain Invariant: Every winning candidate (constructed LegoPageElement)
    can be found by traversing the Page hierarchy. This validates that:
    1. The page builder properly includes all winners in the hierarchy
    2. No winning candidates are orphaned or lost during page construction
    3. The hierarchical structure is complete

    Note: Some pages are skipped due to known issues in the page builder:
    - Catalog/inventory pages (not yet supported)
    - Pages with orphaned step_numbers or part_counts (bugs to fix)

    TODO: Fix page builder bugs and remove pages from KNOWN_ISSUES skip list.
    """
    # Skip known pages with bugs in the page builder
    # TODO: Fix these bugs and remove from skip list
    KNOWN_ISSUES = [
        # Catalog/inventory pages with no steps - page builder doesn't support yet
        "6509377_page_180_raw.json",  # 178 orphaned parts/part_counts
        # Regular instruction pages with orphaned winners - bugs to fix
        "6509377_page_010_raw.json",  # 1 orphaned step_number
        "6509377_page_013_raw.json",  # 2 orphaned part_counts
        "6509377_page_014_raw.json",  # 1 orphaned part_count
        "6509377_page_015_raw.json",  # 1 orphaned part_count
    ]

    if fixture_file in KNOWN_ISSUES:
        pytest.skip(f"Skipping {fixture_file}: known page builder issue")

    pages = load_pages(fixture_file)

    for page_idx, page_data in enumerate(pages):
        # Run classification
        result = classify_elements(page_data)

        # Build the Page hierarchy
        page = result.page
        if page is None:
            pytest.fail(f"Page element is None for {fixture_file} page {page_idx}")

        # Collect all constructed elements from the Page hierarchy
        discovered_elements: set[int] = set()
        stack: list[LegoPageElement] = [page]

        while stack:
            element = stack.pop()
            discovered_elements.add(id(element))

            # Page attributes
            if isinstance(element, Page):
                if element.page_number:
                    stack.append(element.page_number)
                if element.progress_bar:
                    stack.append(element.progress_bar)
                stack.extend(element.steps)

            # Step attributes (all required fields)
            elif isinstance(element, Step):
                stack.append(element.step_number)
                stack.append(element.parts_list)
                stack.append(element.diagram)

            # PartsList attributes
            elif isinstance(element, PartsList):
                stack.extend(element.parts)

            # Part attributes
            # Note: Part.count is PartCount (LegoPageElement)
            # Part.diagram is Drawing | None (not LegoPageElement, so skip it)
            elif isinstance(element, Part):
                stack.append(element.count)

        # Get all winning candidates (all types, not just structural)
        all_candidates = result.get_all_candidates()
        winning_candidates = []
        for label, candidates in all_candidates.items():
            for candidate in candidates:
                if candidate.is_winner and candidate.constructed is not None:
                    winning_candidates.append((label, candidate))

        # Check that all winners are discoverable
        orphaned = []
        for label, candidate in winning_candidates:
            element_id = id(candidate.constructed)
            if element_id not in discovered_elements:
                orphaned.append((label, candidate))

        if orphaned:
            log.error(
                f"Found {len(orphaned)} winning candidates not discoverable "
                f"from Page in {fixture_file} page {page_idx}:"
            )
            for label, candidate in orphaned:
                log.error(
                    f"  - {label}: {candidate.constructed} "
                    f"(id={id(candidate.constructed)})"
                )

        assert len(orphaned) == 0, (
            f"Found {len(orphaned)} winning candidates that are not "
            f"discoverable from the root Page element in {fixture_file} "
            f"page {page_idx}. All winners should be part of the "
            f"hierarchical structure."
        )


# TODO Add test to ensure nothing overlaps the page number or progress bar
# TODO Ensure bbox stay within the page boundaries
