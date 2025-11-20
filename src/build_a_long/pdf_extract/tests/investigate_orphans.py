#!/usr/bin/env python3
"""Debug script to investigate orphaned winners in specific fixture files."""

import logging
from pathlib import Path

from build_a_long.pdf_extract.classifier.classifier import classify_elements
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

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def collect_discovered_elements(page: Page) -> set[int]:
    """Collect all element IDs discoverable from the Page hierarchy."""
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
        elif isinstance(element, Part):
            stack.append(element.count)

    return discovered_elements


def investigate_file(fixture_file: str, page_idx: int = 0) -> None:
    """Investigate orphaned winners in a specific fixture file."""
    print(f"\n{'='*80}")
    print(f"Investigating: {fixture_file}")
    print(f"{'='*80}\n")

    pages = load_pages(fixture_file)
    page_data = pages[page_idx]

    # Run classification
    result = classify_elements(page_data)
    page = result.page

    if page is None:
        print("ERROR: Page is None!")
        return

    # Get discovered elements
    discovered_elements = collect_discovered_elements(page)
    print(f"Total discovered elements: {len(discovered_elements)}")

    # Get all winning candidates
    all_candidates = result.get_all_candidates()
    winning_candidates = []
    for label, candidates in all_candidates.items():
        for candidate in candidates:
            if candidate.is_winner and candidate.constructed is not None:
                winning_candidates.append((label, candidate))

    print(f"Total winning candidates: {len(winning_candidates)}")

    # Find orphaned winners
    orphaned = []
    for label, candidate in winning_candidates:
        element_id = id(candidate.constructed)
        if element_id not in discovered_elements:
            orphaned.append((label, candidate))

    print(f"Orphaned winners: {len(orphaned)}")

    if orphaned:
        print("\nOrphaned winners details:")
        for label, candidate in orphaned:
            print(f"\n  Label: {label}")
            print(f"  Type: {type(candidate.constructed).__name__}")
            print(f"  Element: {candidate.constructed}")
            print(f"  Bbox: {candidate.constructed.bbox if hasattr(candidate.constructed, 'bbox') else 'N/A'}")
            if hasattr(candidate.constructed, 'value'):
                print(f"  Value: {candidate.constructed.value}")

    # Show page structure
    print("\n" + "="*80)
    print("Page Structure:")
    print("="*80)
    print(f"Page has {len(page.steps)} steps")
    for i, step in enumerate(page.steps):
        print(f"\nStep {i}:")
        print(f"  step_number: {step.step_number.value if step.step_number else 'None'}")
        print(f"  parts_list: {len(step.parts_list.parts) if step.parts_list else 0} parts")
        print(f"  diagram: {step.diagram is not None}")


if __name__ == "__main__":
    # Investigate the known issue files
    investigate_file("6509377_page_010_raw.json")
    investigate_file("6509377_page_013_raw.json")
    investigate_file("6509377_page_014_raw.json")
    investigate_file("6509377_page_015_raw.json")
