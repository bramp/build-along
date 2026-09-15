#!/usr/bin/env python3
"""Debug script to investigate unconsumed blocks and their overlapping candidates."""

import json
from pathlib import Path

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.extractor.page_blocks import BBox
from build_a_long.pdf_extract.fixtures import (
    FIXTURES_DIR,
    extract_element_id,
    load_classifier_config,
)

# Load the raw page data
fixture_file = "6509377_page_072_raw.json"
fixture_path = FIXTURES_DIR / fixture_file
extraction = ExtractionResult.model_validate_json(fixture_path.read_text())
page_data = extraction.pages[0]

# Load classifier config with hints
element_id = extract_element_id(fixture_file)
config = load_classifier_config(element_id)

# Run classification
result = classify_elements(page_data, config)

# Get the page
cr = result  # ClassificationResult
page = cr.page
assert page is not None

print(f"Page {page.pdf_page_number}")
print(f"Unconsumed blocks: {page.unconsumed_blocks_count}")
print()


def bbox_overlaps(b1: BBox, b2: BBox) -> bool:
    """Check if two bboxes overlap."""
    return not (b1.x1 < b2.x0 or b2.x1 < b1.x0 or b1.y1 < b2.y0 or b2.y1 < b1.y0)


def bbox_contains(outer: BBox, inner: BBox) -> bool:
    """Check if outer bbox contains inner bbox."""
    return (
        outer.x0 <= inner.x0
        and outer.y0 <= inner.y0
        and outer.x1 >= inner.x1
        and outer.y1 >= inner.y1
    )


# Show all unconsumed blocks with overlapping/containing candidates
unconsumed = cr.get_unconsumed_blocks()
print(f"=== Unconsumed blocks ({len(unconsumed)}) ===\n")

for i, block in enumerate(unconsumed, 1):
    block_bbox = block.bbox
    print(f"{i}. {type(block).__name__} at {block_bbox}")

    # Find all candidates that overlap or contain this block
    overlapping = []
    containing = []

    for label in cr.get_labels():
        for candidate in cr.get_candidates(label):
            cand_bbox = candidate.bbox
            if bbox_contains(cand_bbox, block_bbox):
                selected = cr.is_solver_selected(candidate)
                containing.append(
                    f"{label} id={candidate.id} at {cand_bbox} selected={selected}"
                )
            elif bbox_overlaps(cand_bbox, block_bbox):
                selected = cr.is_solver_selected(candidate)
                overlapping.append(
                    f"{label} id={candidate.id} at {cand_bbox} selected={selected}"
                )

    if containing:
        print(f"   CONTAINED BY:")
        for c in containing:
            print(f"     - {c}")
    if overlapping:
        print(f"   OVERLAPS WITH:")
        for o in overlapping:
            print(f"     - {o}")
    if not containing and not overlapping:
        print(f"   NO OVERLAPPING CANDIDATES")
    print()

# Compare with expected golden file
expected_path = FIXTURES_DIR / "6509377_page_072_expected.json"
expected = json.loads(expected_path.read_text())
print(f"Expected unconsumed_blocks_count: {expected.get('unconsumed_blocks_count')}")
print(f"Actual unconsumed_blocks_count: {page.unconsumed_blocks_count}")
