"""Debug script to trace Part IDs through the classification flow."""

import sys
from pathlib import Path

sys.path.insert(0, "src")

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.fixtures import (
    FIXTURES_DIR,
    extract_element_id,
    load_classifier_config,
)

# Load raw page data
fixture_file = "6509377_page_016_raw.json"
fixture_path = FIXTURES_DIR / fixture_file

extraction = ExtractionResult.model_validate_json(fixture_path.read_text())
page = extraction.pages[0]
print(f"Loaded page {page.page_number} with {len(page.blocks)} blocks")

# Load classifier config
element_id = extract_element_id(fixture_file)
config = load_classifier_config(element_id)

# Run classification
result = classify_elements(page, config)

# Get all part candidates and their constructed Parts
print("\n=== Part Candidates and Constructed Parts ===")
part_candidates = result.get_scored_candidates("part")
for pc in part_candidates:
    elem = result.get_constructed(pc)
    if elem:
        print(f"  Candidate.id={pc.id} -> Part.id={elem.id}, bbox={elem.bbox}")
        print(f"    count.bbox={elem.count.bbox}, count.count={elem.count.count}")
        print(f"    diagram.bbox={elem.diagram.bbox}")
        # Check score_details
        score = pc.score_details
        print(
            f"    score.part_count_candidate.id={score.part_count_candidate.id}, "
            f"score.part_image_candidate.id={score.part_image_candidate.id}"
        )
    else:
        print(f"  Candidate.id={pc.id} -> NOT BUILT, bbox={pc.bbox}")

# Get steps and trace their parts
print("\n=== Steps and their PartsList Parts ===")
step_candidates = result.get_built_candidates("step")
for step_cand in step_candidates:
    step = result.get_constructed(step_cand)
    print(f"  Step {step.step_number.value}:")
    if step.parts_list:
        for part in step.parts_list.parts:
            print(f"    Part.id={part.id}, bbox={part.bbox}")
    else:
        print("    (no parts_list)")

# Now simulate what _get_standalone_parts does
print("\n=== Simulating _get_standalone_parts ===")

# Build all parts (like _build_parts does)
all_parts = []
for candidate in result.get_scored_candidates("part"):
    elem = result.get_constructed(candidate)
    if elem:
        all_parts.append(elem)

print(f"all_parts has {len(all_parts)} parts:")
for p in all_parts:
    print(f"  Part.id={p.id}")

# Build steps (like _build_steps does)
steps = []
for step_cand in result.get_built_candidates("step"):
    step = result.get_constructed(step_cand)
    steps.append(step)

# Collect Part.ids from steps
parts_in_steps = set()
for step in steps:
    if step.parts_list:
        for part in step.parts_list.parts:
            parts_in_steps.add(part.id)

print(f"\nparts_in_steps has {len(parts_in_steps)} Part.ids:")
for pid in sorted(parts_in_steps):
    print(f"  Part.id={pid}")

# Find standalone (should be empty for instruction pages)
standalone = [p for p in all_parts if p.id not in parts_in_steps]
print(f"\nstandalone has {len(standalone)} parts:")
for p in standalone:
    print(f"  Part.id={p.id}, bbox={p.bbox}")

# Check if any Part.id appears in both
all_part_ids = {p.id for p in all_parts}
print(f"\nall_part_ids: {sorted(all_part_ids)}")
print(f"parts_in_steps: {sorted(parts_in_steps)}")
print(f"Intersection: {sorted(all_part_ids & parts_in_steps)}")
print(f"In all_parts but not steps: {sorted(all_part_ids - parts_in_steps)}")
