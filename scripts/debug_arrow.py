#!/usr/bin/env python3
"""Debug script to investigate why arrows aren't being selected by solver."""

import sys

sys.path.insert(0, "src")

from pathlib import Path

from build_a_long.pdf_extract.classifier.classifier import Classifier
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.extractor.extractor import PageData

# Load page 014
raw_path = Path("src/build_a_long/pdf_extract/fixtures/6509377_page_014_raw.json")
page_data = PageData.model_validate_json(raw_path.read_text())

config = ClassifierConfig()
classifier = Classifier(config, use_constraint_solver=True)

# Run classification
result = classifier.classify(page_data)

# Check arrow candidates
print("\n=== ARROW CANDIDATES ===")
arrow_cands = result.get_candidates("arrow")
print(f"Total arrow candidates: {len(arrow_cands)}")
for cand in arrow_cands:
    selected = cand.id in result._solver_selected_ids if result._solver_selected_ids else "N/A"
    print(f"  Arrow at {cand.bbox}: score={cand.score:.3f}, selected={selected}")
    print(f"    Source blocks: {[b.id for b in cand.source_blocks]}")

# Check what's using the same blocks
print("\n=== BLOCK CONFLICTS ===")
for arrow_cand in arrow_cands:
    for block in arrow_cand.source_blocks:
        print(f"\nBlock {block.id} at {block.bbox}:")
        for label in ["shine", "part_image", "divider", "progress_bar_bar"]:
            for cand in result.get_candidates(label):
                if any(b.id == block.id for b in cand.source_blocks):
                    selected = (
                        cand.id in result._solver_selected_ids
                        if result._solver_selected_ids
                        else "N/A"
                    )
                    print(
                        f"  Also used by {label} at {cand.bbox}: "
                        f"score={cand.score:.3f}, selected={selected}"
                    )

# Check if arrow is in solver labels
print("\n=== SOLVER LABELS ===")
print(f"Arrow in solver labels: {'arrow' in classifier.use_solver_for}")
print(f"Solver labels: {classifier.use_solver_for}")

# Check solver selected IDs
print("\n=== SOLVER SELECTION ===")
if result._solver_selected_ids:
    print(f"Total selected: {len(result._solver_selected_ids)}")
    arrow_ids = {c.id for c in arrow_cands}
    selected_arrows = arrow_ids & result._solver_selected_ids
    print(f"Arrow IDs: {arrow_ids}")
    print(f"Selected arrow IDs: {selected_arrows}")
else:
    print("No solver selection (solver not used)")
