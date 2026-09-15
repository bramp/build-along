#!/usr/bin/env python3
"""Test script to debug classifiers."""

import logging
logging.basicConfig(level=logging.DEBUG, format='%(name)s: %(message)s')

from pathlib import Path

from build_a_long.pdf_extract.extractor.extractor import ExtractionResult
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.classifier.classifier import Classifier

fixture_path = Path("src/build_a_long/pdf_extract/fixtures/6509377_page_010_raw.json")
extraction = ExtractionResult.model_validate_json(fixture_path.read_text())
page = extraction.pages[0]

config = ClassifierConfig()

# Create classifier with solver
classifier = Classifier(config)
print(f"Solver labels: {classifier.use_solver_for}")

result = classifier.classify(page)

print()
print("=== All open_bag candidates ===")
for cand in result.candidates.get("open_bag", []):
    selected = result.is_solver_selected(cand)
    constructed = result.get_constructed(cand)
    print(f"  {cand.bbox} score={cand.score:.2f} selected={selected} constructed={constructed is not None}")

print()
print("=== All subassembly candidates ===")
for cand in result.candidates.get("subassembly", []):
    selected = result.is_solver_selected(cand)
    constructed = result.get_constructed(cand)
    print(f"  {cand.bbox} score={cand.score:.2f} selected={selected} constructed={constructed is not None}")

print()
print("=== Candidates competing for block 0 (text '1') ===")
for label, candidates in result.candidates.items():
    for cand in candidates:
        # Check if block 0 is in source_blocks
        if any(b.id == 0 for b in cand.source_blocks):
            selected = result.is_solver_selected(cand)
            constructed = result.get_constructed(cand)
            print(f"  {label}: bbox={cand.bbox} score={cand.score:.2f} selected={selected} constructed={constructed is not None}")

print()
print("=== Failure reasons ===")
for label, candidates in result.candidates.items():
    for cand in candidates:
        if cand.id in result._failure_reasons:
            print(f"  {label} {cand.bbox}: {result._failure_reasons[cand.id]}")

print()
print("=== Bag number candidates detailed ===")
for cand in result.candidates.get("bag_number", []):
    selected = result.is_solver_selected(cand)
    constructed = result.get_constructed(cand)
    failure = result._failure_reasons.get(cand.id, "no failure")
    blocks = [b.id for b in cand.source_blocks]
    print(f"  {cand.bbox} score={cand.score:.2f} selected={selected} constructed={constructed is not None} failure={failure}")
    print(f"    blocks: {blocks}")

print()
print("=== OpenBag candidates detailed ===")
for cand in result.candidates.get("open_bag", []):
    selected = result.is_solver_selected(cand)
    constructed = result.get_constructed(cand)
    failure = result._failure_reasons.get(cand.id, "no failure")
    blocks = [b.id for b in cand.source_blocks]
    print(f"  {cand.bbox} score={cand.score:.2f} selected={selected} constructed={constructed is not None} failure={failure}")
    print(f"    blocks: {blocks[:20]}..." if len(blocks) > 20 else f"    blocks: {blocks}")

print()
print("=== SubAssembly candidates detailed ===")
for cand in result.candidates.get("subassembly", []):
    selected = result.is_solver_selected(cand)
    constructed = result.get_constructed(cand)
    failure = result._failure_reasons.get(cand.id, "no failure")
    blocks = [b.id for b in cand.source_blocks]
    print(f"  {cand.bbox} score={cand.score:.2f} selected={selected} constructed={constructed is not None} failure={failure}")
    print(f"    blocks: {blocks[:20]}..." if len(blocks) > 20 else f"    blocks: {blocks}")
