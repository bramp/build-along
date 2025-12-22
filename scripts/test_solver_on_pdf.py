#!/usr/bin/env python3
"""Test the constraint solver on a real PDF fixture.

This script runs classification with and without the solver and compares results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from build_a_long.pdf_extract.classifier.classifier import Classifier
from build_a_long.pdf_extract.classifier.classifier_config import ClassifierConfig
from build_a_long.pdf_extract.extractor import ExtractionResult
from build_a_long.pdf_extract.fixtures import FIXTURES_DIR, load_classifier_config

# Simple fixture with parts
FIXTURE_FILE = "6509377_page_010_raw.json"


def load_page():
    """Load a page from the fixture."""
    fixture_path = FIXTURES_DIR / FIXTURE_FILE
    extraction = ExtractionResult.model_validate_json(fixture_path.read_text())
    return extraction.pages[0]


def count_elements(result) -> dict[str, int]:
    """Count constructed elements by label."""
    counts: dict[str, int] = {}
    for label, candidates in result.get_all_candidates().items():
        count = sum(1 for c in candidates if c.constructed is not None)
        if count > 0:
            counts[label] = count
    return counts


def main():
    # Load test data
    page = load_page()
    config = load_classifier_config("6509377")
    print(f"Testing with fixture: {FIXTURE_FILE}")
    print(f"Page has {len(page.blocks)} blocks")
    print()

    # Run without solver (explicitly disable with empty set since default now uses solver)
    classifier_no_solver = Classifier(config, use_solver_for=set())
    result_no_solver = classifier_no_solver.classify(page)
    counts_no_solver = count_elements(result_no_solver)

    print("=== WITHOUT SOLVER ===")
    for label, count in sorted(counts_no_solver.items()):
        print(f"  {label}: {count}")
    print()

    # Run with solver enabled for parts labels (now the default)
    classifier_with_solver = Classifier(config)  # Uses DEFAULT_SOLVER_LABELS
    result_with_solver = classifier_with_solver.classify(page)
    counts_with_solver = count_elements(result_with_solver)

    print("=== WITH SOLVER (parts labels - default) ===")
    for label, count in sorted(counts_with_solver.items()):
        print(f"  {label}: {count}")
    print()

    # Compare
    print("=== COMPARISON ===")
    all_labels = set(counts_no_solver.keys()) | set(counts_with_solver.keys())
    has_diff = False
    for label in sorted(all_labels):
        no_solver = counts_no_solver.get(label, 0)
        with_solver = counts_with_solver.get(label, 0)
        if no_solver != with_solver:
            print(f"  {label}: {no_solver} -> {with_solver} ({'CHANGED' if no_solver != with_solver else 'same'})")
            has_diff = True
        else:
            print(f"  {label}: {no_solver} (same)")

    if not has_diff:
        print("\n✅ Results are identical!")
    else:
        print("\n⚠️  Results differ!")


if __name__ == "__main__":
    main()
