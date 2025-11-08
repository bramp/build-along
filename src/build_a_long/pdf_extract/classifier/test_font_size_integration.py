#!/usr/bin/env python3
"""Quick test to verify font size hints are being used in scoring."""

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.font_size_hints import FontSizeHints
from build_a_long.pdf_extract.classifier.page_number_classifier import (
    PageNumberClassifier,
)
from build_a_long.pdf_extract.classifier.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.extractor import PageData
from build_a_long.pdf_extract.extractor.page_blocks import Text


def test_part_count_with_font_hints():
    """Test that PartCountClassifier uses font size hints."""
    # Create font size hints
    hints = FontSizeHints(
        part_count_size=10.0,
        catalog_part_count_size=None,
        catalog_element_id_size=None,
        step_number_size=None,
        step_repeat_size=None,
        page_number_size=None,
        remaining_font_sizes={},
    )

    # Create config with hints
    config = ClassifierConfig(font_size_hints=hints)
    classifier = PartCountClassifier(config, None)  # type: ignore

    # Create page data with part count text
    # One with matching size, one with different size
    matching_text = Text(text="2x", bbox=BBox(0, 0, 10, 10), id=1)  # height = 10
    different_text = Text(text="3x", bbox=BBox(0, 0, 15, 15), id=2)  # height = 15

    page_data = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[matching_text, different_text],
    )

    # Classify
    result = ClassificationResult(page_data=page_data)
    classifier.evaluate(page_data, result)

    candidates = result.get_candidates("part_count")
    assert len(candidates) == 2

    # Find the candidates
    matching_candidate = next(c for c in candidates if c.source_block == matching_text)
    different_candidate = next(
        c for c in candidates if c.source_block == different_text
    )

    # The matching size should have a higher score
    print(f"Matching size score: {matching_candidate.score:.3f}")
    print(f"Different size score: {different_candidate.score:.3f}")
    print(
        f"Matching score details: {matching_candidate.score_details}"  # type: ignore
    )
    print(
        f"Different score details: {different_candidate.score_details}"  # type: ignore
    )

    assert (
        matching_candidate.score > different_candidate.score
    ), "Matching font size should score higher"
    print("✓ PartCountClassifier uses font size hints correctly")


def test_step_number_with_font_hints():
    """Test that StepNumberClassifier uses font size hints."""
    hints = FontSizeHints(
        part_count_size=None,
        catalog_part_count_size=None,
        catalog_element_id_size=None,
        step_number_size=15.0,
        step_repeat_size=None,
        page_number_size=None,
        remaining_font_sizes={},
    )

    config = ClassifierConfig(font_size_hints=hints)
    classifier = StepNumberClassifier(config, None)  # type: ignore

    # Create texts with step numbers at different sizes
    matching_text = Text(text="1", bbox=BBox(10, 10, 25, 25), id=1)  # height = 15
    different_text = Text(text="2", bbox=BBox(10, 40, 30, 60), id=2)  # height = 20

    page_data = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[matching_text, different_text],
    )

    result = ClassificationResult(page_data=page_data)
    classifier.evaluate(page_data, result)

    candidates = result.get_candidates("step_number")
    assert len(candidates) == 2

    matching_candidate = next(c for c in candidates if c.source_block == matching_text)
    different_candidate = next(
        c for c in candidates if c.source_block == different_text
    )

    print(f"\nMatching size score: {matching_candidate.score:.3f}")
    print(f"Different size score: {different_candidate.score:.3f}")
    print(
        f"Matching score details: {matching_candidate.score_details}"  # type: ignore
    )
    print(
        f"Different score details: {different_candidate.score_details}"  # type: ignore
    )

    assert (
        matching_candidate.score > different_candidate.score
    ), "Matching font size should score higher"
    print("✓ StepNumberClassifier uses font size hints correctly")


def test_page_number_with_font_hints():
    """Test that PageNumberClassifier uses font size hints."""
    hints = FontSizeHints(
        part_count_size=None,
        catalog_part_count_size=None,
        catalog_element_id_size=None,
        step_number_size=None,
        step_repeat_size=None,
        page_number_size=8.0,
        remaining_font_sizes={},
    )

    config = ClassifierConfig(font_size_hints=hints)
    classifier = PageNumberClassifier(config, None)  # type: ignore

    # Create page numbers at bottom of page with different sizes
    # Both in correct position, but different sizes
    matching_text = Text(text="1", bbox=BBox(10, 90, 18, 98), id=1)  # height = 8
    different_text = Text(text="2", bbox=BBox(80, 90, 92, 102), id=2)  # height = 12

    page_data = PageData(
        page_number=1,
        bbox=BBox(0, 0, 100, 100),
        blocks=[matching_text, different_text],
    )

    result = ClassificationResult(page_data=page_data)
    classifier.evaluate(page_data, result)

    candidates = result.get_candidates("page_number")
    assert len(candidates) == 2

    matching_candidate = next(c for c in candidates if c.source_block == matching_text)
    different_candidate = next(
        c for c in candidates if c.source_block == different_text
    )

    print(f"\nMatching size score: {matching_candidate.score:.3f}")
    print(f"Different size score: {different_candidate.score:.3f}")
    print(
        f"Matching score details: {matching_candidate.score_details}"  # type: ignore
    )
    print(
        f"Different score details: {different_candidate.score_details}"  # type: ignore
    )

    assert (
        matching_candidate.score > different_candidate.score
    ), "Matching font size should score higher"
    print("✓ PageNumberClassifier uses font size hints correctly")


if __name__ == "__main__":
    test_part_count_with_font_hints()
    test_step_number_with_font_hints()
    test_page_number_with_font_hints()
    print("\n✓ All font size hint integration tests passed!")

