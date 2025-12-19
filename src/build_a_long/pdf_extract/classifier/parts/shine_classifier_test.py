"""Tests for shine classification."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.classifier.classifier_rules_test import (
    _load_config_for_fixture,
)
from build_a_long.pdf_extract.classifier.classifier_rules_test import (
    _load_pages_from_fixture as load_pages,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import PartImage, Shine


def test_shine_classification() -> None:
    """Test that shines are detected and associated with part images."""
    # This page contains a shine on step 16 (image_8 and drawing id=70)
    fixture_file = "6509377_page_015_raw.json"
    pages = load_pages(fixture_file)
    config = _load_config_for_fixture(fixture_file)
    page_data = pages[0]

    result = classify_elements(page_data, config=config)

    # Check if shine candidates were created
    shine_candidates = result.get_scored_candidates(
        "shine", valid_only=False, exclude_failed=True
    )
    assert len(shine_candidates) > 0

    # The shine should be consumed by a PartImage
    # Find the PartImage candidate that consumed the shine
    part_image_candidates = result.get_winners_by_score("part_image", PartImage)
    shine_consumers = [pi for pi in part_image_candidates if pi.shine is not None]

    assert len(shine_consumers) == 1
    shine_part_image = shine_consumers[0]

    # Verify it's the correct shine drawing (check bbox match as proxy)
    # The shine element doesn't store the original block ID directly,
    # but we can check bbox
    assert isinstance(shine_part_image.shine, Shine)
    # BBox should match drawing id=67/68: [114.88..., 258.48...]
    # This is the shine on the part image in step 16
    assert shine_part_image.shine.bbox.x0 > 110
    assert shine_part_image.shine.bbox.x0 < 120
    assert shine_part_image.shine.bbox.y0 > 250
    assert shine_part_image.shine.bbox.y0 < 270
