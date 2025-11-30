"""Tests for shine classification."""

from typing import Any, cast

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

    result = classify_elements(page_data, config)

    # Check if 'shine' candidates were created
    shine_candidates = result.get_candidates("shine")
    assert len(shine_candidates) > 0, "No shine candidates found"

    # Verify we found the expected shine (id=70)
    found_shine_70 = False
    for cand in shine_candidates:
        for block in cand.source_blocks:
            if block.id == 70:
                found_shine_70 = True
                break
    assert found_shine_70, "Expected shine (id=70) not found"

    # Check if part_image associated with shine
    part_image_candidates = result.get_candidates("part_image")

    # Find part_image for image_8
    image_8_cand = None
    for cand in part_image_candidates:
        # Check score details for image_8
        details = cast(Any, cand.score_details)
        if details.image.image_id == "image_8":
            image_8_cand = cand
            break

    assert image_8_cand is not None, "PartImage for image_8 not found"

    # Verify it has a shine candidate
    details = cast(Any, image_8_cand.score_details)
    assert details.shine_candidate is not None, (
        "PartImage for image_8 should have a shine candidate"
    )

    # Verify constructed element has shine
    part_image = result.build(image_8_cand)
    assert isinstance(part_image, PartImage)
    assert isinstance(part_image.shine, Shine)
    assert part_image.shine.bbox.width > 0
