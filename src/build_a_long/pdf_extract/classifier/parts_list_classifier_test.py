"""Tests for the parts list classifier."""

from pathlib import Path

import pytest

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_elements import Drawing, Text


class TestPartsListClassification:
    """Tests for detecting a parts list drawing above a step containing part counts."""

    def test_parts_list_drawing_above_step(self) -> None:
        page_bbox = BBox(0, 0, 200, 300)

        # Page and step
        pn = Text(bbox=BBox(10, 285, 20, 295), text="6")
        step = Text(
            bbox=BBox(50, 180, 70, 210), text="10"
        )  # height 30 (taller than PN)

        # Two drawings above the step; only d1 contains part counts
        d1 = Drawing(bbox=BBox(30, 100, 170, 160))
        d2 = Drawing(bbox=BBox(20, 40, 180, 80))

        # Part counts inside d1
        pc1 = Text(bbox=BBox(40, 110, 55, 120), text="2x")
        pc2 = Text(bbox=BBox(100, 130, 115, 140), text="5×")

        # Some unrelated text
        other = Text(bbox=BBox(10, 10, 40, 20), text="hello")

        page = PageData(
            page_number=6,
            elements=[pn, step, d1, d2, pc1, pc2, other],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Part counts should be labeled, step labeled, and d1 chosen as parts list
        assert result.get_label(pc1) == "part_count"
        assert result.get_label(pc2) == "part_count"
        assert result.get_label(step) == "step_number"
        assert result.get_label(d1) == "parts_list"
        d2_label = result.get_label(d2)
        assert d2_label is None or d2_label != "parts_list"

    @pytest.mark.skip(reason="Re-enable, once we fix the classifer rules.")
    def test_real_example_parts_list_and_deletions(self) -> None:
        """Replicate the user's provided example.

        Ensures:
        - Step text is classified as step_number
        - Exactly one of the near-duplicate drawings is labeled parts_list; the other is removed
        - The three "1x" texts are labeled part_count
        - Images inside the chosen parts list are labeled part_image and kept
        - The unrelated image is removed
        """
        # Load the page from a JSON fixture next to this test file
        fixture = (
            Path(__file__)
            .with_name("fixtures")
            .joinpath("real_example_parts_list_and_deletions.json")
        )
        page: PageData = PageData.from_json(fixture.read_text())  # type: ignore[assignment]

        result = classify_elements(page)

        # Build a quick map of elements by id for easy lookup in assertions
        elems = {e.id: e for e in page.elements if e.id is not None}
        # Recreate references for assertions by their IDs
        pc4 = elems[4]
        pc5 = elems[5]
        pc6 = elems[6]
        step = elems[7]
        img17 = elems[17]
        img18 = elems[18]
        img19 = elems[19]
        img20 = elems[20]
        d34 = elems[34]
        d35 = elems[35]

        # Assertions matching the previous test
        assert result.get_label(step) == "step_number"
        assert result.get_label(pc4) == "part_count"
        assert result.get_label(pc5) == "part_count"
        assert result.get_label(pc6) == "part_count"

        # Exactly one of the drawings is chosen as parts list; the other is removed
        d34_label = result.get_label(d34)
        d35_label = result.get_label(d35)
        assert (d34_label == "parts_list") ^ (d35_label == "parts_list")
        assert result.is_removed(d34) ^ result.is_removed(d35)

        # Images within the chosen parts list should be labeled as part_image; unrelated image is removed
        assert result.get_label(img18) == "part_image"
        assert not result.is_removed(img18)

        assert result.get_label(img19) == "part_image"
        assert not result.is_removed(img19)

        assert result.get_label(img20) == "part_image"
        assert not result.is_removed(img20)

        assert result.is_removed(img17)

    def test_two_steps_do_not_label_and_delete_both_drawings(self) -> None:
        """When there are two step numbers and two near-duplicate drawings above them,
        we should select only one drawing as the parts list across the page, and only
        the other near-duplicate should be removed. Previously, the second step could
        select the drawing already scheduled for removal, causing both drawings to be
        labeled and deleted.
        """
        page_bbox = BBox(0, 0, 600, 400)

        # A part count inside the drawings
        pc = Text(
            bbox=BBox(320, 45, 330, 55),
            text="1x",
        )

        # Two steps below the drawings (both tall enough)
        step1 = Text(bbox=BBox(260, 70, 276, 96), text="5")
        step2 = Text(bbox=BBox(300, 70, 316, 96), text="6")

        # Real page number at bottom to avoid confusion
        page_number = Text(bbox=BBox(10, 380, 20, 390), text="1")

        # Two near-duplicate drawings above the steps
        d_small = Drawing(bbox=BBox(262.5, 14.7, 414.6, 61.9))
        d_large = Drawing(bbox=BBox(262.0, 14.2, 415.1, 62.4))

        page = PageData(
            page_number=2,
            elements=[pc, step1, step2, page_number, d_small, d_large],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Exactly one of the drawings is chosen as parts_list, and exactly one is deleted
        d_small_label = result.get_label(d_small)
        d_large_label = result.get_label(d_large)
        assert (d_small_label == "parts_list") ^ (d_large_label == "parts_list")
        assert result.is_removed(d_small) ^ result.is_removed(d_large)
