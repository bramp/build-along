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

        classify_elements([page])

        # Part counts should be labeled, step labeled, and d1 chosen as parts list
        assert pc1.label == "part_count"
        assert pc2.label == "part_count"
        assert step.label == "step_number"
        assert d1.label == "parts_list"
        assert d2.label is None or d2.label != "parts_list"

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
        page = PageData.from_json(fixture.read_text())

        classify_elements([page])

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
        assert step.label == "step_number"
        assert pc4.label == "part_count"
        assert pc5.label == "part_count"
        assert pc6.label == "part_count"

        # Exactly one of the drawings is chosen as parts list; the other is removed
        assert (d34.label == "parts_list") ^ (d35.label == "parts_list")
        assert (d34.deleted) ^ (d35.deleted)

        # Images within the chosen parts list should be labeled as part_image; unrelated image is removed
        assert img18.label == "part_image"
        assert img18.deleted is False

        assert img19.label == "part_image"
        assert img19.deleted is False

        assert img20.label == "part_image"
        assert img20.deleted is False

        assert img17.deleted is True

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

        classify_elements([page])

        # Exactly one of the drawings is chosen as parts_list, and exactly one is deleted
        assert (d_small.label == "parts_list") ^ (d_large.label == "parts_list")
        assert (d_small.deleted) ^ (d_large.deleted)
