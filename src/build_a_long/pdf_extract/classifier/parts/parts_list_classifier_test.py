"""Tests for the parts list classifier."""

from build_a_long.pdf_extract.classifier import ClassifierConfig
from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


class TestPartsListClassification:
    """Tests for detecting a parts list drawing above a step containing part counts."""

    def test_parts_list_drawing_above_step(self) -> None:
        page_bbox = BBox(0, 0, 200, 300)

        # Page and step
        pn = Text(id=0, bbox=BBox(10, 285, 20, 295), text="6")
        step = Text(
            id=1, bbox=BBox(50, 180, 70, 210), text="10"
        )  # height 30 (taller than PN)

        # Two drawings above the step; only d1 contains part counts
        d1 = Drawing(id=2, bbox=BBox(30, 100, 170, 160))
        d2 = Drawing(id=3, bbox=BBox(20, 40, 180, 80))

        # Part counts inside d1
        pc1 = Text(id=4, bbox=BBox(40, 135, 55, 145), text="2x")
        pc2 = Text(id=5, bbox=BBox(100, 145, 115, 155), text="5×")

        # Images inside d1, above the part counts
        img1 = Image(id=7, bbox=BBox(40, 110, 55, 125))
        img2 = Image(id=8, bbox=BBox(100, 120, 115, 135))

        # Some unrelated text
        other = Text(id=6, bbox=BBox(10, 10, 40, 20), text="hello")

        page = PageData(
            page_number=6,
            blocks=[pn, step, d1, d2, pc1, pc2, img1, img2, other],
            bbox=page_bbox,
        )

        result = classify_elements(page, config=ClassifierConfig())

        # Check the Page structure was built correctly
        assert result.page is not None
        assert len(result.page.steps) == 1

        step_elem = result.page.steps[0]
        assert step_elem.step_number.value == 10
        assert step_elem.parts_list is not None
        assert len(step_elem.parts_list.parts) == 2  # pc1 and pc2

        # Verify the part counts in the parts list
        part_counts = sorted([p.count.count for p in step_elem.parts_list.parts])
        assert part_counts == [2, 5]

    def test_two_steps_do_not_label_and_delete_both_drawings(self) -> None:
        """Test that near-duplicate drawings are handled correctly with multiple steps.

        When there are two step numbers and two near-duplicate drawings above them,
        only one drawing should be selected as the parts list, and both steps should
        be created successfully.
        """
        page_bbox = BBox(0, 0, 600, 400)

        # A part count inside the drawings
        pc = Text(
            id=0,
            bbox=BBox(320, 45, 330, 55),
            text="1x",
        )

        # Image inside the drawings, above the part count
        img = Image(id=6, bbox=BBox(320, 20, 330, 40))

        # Two steps below the drawings (both tall enough)
        step1 = Text(id=1, bbox=BBox(260, 70, 276, 96), text="5")
        step2 = Text(id=2, bbox=BBox(300, 70, 316, 96), text="6")

        # Real page number at bottom to avoid confusion
        page_number = Text(id=3, bbox=BBox(10, 380, 20, 390), text="1")

        # Two near-duplicate drawings above the steps
        d_small = Drawing(id=4, bbox=BBox(262.5, 14.7, 414.6, 61.9))
        d_large = Drawing(id=5, bbox=BBox(262.0, 14.2, 415.1, 62.4))

        page = PageData(
            page_number=2,
            blocks=[pc, img, step1, step2, page_number, d_small, d_large],
            bbox=page_bbox,
        )

        result = classify_elements(page, config=ClassifierConfig())

        # Check that both steps were created successfully
        assert result.page is not None
        assert len(result.page.steps) == 2

        # Verify both steps have the correct step numbers
        step_numbers = sorted([s.step_number.value for s in result.page.steps])
        assert step_numbers == [5, 6]

        # At least one step should have a parts_list with the part
        steps_with_parts = [
            s for s in result.page.steps if s.parts_list and len(s.parts_list.parts) > 0
        ]
        assert len(steps_with_parts) >= 1, "At least one step should have parts"
