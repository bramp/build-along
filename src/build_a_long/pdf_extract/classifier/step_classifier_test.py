"""Tests for the step classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import Step
from build_a_long.pdf_extract.extractor.page_elements import Drawing, Text


class TestStepClassification:
    """Tests for detecting complete Step structures."""

    def test_step_with_parts_list(self) -> None:
        """Test a step that has an associated parts list."""
        page_bbox = BBox(0, 0, 200, 300)

        # Page number
        pn = Text(bbox=BBox(10, 285, 20, 295), text="6")

        # Step number (taller than page number)
        step = Text(bbox=BBox(50, 180, 70, 210), text="10")

        # Parts list drawing above the step
        d1 = Drawing(bbox=BBox(30, 100, 170, 160))

        # Part counts inside d1
        pc1 = Text(bbox=BBox(40, 110, 55, 120), text="2x")
        pc2 = Text(bbox=BBox(100, 130, 115, 140), text="5×")

        page = PageData(
            page_number=6,
            elements=[pn, step, d1, pc1, pc2],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Check that step_number is still labeled as step_number
        assert result.get_label(step) == "step_number"

        # Get the constructed Step element from candidates
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 1

        # The step should be constructed as a Step object
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 1
        constructed = step_candidates[0].constructed
        assert isinstance(constructed, Step)

        # Verify the Step has the correct components
        assert constructed.step_number.value == 10
        assert len(constructed.parts_list.parts) > 0  # Should have parts
        assert constructed.diagram is not None

    def test_step_without_parts_list(self) -> None:
        """Test a step that has no associated parts list."""
        page_bbox = BBox(0, 0, 200, 300)

        # Page number
        pn = Text(bbox=BBox(10, 285, 20, 295), text="6")

        # Step number (taller than page number)
        step = Text(bbox=BBox(50, 180, 70, 210), text="5")

        page = PageData(
            page_number=6,
            elements=[pn, step],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Check that step_number is still labeled as step_number
        assert result.get_label(step) == "step_number"

        # Get the constructed Step element
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 1

        # Get the constructed Step element
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 1
        constructed = step_candidates[0].constructed
        assert isinstance(constructed, Step)

        # Verify the Step has the correct components
        assert constructed.step_number.value == 5
        assert len(constructed.parts_list.parts) == 0  # Should have no parts
        assert constructed.diagram is not None

    def test_multiple_steps_on_page(self) -> None:
        """Test a page with multiple steps."""
        page_bbox = BBox(0, 0, 400, 300)

        # Page number
        pn = Text(bbox=BBox(10, 285, 20, 295), text="6")

        # First step
        step1 = Text(bbox=BBox(50, 180, 70, 210), text="1")
        d1 = Drawing(bbox=BBox(30, 100, 170, 160))
        pc1 = Text(bbox=BBox(40, 110, 55, 120), text="2x")

        # Second step
        step2 = Text(bbox=BBox(250, 180, 270, 210), text="2")
        d2 = Drawing(bbox=BBox(230, 100, 370, 160))
        pc2 = Text(bbox=BBox(240, 110, 255, 120), text="3x")

        page = PageData(
            page_number=6,
            elements=[pn, step1, d1, pc1, step2, d2, pc2],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Check that both step_numbers are still labeled as step_number
        assert result.get_label(step1) == "step_number"
        assert result.get_label(step2) == "step_number"

        # Check that both steps are classified as "step" candidates
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 2

        # Verify both steps are constructed
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 2

        # Check that steps are in order
        steps = [
            c.constructed
            for c in step_candidates
            if c.constructed and isinstance(c.constructed, Step)
        ]
        assert len(steps) == 2
        steps_sorted = sorted(steps, key=lambda s: s.step_number.value)
        assert steps_sorted[0].step_number.value == 1
        assert steps_sorted[1].step_number.value == 2

    def test_step_score_ordering(self) -> None:
        """Test that steps are ordered correctly by their score."""
        page_bbox = BBox(0, 0, 400, 300)

        # Page number
        pn = Text(bbox=BBox(10, 285, 20, 295), text="6")

        # Step 2 appears first in the element list
        step2 = Text(bbox=BBox(250, 180, 270, 210), text="2")

        # Step 1 appears second
        step1 = Text(bbox=BBox(50, 180, 70, 210), text="1")

        page = PageData(
            page_number=6,
            elements=[pn, step2, step1],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Get the candidates in sorted order
        step_candidates = result.get_candidates("step")
        sorted_candidates = sorted(
            step_candidates,
            key=lambda c: c.score_details.sort_key(),
        )

        # Step 1 should come before Step 2 due to lower step number value
        assert len(sorted_candidates) >= 2
        step1 = sorted_candidates[0].constructed
        step2 = sorted_candidates[1].constructed
        assert isinstance(step1, Step)
        assert isinstance(step2, Step)
        assert step1.step_number.value == 1
        assert step2.step_number.value == 2
