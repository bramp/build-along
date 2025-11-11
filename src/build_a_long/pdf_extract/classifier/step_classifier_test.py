"""Tests for the step classifier."""

from build_a_long.pdf_extract.classifier.classifier import classify_elements
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import Step
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


class TestStepClassification:
    """Tests for detecting complete Step structures."""

    def test_step_with_parts_list(self) -> None:
        """Test a step that has an associated parts list."""
        page_bbox = BBox(0, 0, 200, 300)

        # Page number
        pn = Text(id=0, bbox=BBox(10, 285, 20, 295), text="6")

        # Step number (taller than page number)
        step = Text(id=1, bbox=BBox(50, 180, 70, 210), text="10")

        # Parts list drawing above the step
        d1 = Drawing(id=2, bbox=BBox(30, 100, 170, 160))

        # Part counts inside d1
        pc1 = Text(id=3, bbox=BBox(40, 110, 55, 120), text="2x")
        pc2 = Text(id=4, bbox=BBox(100, 130, 115, 140), text="5×")

        # Images above the part counts (required for PartsClassifier)
        img1 = Image(id=5, bbox=BBox(40, 90, 55, 105))
        img2 = Image(id=6, bbox=BBox(100, 115, 115, 125))

        page = PageData(
            page_number=6,
            blocks=[pn, step, d1, pc1, pc2, img1, img2],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Check that step_number is still labeled as step_number
        assert result.get_label(step) == "step_number"

        # Get the constructed Step element using the new get_winners method
        winning_steps = result.get_winners("step", Step)
        assert len(winning_steps) == 1

        # Verify the Step has the correct components
        constructed = winning_steps[0]
        assert constructed.step_number.value == 10
        assert len(constructed.parts_list.parts) > 0  # Should have parts
        assert constructed.diagram is not None

    def test_step_without_parts_list(self) -> None:
        """Test a step that has no associated parts list."""
        page_bbox = BBox(0, 0, 200, 300)

        # Page number
        pn = Text(id=0, bbox=BBox(10, 285, 20, 295), text="6")

        # Step number (taller than page number)
        step = Text(id=1, bbox=BBox(50, 180, 70, 210), text="5")

        page = PageData(
            page_number=6,
            blocks=[pn, step],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Check that step_number is still labeled as step_number
        assert result.get_label(step) == "step_number"

        # Get the constructed Step element using the new get_winners method
        steps = result.get_winners("step", Step)
        assert len(steps) == 1

        # Verify the Step has the correct components
        constructed = steps[0]
        assert constructed.step_number.value == 5
        assert len(constructed.parts_list.parts) == 0  # Should have no parts
        assert constructed.diagram is not None

    def test_multiple_steps_on_page(self) -> None:
        """Test a page with multiple steps."""
        page_bbox = BBox(0, 0, 400, 300)

        # Page number
        pn = Text(id=0, bbox=BBox(10, 285, 20, 295), text="6")

        # First step
        step1 = Text(id=1, bbox=BBox(50, 180, 70, 210), text="1")
        d1 = Drawing(id=2, bbox=BBox(30, 100, 170, 160))
        pc1 = Text(id=3, bbox=BBox(40, 110, 55, 120), text="2x")

        # Second step
        step2 = Text(id=4, bbox=BBox(250, 180, 270, 210), text="2")
        d2 = Drawing(id=5, bbox=BBox(230, 100, 370, 160))
        pc2 = Text(id=6, bbox=BBox(240, 110, 255, 120), text="3x")

        page = PageData(
            page_number=6,
            blocks=[pn, step1, d1, pc1, step2, d2, pc2],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Check that both step_numbers are still labeled as step_number
        assert result.get_label(step1) == "step_number"
        assert result.get_label(step2) == "step_number"

        # Get the constructed Step elements using the new get_winners method
        steps = result.get_winners("step", Step)
        assert len(steps) == 2

        # Check that steps are in order
        steps_sorted = sorted(steps, key=lambda s: s.step_number.value)
        assert steps_sorted[0].step_number.value == 1
        assert steps_sorted[1].step_number.value == 2

    def test_step_score_ordering(self) -> None:
        """Test that steps are ordered correctly by their score."""
        page_bbox = BBox(0, 0, 400, 300)

        # Page number
        pn = Text(id=0, bbox=BBox(10, 285, 20, 295), text="6")

        # Step 2 appears first in the element list
        step2 = Text(id=1, bbox=BBox(250, 180, 270, 210), text="2")

        # Step 1 appears second
        step1 = Text(id=2, bbox=BBox(50, 180, 70, 210), text="1")

        page = PageData(
            page_number=6,
            blocks=[pn, step2, step1],
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

    def test_duplicate_step_numbers_only_match_one_step(self) -> None:
        """When there are duplicate step numbers (same value), only one Step
        should be created. The StepClassifier should prefer the best-scoring
        StepNumber and skip subsequent ones with the same value.

        This test verifies that the uniqueness constraint is enforced at the
        Step level, not the PartsList level.
        """
        page_bbox = BBox(0, 0, 600, 400)

        # Page number
        page_number = Text(id=0, bbox=BBox(10, 380, 20, 390), text="1")

        # Two step numbers with the SAME value (both are "1")
        step1 = Text(id=1, bbox=BBox(50, 150, 70, 180), text="1")
        step2 = Text(id=2, bbox=BBox(50, 300, 70, 330), text="1")  # Duplicate value

        # Two drawings, each above one of the step numbers
        d1 = Drawing(id=3, bbox=BBox(30, 80, 170, 140))  # Above step1
        d2 = Drawing(id=4, bbox=BBox(30, 230, 170, 290))  # Above step2

        # Part counts inside d1
        pc1 = Text(id=5, bbox=BBox(40, 100, 55, 110), text="2x")
        img1 = Image(id=6, bbox=BBox(40, 85, 55, 95))

        # Part counts inside d2
        pc2 = Text(id=7, bbox=BBox(40, 250, 55, 260), text="3x")
        img2 = Image(id=8, bbox=BBox(40, 235, 55, 245))

        page = PageData(
            page_number=1,
            blocks=[page_number, step1, step2, d1, d2, pc1, img1, pc2, img2],
            bbox=page_bbox,
        )

        result = classify_elements(page)

        # Both step numbers should be labeled
        assert result.get_label(step1) == "step_number"
        assert result.get_label(step2) == "step_number"

        # Both parts lists should be marked as winners
        # (no uniqueness at PartsList level)
        parts_list_candidates = result.get_candidates("parts_list")
        winning_parts_lists = [c for c in parts_list_candidates if c.is_winner]
        assert len(winning_parts_lists) == 2, (
            f"Expected 2 parts list winners, got {len(winning_parts_lists)}"
        )

        # But only ONE step should be created (uniqueness enforced at Step level)
        winning_steps = result.get_winners("step", Step)

        assert len(winning_steps) == 1, (
            f"Expected exactly 1 step winner, got {len(winning_steps)}. "
            "Each step number value should only create one Step."
        )

        # Verify the winning step has value 1
        winning_step = winning_steps[0]
        assert winning_step.step_number.value == 1
