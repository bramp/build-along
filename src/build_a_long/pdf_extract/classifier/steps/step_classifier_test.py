from collections.abc import Callable

import pytest

from build_a_long.pdf_extract.classifier import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.conftest import CandidateFactory
from build_a_long.pdf_extract.classifier.steps.step_classifier import (
    StepClassifier,
    _StepScore,
    filter_subassembly_values,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Step,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


@pytest.fixture
def classifier() -> StepClassifier:
    return StepClassifier(config=ClassifierConfig())


class TestStepClassification:
    """Tests for detecting complete Step structures."""

    def test_step_with_parts_list(
        self,
        classifier: StepClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test a step that has an associated parts list."""
        page_bbox = BBox(0, 0, 200, 300)
        step_text = Text(id=1, bbox=BBox(50, 180, 70, 210), text="10")
        d1 = Drawing(id=2, bbox=BBox(30, 100, 170, 160))
        pc1_text = Text(id=3, bbox=BBox(40, 110, 55, 120), text="2x")
        pc2_text = Text(id=4, bbox=BBox(100, 130, 115, 140), text="5x")
        img1 = Image(id=5, bbox=BBox(40, 90, 55, 105))
        img2 = Image(id=6, bbox=BBox(100, 115, 115, 125))

        page_data = PageData(
            page_number=6,
            blocks=[step_text, d1, pc1_text, pc2_text, img1, img2],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        # Register all relevant classifiers

        factory = candidate_factory(result)

        # Manually score dependencies
        factory.add_step_number(step_text)

        pc1_candidate = factory.add_part_count(pc1_text)
        pc2_candidate = factory.add_part_count(pc2_text)

        part1_candidate = factory.add_part(pc1_candidate, img1)
        part2_candidate = factory.add_part(pc2_candidate, img2)

        factory.add_parts_list(d1, [part1_candidate, part2_candidate])

        classifier.score(result)

        # Build steps using build_all (handles deduplication)
        classifier.build_all(result)

        # Get constructed steps
        steps = [
            c.constructed
            for c in result.get_candidates("step")
            if isinstance(c.constructed, Step)
        ]
        assert len(steps) == 1
        constructed_step = steps[0]

        assert constructed_step.step_number.value == 10
        assert constructed_step.parts_list is not None
        assert len(constructed_step.parts_list.parts) == 2
        # Diagram is None when DiagramClassifier doesn't find any diagrams
        assert constructed_step.diagram is None

    def test_step_without_parts_list(
        self,
        classifier: StepClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test a step that has no associated parts list."""
        page_bbox = BBox(0, 0, 200, 300)
        step_text = Text(id=1, bbox=BBox(50, 180, 70, 210), text="5")

        page_data = PageData(
            page_number=6,
            blocks=[step_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        # Register all relevant classifiers

        factory = candidate_factory(result)

        # Manually score step number candidate
        factory.add_step_number(step_text)

        classifier.score(result)

        # Build steps using build_all (handles deduplication)
        classifier.build_all(result)

        # Get constructed steps
        steps = [
            c.constructed
            for c in result.get_candidates("step")
            if isinstance(c.constructed, Step)
        ]
        assert len(steps) == 1
        constructed_step = steps[0]

        assert constructed_step.step_number.value == 5
        assert constructed_step.parts_list is None  # No parts list candidate
        # Diagram is None when DiagramClassifier doesn't find any diagrams
        assert constructed_step.diagram is None

    def test_multiple_steps_on_page(
        self,
        classifier: StepClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test a page with multiple steps."""
        page_bbox = BBox(0, 0, 400, 300)

        # First step components
        step1_text = Text(id=1, bbox=BBox(50, 180, 70, 210), text="1")
        d1 = Drawing(id=2, bbox=BBox(30, 100, 170, 160))
        pc1_text = Text(id=3, bbox=BBox(40, 110, 55, 120), text="2x")
        img1 = Image(id=7, bbox=BBox(40, 90, 55, 105))

        # Second step components
        step2_text = Text(id=4, bbox=BBox(250, 180, 270, 210), text="2")
        d2 = Drawing(id=5, bbox=BBox(230, 100, 370, 160))
        pc2_text = Text(id=6, bbox=BBox(240, 110, 255, 120), text="3x")
        img2 = Image(id=8, bbox=BBox(240, 90, 255, 105))

        page_data = PageData(
            page_number=6,
            blocks=[step1_text, d1, pc1_text, img1, step2_text, d2, pc2_text, img2],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        # Register all relevant classifiers

        factory = candidate_factory(result)

        # Score dependencies
        factory.add_step_number(step1_text)
        factory.add_step_number(step2_text)

        pc1_candidate = factory.add_part_count(pc1_text)
        pc2_candidate = factory.add_part_count(pc2_text)

        part1_candidate = factory.add_part(pc1_candidate, img1)
        part2_candidate = factory.add_part(pc2_candidate, img2)

        factory.add_parts_list(d1, [part1_candidate])
        factory.add_parts_list(d2, [part2_candidate])

        classifier.score(result)

        # Build steps using build_all (handles deduplication)
        classifier.build_all(result)

        # Get constructed steps
        steps = [
            c.constructed
            for c in result.get_candidates("step")
            if isinstance(c.constructed, Step)
        ]
        assert len(steps) == 2

        # Check that steps are in order by value
        steps_sorted = sorted(steps, key=lambda s: s.step_number.value)
        assert steps_sorted[0].step_number.value == 1
        assert steps_sorted[1].step_number.value == 2

    def test_step_score_ordering(
        self,
        classifier: StepClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that steps are ordered correctly by their score."""
        page_bbox = BBox(0, 0, 400, 300)

        # Step 2 appears first in the element list
        step2_text = Text(id=1, bbox=BBox(250, 180, 270, 210), text="2")

        # Step 1 appears second
        step1_text = Text(id=2, bbox=BBox(50, 180, 70, 210), text="1")

        page_data = PageData(
            page_number=6,
            blocks=[step2_text, step1_text],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        # Register all relevant classifiers

        factory = candidate_factory(result)

        # Score dependencies - assign higher score to step1
        # to ensure it wins if scores are tie-broken
        factory.add_step_number(step2_text, score=0.8)  # Lower score for step2
        factory.add_step_number(step1_text, score=1.0)  # Higher score for step1

        classifier.score(result)

        # Get the candidates in sorted order
        step_candidates = result.get_candidates("step")
        sorted_candidates = sorted(
            step_candidates,
            key=lambda c: (
                c.score_details.sort_key()
                if isinstance(c.score_details, _StepScore)
                else (0.0, 0)
            ),
        )

        # Construct both steps
        assert len(sorted_candidates) >= 2
        constructed_step1 = result.build(sorted_candidates[0])
        constructed_step2 = result.build(sorted_candidates[1])

        assert isinstance(constructed_step1, Step)
        assert isinstance(constructed_step2, Step)
        assert constructed_step1.step_number.value == 1
        assert constructed_step2.step_number.value == 2

    def test_duplicate_step_numbers_only_match_one_step(
        self,
        classifier: StepClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """When there are duplicate step numbers (same value), only one Step
        should be created. The StepClassifier should prefer the best-scoring
        StepNumber and skip subsequent ones with the same value.

        This test verifies that the uniqueness constraint is enforced at the
        Step level, not the PartsList level.
        """
        page_bbox = BBox(0, 0, 600, 400)

        # Two step numbers with the SAME value (both are "1")
        step1_text = Text(id=1, bbox=BBox(50, 150, 70, 180), text="1")
        step2_text = Text(
            id=2, bbox=BBox(50, 300, 70, 330), text="1"
        )  # Duplicate value

        # Two drawings, each above one of the step numbers
        d1 = Drawing(id=3, bbox=BBox(30, 80, 170, 140))  # Above step1
        d2 = Drawing(id=4, bbox=BBox(30, 230, 170, 290))  # Above step2

        # Part counts and images inside d1
        pc1_text = Text(id=5, bbox=BBox(40, 100, 55, 110), text="2x")
        img1 = Image(id=6, bbox=BBox(40, 85, 55, 95))

        # Part counts and images inside d2
        pc2_text = Text(id=7, bbox=BBox(40, 250, 55, 260), text="3x")
        img2 = Image(id=8, bbox=BBox(40, 235, 55, 245))

        page_data = PageData(
            page_number=1,
            blocks=[step1_text, step2_text, d1, d2, pc1_text, img1, pc2_text, img2],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)
        # Register all relevant classifiers

        factory = candidate_factory(result)

        # Score dependencies
        factory.add_step_number(step1_text, score=1.0)  # Best scoring step number
        factory.add_step_number(step2_text, score=0.9)  # Lower scoring duplicate

        pc1_candidate = factory.add_part_count(pc1_text)
        pc2_candidate = factory.add_part_count(pc2_text)

        part1_candidate = factory.add_part(pc1_candidate, img1)
        part2_candidate = factory.add_part(pc2_candidate, img2)

        factory.add_parts_list(d1, [part1_candidate])
        factory.add_parts_list(d2, [part2_candidate])

        classifier.score(result)

        # Build steps using build_all (handles deduplication)
        classifier.build_all(result)

        # Get constructed steps
        steps = [
            c.constructed
            for c in result.get_candidates("step")
            if isinstance(c.constructed, Step)
        ]

        # Only ONE step should be created (uniqueness enforced at Step level)
        assert len(steps) == 1
        assert steps[0].step_number.value == 1


class TestFilterSubassemblyValues:
    """Tests for filter_subassembly_values function.

    This function filters out items with values likely to be subassembly steps
    (e.g., 1, 2) when the list also contains higher-numbered page-level values
    (e.g., 15, 16).
    """

    def test_empty_list_returns_empty(self) -> None:
        """An empty list should return empty."""
        assert filter_subassembly_values([]) == []

    def test_single_item_returns_unchanged(self) -> None:
        """A single item should be returned unchanged."""
        items = [(5, "a")]
        assert filter_subassembly_values(items) == items

    def test_consecutive_values_no_gap_returns_unchanged(self) -> None:
        """Values with no significant gap (e.g., 15, 16, 17) return unchanged."""
        items = [(15, "a"), (16, "b"), (17, "c")]
        assert filter_subassembly_values(items) == items

    def test_gap_exactly_3_does_not_filter(self) -> None:
        """A gap of exactly 3 (not > 3) should not filter."""
        # Gap of 3: 1 -> 4 (4 - 1 = 3, not > 3)
        items = [(1, "a"), (4, "b")]
        assert filter_subassembly_values(items) == items

    def test_gap_exactly_4_filters(self) -> None:
        """A gap of exactly 4 (> 3) should filter when min_value <= 3."""
        # Gap of 4: 1 -> 5 (5 - 1 = 4, which is > 3)
        items = [(1, "a"), (5, "b")]
        assert filter_subassembly_values(items) == [(5, "b")]

    def test_gap_but_min_value_greater_than_3_no_filter(self) -> None:
        """Gap > 3 but min_value > 3 should not filter (e.g., 5, 6, 15, 16)."""
        # Gap of 9 (15 - 6 = 9) but min is 5 (> 3)
        items = [(5, "a"), (6, "b"), (15, "c"), (16, "d")]
        assert filter_subassembly_values(items) == items

    def test_min_value_exactly_3_filters(self) -> None:
        """Gap > 3 and min_value == 3 should filter (3 <= 3)."""
        # Gap of 12 (15 - 3 = 12) and min is 3 (<= 3)
        items = [(3, "a"), (15, "b"), (16, "c")]
        result = filter_subassembly_values(items)
        assert result == [(15, "b"), (16, "c")]

    def test_min_value_exactly_4_no_filter(self) -> None:
        """Gap > 3 but min_value == 4 should NOT filter (4 > 3)."""
        # Gap of 11 (15 - 4 = 11) but min is 4 (> 3)
        items = [(4, "a"), (15, "b"), (16, "c")]
        assert filter_subassembly_values(items) == items

    def test_typical_subassembly_case(self) -> None:
        """Typical case: steps 1, 2 (subassembly) + 15, 16 (page-level)."""
        items = [(1, "a"), (2, "b"), (15, "c"), (16, "d")]
        result = filter_subassembly_values(items)
        assert result == [(15, "c"), (16, "d")]

    def test_multiple_gaps_uses_largest(self) -> None:
        """When there are multiple gaps, the largest one determines filtering."""
        # Values: 1, 2, 5, 20, 21
        # Gaps: 2->5 = 3, 5->20 = 15 (largest)
        items = [(1, "a"), (2, "b"), (5, "c"), (20, "d"), (21, "e")]
        result = filter_subassembly_values(items)
        # Largest gap is 5->20, threshold = 20
        assert result == [(20, "d"), (21, "e")]

    def test_unordered_input_handled_correctly(self) -> None:
        """Items passed in non-sorted order should be handled correctly."""
        # Pass in non-sorted order
        items = [(16, "d"), (1, "a"), (15, "c"), (2, "b")]
        result = filter_subassembly_values(items)
        # Should filter and return sorted
        assert result == [(15, "c"), (16, "d")]

    def test_preserves_associated_data(self) -> None:
        """The associated data (second element of tuple) should be preserved."""
        items = [
            (1, {"name": "step1"}),
            (2, {"name": "step2"}),
            (15, {"name": "step15"}),
        ]
        result = filter_subassembly_values(items)
        assert result == [(15, {"name": "step15"})]

    def test_custom_min_gap_parameter(self) -> None:
        """Custom min_gap parameter should be respected."""
        items = [(1, "a"), (3, "b")]  # Gap of 2
        # Default min_gap=3, so gap of 2 doesn't filter
        assert filter_subassembly_values(items) == items
        # With min_gap=1, gap of 2 > 1 should filter
        assert filter_subassembly_values(items, min_gap=1) == [(3, "b")]

    def test_custom_max_subassembly_start_parameter(self) -> None:
        """Custom max_subassembly_start parameter should be respected."""
        items = [(4, "a"), (15, "b")]  # Gap of 11, min=4
        # Default max_subassembly_start=3, so min=4 doesn't filter
        assert filter_subassembly_values(items) == items
        # With max_subassembly_start=4, min=4 should filter
        assert filter_subassembly_values(items, max_subassembly_start=4) == [(15, "b")]
