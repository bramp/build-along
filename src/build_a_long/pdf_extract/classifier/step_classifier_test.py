from collections.abc import Callable

import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.pdf_extract.classifier.part_number_classifier import (
    PartNumberClassifier,
)
from build_a_long.pdf_extract.classifier.parts_classifier import (
    PartsClassifier,
)
from build_a_long.pdf_extract.classifier.parts_list_classifier import (
    PartsListClassifier,
)
from build_a_long.pdf_extract.classifier.piece_length_classifier import (
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.classifier.step_classifier import (
    StepClassifier,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Step,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text

from .conftest import CandidateFactory


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
        config = classifier.config
        # Register all relevant classifiers
        result.register_classifier("step_number", StepNumberClassifier(config))
        result.register_classifier("parts_list", PartsListClassifier(config))
        result.register_classifier("part", PartsClassifier(config))
        result.register_classifier("part_count", PartCountClassifier(config))
        result.register_classifier("part_number", PartNumberClassifier(config))
        result.register_classifier("piece_length", PieceLengthClassifier(config))
        result.register_classifier("step", classifier)

        factory = candidate_factory(result)

        # Manually score dependencies
        factory.add_step_number(step_text)

        pc1_candidate = factory.add_part_count(pc1_text)
        pc2_candidate = factory.add_part_count(pc2_text)

        part1_candidate = factory.add_part(pc1_candidate, img1)
        part2_candidate = factory.add_part(pc2_candidate, img2)

        factory.add_parts_list(d1, [part1_candidate, part2_candidate])

        classifier.score(result)

        # Construct the Step
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 1
        constructed_step = result.construct_candidate(step_candidates[0])
        assert isinstance(constructed_step, Step)

        assert constructed_step.step_number.value == 10
        assert len(constructed_step.parts_list.parts) == 2
        assert constructed_step.diagram is not None

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
        config = classifier.config
        # Register all relevant classifiers
        result.register_classifier("step_number", StepNumberClassifier(config))
        result.register_classifier("parts_list", PartsListClassifier(config))
        result.register_classifier("part", PartsClassifier(config))
        result.register_classifier("part_count", PartCountClassifier(config))
        result.register_classifier("part_number", PartNumberClassifier(config))
        result.register_classifier("piece_length", PieceLengthClassifier(config))
        result.register_classifier("step", classifier)

        factory = candidate_factory(result)

        # Manually score step number candidate
        factory.add_step_number(step_text)

        classifier.score(result)

        # Construct the Step
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 1
        constructed_step = result.construct_candidate(step_candidates[0])
        assert isinstance(constructed_step, Step)

        assert constructed_step.step_number.value == 5
        assert len(constructed_step.parts_list.parts) == 0  # Should have no parts
        assert constructed_step.diagram is not None

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
        config = classifier.config
        # Register all relevant classifiers
        result.register_classifier("step_number", StepNumberClassifier(config))
        result.register_classifier("parts_list", PartsListClassifier(config))
        result.register_classifier("part", PartsClassifier(config))
        result.register_classifier("part_count", PartCountClassifier(config))
        result.register_classifier("part_number", PartNumberClassifier(config))
        result.register_classifier("piece_length", PieceLengthClassifier(config))
        result.register_classifier("step", classifier)

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

        # Construct the Steps
        steps: list[Step] = []
        for step_candidate in result.get_candidates("step"):
            constructed_step = result.construct_candidate(step_candidate)
            assert isinstance(constructed_step, Step)
            steps.append(constructed_step)

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
        config = classifier.config
        # Register all relevant classifiers
        result.register_classifier("step_number", StepNumberClassifier(config))
        result.register_classifier("parts_list", PartsListClassifier(config))
        result.register_classifier("part", PartsClassifier(config))
        result.register_classifier("part_count", PartCountClassifier(config))
        result.register_classifier("part_number", PartNumberClassifier(config))
        result.register_classifier("piece_length", PieceLengthClassifier(config))
        result.register_classifier("step", classifier)

        factory = candidate_factory(result)

        # Score dependencies - assign higher score to step1 to ensure it wins if scores are tie-broken
        factory.add_step_number(step2_text, score=0.8)  # Lower score for step2
        factory.add_step_number(step1_text, score=1.0)  # Higher score for step1

        classifier.score(result)

        # Get the candidates in sorted order
        step_candidates = result.get_candidates("step")
        sorted_candidates = sorted(
            step_candidates,
            key=lambda c: c.score_details.sort_key(),
        )

        # Construct both steps
        assert len(sorted_candidates) >= 2
        constructed_step1 = result.construct_candidate(sorted_candidates[0])
        constructed_step2 = result.construct_candidate(sorted_candidates[1])

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
        config = classifier.config
        # Register all relevant classifiers
        result.register_classifier("step_number", StepNumberClassifier(config))
        result.register_classifier("parts_list", PartsListClassifier(config))
        result.register_classifier("part", PartsClassifier(config))
        result.register_classifier("part_count", PartCountClassifier(config))
        result.register_classifier("part_number", PartNumberClassifier(config))
        result.register_classifier("piece_length", PieceLengthClassifier(config))
        result.register_classifier("step", classifier)

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

        # Construct the Steps
        steps: list[Step] = []
        for step_candidate in result.get_candidates("step"):
            constructed_step = result.construct_candidate(step_candidate)
            assert isinstance(constructed_step, Step)
            steps.append(constructed_step)

        # Only ONE step should be created (uniqueness enforced at Step level)
        assert len(steps) == 1
        assert steps[0].step_number.value == 1
