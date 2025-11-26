"""Tests for the step classifier."""

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.part_count_classifier import (
    PartCountClassifier,
    _PartCountScore,
)
from build_a_long.pdf_extract.classifier.part_number_classifier import (
    PartNumberClassifier,
)
from build_a_long.pdf_extract.classifier.parts_classifier import (
    PartsClassifier,
    _PartPairScore,
)
from build_a_long.pdf_extract.classifier.parts_list_classifier import (
    PartsListClassifier,
    _PartsListScore,
)
from build_a_long.pdf_extract.classifier.piece_length_classifier import (
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.classifier.step_classifier import (
    StepClassifier,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import (
    StepNumberClassifier,
    _StepNumberScore,
)
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    Step,
)
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


class TestStepClassification:
    """Tests for detecting complete Step structures."""

    def _setup_step_classifier_test(
        self, page_data: PageData
    ) -> tuple[StepClassifier, ClassificationResult]:
        config = ClassifierConfig()
        step_classifier = StepClassifier(config)
        result = ClassificationResult(page_data=page_data)

        # Register all relevant classifiers
        result.register_classifier("step_number", StepNumberClassifier(config))
        result.register_classifier("parts_list", PartsListClassifier(config))
        result.register_classifier("part", PartsClassifier(config))
        result.register_classifier("part_count", PartCountClassifier(config))
        result.register_classifier("part_number", PartNumberClassifier(config))
        result.register_classifier("piece_length", PieceLengthClassifier(config))
        result.register_classifier("step", step_classifier)

        return step_classifier, result

    def _create_and_score_step_number_candidate(
        self, result: ClassificationResult, text_block: Text, score_value: float = 1.0
    ) -> Candidate:
        score_details = _StepNumberScore(text_score=score_value, font_size_score=0.5)
        candidate = Candidate(
            bbox=text_block.bbox,
            label="step_number",
            score=score_value,
            score_details=score_details,
            constructed=None,
            source_blocks=[text_block],
        )
        result.add_candidate("step_number", candidate)
        return candidate

    def _create_and_score_part_count_candidate(
        self, result: ClassificationResult, text_block: Text, score_value: float = 1.0
    ) -> Candidate:
        score_details = _PartCountScore(
            text_score=score_value,
            font_size_score=0.5,
            matched_hint="catalog_part_count",
        )
        candidate = Candidate(
            bbox=text_block.bbox,
            label="part_count",
            score=score_value,
            score_details=score_details,
            constructed=None,
            source_blocks=[text_block],
        )
        result.add_candidate("part_count", candidate)
        return candidate

    def _create_and_score_part_candidate(
        self,
        result: ClassificationResult,
        part_count_candidate: Candidate,
        image_block: Image,
        score_value: float = 1.0,
    ) -> Candidate:
        score_details = _PartPairScore(
            distance=10.0,  # dummy
            part_count_candidate=part_count_candidate,
            image=image_block,
            part_number_candidate=None,
            piece_length_candidate=None,
        )
        candidate = Candidate(
            bbox=part_count_candidate.bbox.union(image_block.bbox),
            label="part",
            score=score_value,
            score_details=score_details,
            constructed=None,
            source_blocks=[image_block],  # Source block for Part is typically the image
        )
        result.add_candidate("part", candidate)
        return candidate

    def _create_and_score_parts_list_candidate(
        self,
        result: ClassificationResult,
        drawing_block: Drawing,
        part_candidates: list[Candidate],
        score_value: float = 1.0,
    ) -> Candidate:
        score_details = _PartsListScore(part_candidates=part_candidates)
        candidate = Candidate(
            bbox=drawing_block.bbox,
            label="parts_list",
            score=score_value,
            score_details=score_details,
            constructed=None,
            source_blocks=[drawing_block],
        )
        result.add_candidate("parts_list", candidate)
        return candidate

    def test_step_with_parts_list(self) -> None:
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

        step_classifier, result = self._setup_step_classifier_test(page_data)

        # Manually score dependencies
        sn_candidate = self._create_and_score_step_number_candidate(result, step_text)

        pc1_candidate = self._create_and_score_part_count_candidate(result, pc1_text)
        pc2_candidate = self._create_and_score_part_count_candidate(result, pc2_text)

        part1_candidate = self._create_and_score_part_candidate(
            result, pc1_candidate, img1
        )
        part2_candidate = self._create_and_score_part_candidate(
            result, pc2_candidate, img2
        )

        pl_candidate = self._create_and_score_parts_list_candidate(
            result, d1, [part1_candidate, part2_candidate]
        )

        step_classifier.score(result)

        # Construct the Step
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 1
        constructed_step = result.construct_candidate(step_candidates[0])
        assert isinstance(constructed_step, Step)

        assert constructed_step.step_number.value == 10
        assert len(constructed_step.parts_list.parts) == 2
        assert constructed_step.diagram is not None

    def test_step_without_parts_list(self) -> None:
        """Test a step that has no associated parts list."""
        page_bbox = BBox(0, 0, 200, 300)
        step_text = Text(id=1, bbox=BBox(50, 180, 70, 210), text="5")

        page_data = PageData(
            page_number=6,
            blocks=[step_text],
            bbox=page_bbox,
        )

        step_classifier, result = self._setup_step_classifier_test(page_data)

        # Manually score step number candidate
        self._create_and_score_step_number_candidate(result, step_text)

        step_classifier.score(result)

        # Construct the Step
        step_candidates = result.get_candidates("step")
        assert len(step_candidates) == 1
        constructed_step = result.construct_candidate(step_candidates[0])
        assert isinstance(constructed_step, Step)

        assert constructed_step.step_number.value == 5
        assert len(constructed_step.parts_list.parts) == 0  # Should have no parts
        assert constructed_step.diagram is not None

    def test_multiple_steps_on_page(self) -> None:
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

        step_classifier, result = self._setup_step_classifier_test(page_data)

        # Score dependencies
        sn1_candidate = self._create_and_score_step_number_candidate(result, step1_text)
        sn2_candidate = self._create_and_score_step_number_candidate(result, step2_text)

        pc1_candidate = self._create_and_score_part_count_candidate(result, pc1_text)
        pc2_candidate = self._create_and_score_part_count_candidate(result, pc2_text)

        part1_candidate = self._create_and_score_part_candidate(
            result, pc1_candidate, img1
        )
        part2_candidate = self._create_and_score_part_candidate(
            result, pc2_candidate, img2
        )

        pl1_candidate = self._create_and_score_parts_list_candidate(
            result, d1, [part1_candidate]
        )
        pl2_candidate = self._create_and_score_parts_list_candidate(
            result, d2, [part2_candidate]
        )

        step_classifier.score(result)

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

    def test_step_score_ordering(self) -> None:
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

        step_classifier, result = self._setup_step_classifier_test(page_data)

        # Score dependencies - assign higher score to step1 to ensure it wins if scores are tie-broken
        self._create_and_score_step_number_candidate(
            result, step2_text, score_value=0.8
        )  # Lower score for step2
        self._create_and_score_step_number_candidate(
            result, step1_text, score_value=1.0
        )  # Higher score for step1

        step_classifier.score(result)

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

    def test_duplicate_step_numbers_only_match_one_step(self) -> None:
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

        step_classifier, result = self._setup_step_classifier_test(page_data)

        # Score dependencies
        sn1_candidate = self._create_and_score_step_number_candidate(
            result, step1_text, score_value=1.0
        )  # Best scoring step number
        sn2_candidate = self._create_and_score_step_number_candidate(
            result, step2_text, score_value=0.9
        )  # Lower scoring duplicate

        pc1_candidate = self._create_and_score_part_count_candidate(result, pc1_text)
        pc2_candidate = self._create_and_score_part_count_candidate(result, pc2_text)

        part1_candidate = self._create_and_score_part_candidate(
            result, pc1_candidate, img1
        )
        part2_candidate = self._create_and_score_part_candidate(
            result, pc2_candidate, img2
        )

        pl1_candidate = self._create_and_score_parts_list_candidate(
            result, d1, [part1_candidate]
        )
        pl2_candidate = self._create_and_score_parts_list_candidate(
            result, d2, [part2_candidate]
        )

        step_classifier.score(result)

        # Construct the Steps
        steps: list[Step] = []
        for step_candidate in result.get_candidates("step"):
            constructed_step = result.construct_candidate(step_candidate)
            assert isinstance(constructed_step, Step)
            steps.append(constructed_step)

        # Only ONE step should be created (uniqueness enforced at Step level)
        assert len(steps) == 1
        assert steps[0].step_number.value == 1
