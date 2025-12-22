from collections.abc import Callable

import pytest

from build_a_long.pdf_extract.classifier import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.conftest import CandidateFactory
from build_a_long.pdf_extract.classifier.steps.arrow_classifier import (
    ArrowClassifier,
    _ArrowHeadData,
    _ArrowScore,
)
from build_a_long.pdf_extract.classifier.steps.step_classifier import StepClassifier
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Text
from build_a_long.pdf_extract.validation.rules import (
    assert_constructed_elements_on_page,
)


@pytest.fixture
def classifier() -> StepClassifier:
    return StepClassifier(config=ClassifierConfig())


class TestOrphanedElements:
    """Tests for orphaned elements issue."""

    def _add_arrow_candidate(
        self, result: ClassificationResult, drawing: Drawing
    ) -> None:
        """Helper to manually add an arrow candidate."""
        # Register classifier
        result._register_classifier("arrow", ArrowClassifier(config=ClassifierConfig()))

        # Create minimal valid score details
        head_data = _ArrowHeadData(
            tip=(drawing.bbox.x1, drawing.bbox.y1),
            direction=0,
            shape_score=1.0,
            size_score=1.0,
            block=drawing,
        )
        score_details = _ArrowScore(heads=[head_data])

        candidate = Candidate(
            bbox=drawing.bbox,
            label="arrow",
            score=1.0,
            score_details=score_details,
            source_blocks=[drawing],
        )
        result.add_candidate(candidate)

    @pytest.mark.xfail(reason="Orphan cleanup not yet implemented - see Phase 11 TODO")
    def test_orphaned_elements_cleanup(
        self,
        classifier: StepClassifier,
        candidate_factory: Callable[[ClassificationResult], CandidateFactory],
    ) -> None:
        """Test that arrows are cleaned up when their step fails to build.

        Scenario:
        1. An arrow is present.
        2. A step is present (Step 1).
        3. The step's source blocks are ALREADY consumed by something else.

        Execution:
        - Phase 3: Arrow is built (speculatively).
        - Phase 5: Step attempts to build but fails due to consumed blocks.

        Expected Result:
        - The arrow should be rolled back/cleaned up because it wasn't
          assigned to any step.
        - assert_constructed_elements_on_page should pass.
        """
        page_bbox = BBox(0, 0, 400, 400)

        # Step number text
        step_text = Text(id=1, bbox=BBox(50, 50, 70, 70), text="1")

        # Arrow drawing (near the step)
        arrow_drawing = Drawing(id=2, bbox=BBox(100, 100, 200, 110))

        page_data = PageData(
            page_number=1,
            blocks=[step_text, arrow_drawing],
            bbox=page_bbox,
        )

        result = ClassificationResult(page_data=page_data)

        # Setup candidates
        factory = candidate_factory(result)

        # Add arrow candidate manually
        self._add_arrow_candidate(result, arrow_drawing)

        # Add step number candidate
        factory.add_step_number(step_text)

        classifier.score(result)

        # Simulate that step_text blocks are already consumed by another element
        # (e.g. a bag number or unrelated text)
        result._consumed_blocks.add(step_text.id)

        # Run build_all
        classifier.build_all(result)

        # Verify that the Step failed to build (due to conflict)
        steps = result.get_built_candidates("step")
        assert len(steps) == 0, "Step should have failed to build"

        # Verify that the Arrow was cleaned up (Phase 11)
        arrows = result.get_built_candidates("arrow")
        assert len(arrows) == 0, "Arrow should have been cleaned up"

        # This assertion should PASS now
        assert_constructed_elements_on_page(result)
