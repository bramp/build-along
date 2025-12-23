import pytest

from build_a_long.pdf_extract.classifier import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.parts.part_count_classifier import (
    PartCountClassifier,
)
from build_a_long.pdf_extract.classifier.parts.part_number_classifier import (
    PartNumberClassifier,
)
from build_a_long.pdf_extract.classifier.parts.parts_classifier import (
    PartsClassifier,
    _PartPairScore,
)
from build_a_long.pdf_extract.classifier.parts.parts_image_classifier import (
    PartsImageClassifier,
)
from build_a_long.pdf_extract.classifier.parts.parts_list_classifier import (
    PartsListClassifier,
    _PartsListScore,
)
from build_a_long.pdf_extract.classifier.parts.piece_length_classifier import (
    PieceLengthClassifier,
)
from build_a_long.pdf_extract.classifier.rule_based_classifier import (
    RuleScore,
    StepNumberScore,
)
from build_a_long.pdf_extract.classifier.steps.step_number_classifier import (
    StepNumberClassifier,
)
from build_a_long.pdf_extract.classifier.text import extract_step_number_value
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


# TODO Re-evaluate the use of this, and if its needed
class CandidateFactory:
    """Helper to create and register candidates with proper score details.

    This factory manually registers the classifiers it creates candidates for,
    since it bypasses the normal score() flow.
    """

    def __init__(
        self, result: ClassificationResult, config: ClassifierConfig | None = None
    ):
        self.result = result
        self.config = config or ClassifierConfig()
        self._registered_classifiers: set[str] = set()

    def _ensure_classifier_registered(self, label: str) -> None:
        """Ensure the classifier for this label is registered."""
        if label in self._registered_classifiers:
            return

        # Map labels to their classifier classes
        classifier_map = {
            "part_count": PartCountClassifier,
            "step_number": StepNumberClassifier,
            "part": PartsClassifier,
            "part_image": PartsImageClassifier,
            "parts_list": PartsListClassifier,
            "piece_length": PieceLengthClassifier,
            "part_number": PartNumberClassifier,
        }

        if label in classifier_map:
            classifier = classifier_map[label](config=self.config)
            self.result._register_classifier(label, classifier)
            self._registered_classifiers.add(label)

    def add_part_count(self, block: Text, score: float = 1.0) -> Candidate:
        self._ensure_classifier_registered("part_count")
        score_details = RuleScore(
            components={
                "text_score": score,
                "font_size_score": 0.5,
            },
            total_score=score,
        )
        candidate = Candidate(
            bbox=block.bbox,
            label="part_count",
            score=score,
            score_details=score_details,
            source_blocks=[block],
        )
        self.result.add_candidate(candidate)
        return candidate

    def add_step_number(self, block: Text, score: float = 1.0) -> Candidate:
        self._ensure_classifier_registered("step_number")

        # Parse step value from the text block
        step_value = extract_step_number_value(block.text) or 0

        score_details = StepNumberScore(
            components={
                "text_score": score,
                "font_size_score": 0.5,
            },
            total_score=score,
            step_value=step_value,
        )
        candidate = Candidate(
            bbox=block.bbox,
            label="step_number",
            score=score,
            score_details=score_details,
            source_blocks=[block],
        )
        self.result.add_candidate(candidate)
        return candidate

    def add_part(
        self,
        part_count_candidate: Candidate,
        image_block: Image,
        score: float = 1.0,
        part_number_candidate: Candidate | None = None,
        piece_length_candidate: Candidate | None = None,
    ) -> Candidate:
        self._ensure_classifier_registered("part")
        self._ensure_classifier_registered("part_image")

        # Create a part_image candidate for the image
        part_image_candidate = Candidate(
            bbox=image_block.bbox,
            label="part_image",
            score=1.0,
            score_details=RuleScore(components={"size_ratio": 1.0}, total_score=1.0),
            source_blocks=[image_block],
        )
        self.result.add_candidate(part_image_candidate)

        score_details = _PartPairScore(
            distance=10.0,
            alignment_offset=0.0,  # Perfect alignment for test fixtures
            part_count_candidate=part_count_candidate,
            part_image_candidate=part_image_candidate,
            part_number_candidate=part_number_candidate,
            piece_length_candidate=piece_length_candidate,
        )
        candidate = Candidate(
            bbox=part_count_candidate.bbox.union(image_block.bbox),
            label="part",
            score=score,
            score_details=score_details,
            source_blocks=[],  # Part is composite, no direct source blocks
        )
        self.result.add_candidate(candidate)
        return candidate

    def add_parts_list(
        self,
        drawing_block: Drawing,
        part_candidates: list[Candidate],
        score: float = 1.0,
    ) -> Candidate:
        self._ensure_classifier_registered("parts_list")
        score_details = _PartsListScore(part_candidates=part_candidates)
        candidate = Candidate(
            bbox=drawing_block.bbox,
            label="parts_list",
            score=score,
            score_details=score_details,
            source_blocks=[drawing_block],
        )
        self.result.add_candidate(candidate)
        return candidate

    def add_piece_length(
        self,
        text_block: Text,
        drawing_block: Drawing,
        score: float = 1.0,
        value: int = 0,
    ) -> Candidate:
        self._ensure_classifier_registered("piece_length")
        score_details = RuleScore(
            components={
                "Value": score,
                "ContainerFit": score,
                "FontSize": 0.5,
            },
            total_score=score,
        )
        candidate = Candidate(
            bbox=text_block.bbox.union(drawing_block.bbox),
            label="piece_length",
            score=score,
            score_details=score_details,
            source_blocks=[text_block, drawing_block],
        )
        self.result.add_candidate(candidate)
        return candidate


@pytest.fixture
def candidate_factory():
    def _factory(result: ClassificationResult) -> CandidateFactory:
        return CandidateFactory(result)

    return _factory
