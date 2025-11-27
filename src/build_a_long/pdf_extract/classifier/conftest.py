import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.parts.part_count_classifier import (
    PartCountClassifier,
    _PartCountScore,
)
from build_a_long.pdf_extract.classifier.parts.part_number_classifier import (
    PartNumberClassifier,
)
from build_a_long.pdf_extract.classifier.parts.parts_classifier import (
    PartsClassifier,
    _PartPairScore,
)
from build_a_long.pdf_extract.classifier.parts.parts_list_classifier import (
    PartsListClassifier,
    _PartsListScore,
)
from build_a_long.pdf_extract.classifier.parts.piece_length_classifier import (
    PieceLengthClassifier,
    _PieceLengthScore,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import (
    StepNumberClassifier,
    _StepNumberScore,
)
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
            "parts_list": PartsListClassifier,
            "piece_length": PieceLengthClassifier,
            "part_number": PartNumberClassifier,
        }

        if label in classifier_map:
            classifier = classifier_map[label](self.config)
            self.result._register_classifier(label, classifier)
            self._registered_classifiers.add(label)

    def add_part_count(self, block: Text, score: float = 1.0) -> Candidate:
        self._ensure_classifier_registered("part_count")
        score_details = _PartCountScore(
            text_score=score,
            font_size_score=0.5,
            matched_hint="catalog_part_count",
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
        score_details = _StepNumberScore(text_score=score, font_size_score=0.5)
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
        score_details = _PartPairScore(
            distance=10.0,
            part_count_candidate=part_count_candidate,
            image=image_block,
            part_number_candidate=part_number_candidate,
            piece_length_candidate=piece_length_candidate,
        )
        candidate = Candidate(
            bbox=part_count_candidate.bbox.union(image_block.bbox),
            label="part",
            score=score,
            score_details=score_details,
            source_blocks=[image_block],
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
        score_details = _PieceLengthScore(
            text_score=score,
            context_score=1.0,
            font_size_score=0.5,
            value=value,
            containing_drawing=drawing_block,
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
