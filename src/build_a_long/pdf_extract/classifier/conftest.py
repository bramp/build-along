import pytest

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.part_count_classifier import _PartCountScore
from build_a_long.pdf_extract.classifier.parts_classifier import _PartPairScore
from build_a_long.pdf_extract.classifier.parts_list_classifier import _PartsListScore
from build_a_long.pdf_extract.classifier.piece_length_classifier import (
    _PieceLengthScore,
)
from build_a_long.pdf_extract.classifier.step_number_classifier import _StepNumberScore
from build_a_long.pdf_extract.extractor.page_blocks import Drawing, Image, Text


class CandidateFactory:
    """Helper to create and register candidates with proper score details."""

    def __init__(self, result: ClassificationResult):
        self.result = result

    def add_part_count(self, block: Text, score: float = 1.0) -> Candidate:
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
            constructed=None,
            source_blocks=[block],
        )
        self.result.add_candidate("part_count", candidate)
        return candidate

    def add_step_number(self, block: Text, score: float = 1.0) -> Candidate:
        score_details = _StepNumberScore(text_score=score, font_size_score=0.5)
        candidate = Candidate(
            bbox=block.bbox,
            label="step_number",
            score=score,
            score_details=score_details,
            constructed=None,
            source_blocks=[block],
        )
        self.result.add_candidate("step_number", candidate)
        return candidate

    def add_part(
        self,
        part_count_candidate: Candidate,
        image_block: Image,
        score: float = 1.0,
        part_number_candidate: Candidate | None = None,
        piece_length_candidate: Candidate | None = None,
    ) -> Candidate:
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
            constructed=None,
            source_blocks=[image_block],
        )
        self.result.add_candidate("part", candidate)
        return candidate

    def add_parts_list(
        self,
        drawing_block: Drawing,
        part_candidates: list[Candidate],
        score: float = 1.0,
    ) -> Candidate:
        score_details = _PartsListScore(part_candidates=part_candidates)
        candidate = Candidate(
            bbox=drawing_block.bbox,
            label="parts_list",
            score=score,
            score_details=score_details,
            constructed=None,
            source_blocks=[drawing_block],
        )
        self.result.add_candidate("parts_list", candidate)
        return candidate

    def add_piece_length(
        self,
        text_block: Text,
        drawing_block: Drawing,
        score: float = 1.0,
        value: int = 0,
    ) -> Candidate:
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
            constructed=None,
            source_blocks=[text_block, drawing_block],
        )
        self.result.add_candidate("piece_length", candidate)
        return candidate


@pytest.fixture
def candidate_factory():
    def _factory(result: ClassificationResult) -> CandidateFactory:
        return CandidateFactory(result)

    return _factory
