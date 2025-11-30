"""
Shine classifier.

Purpose
-------
Identify small star-like drawings that indicate shiny/metallic parts.
These appear in the top-right area of part images.
"""

import logging

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import Shine
from build_a_long.pdf_extract.extractor.page_blocks import Drawing

log = logging.getLogger(__name__)


class _ShineScore(Score):
    """Internal score representation for shine/sparkle classification."""

    shape_score: float
    """Score based on shape being star-like (0.0-1.0)."""

    size_score: float
    """Score based on size being small (0.0-1.0)."""

    def score(self) -> Weight:
        """Calculate final weighted score from components."""
        return self.shape_score * 0.7 + self.size_score * 0.3


class ShineClassifier(LabelClassifier):
    """Classifier for 'shine' or 'sparkle' effects."""

    output = "shine"
    requires = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score drawing blocks and create candidates."""
        page_data = result.page_data
        if not page_data.blocks:
            return

        # Find all Drawing elements
        drawings = [b for b in page_data.blocks if isinstance(b, Drawing)]

        for drawing in drawings:
            # Heuristic: Shines are small (approx 5-15 units)
            # Example: 6.81 x 8.5
            width = drawing.bbox.width
            height = drawing.bbox.height

            # Check size constraints
            if not (5.0 <= width <= 15.0 and 5.0 <= height <= 15.0):
                continue

            # Check aspect ratio (should be roughly square)
            ratio = width / height
            if not (0.7 <= ratio <= 1.4):
                continue

            # Simple score based on size/shape match
            # TODO: Implement better shape analysis
            shape_score = 1.0
            size_score = 1.0

            score_details = _ShineScore(
                shape_score=shape_score,
                size_score=size_score,
            )

            log.debug(
                "[shine] Candidate: drawing id=%d bbox=%s size=%.1fx%.1f",
                drawing.id,
                drawing.bbox,
                width,
                height,
            )

            result.add_candidate(
                Candidate(
                    bbox=drawing.bbox,
                    label="shine",
                    score=score_details.score(),
                    score_details=score_details,
                    source_blocks=[drawing],
                ),
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> Shine:
        """Construct a Shine element from a single candidate."""
        return Shine(bbox=candidate.bbox)
