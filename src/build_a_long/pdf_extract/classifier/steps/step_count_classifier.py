"""
Step count classifier.

Purpose
-------
Detect step-count text like "2x" that appears in substep callout boxes.
These are similar to part counts but use a larger font size (typically 16pt),
between part count size and step number size.

Debugging
---------
Enable DEBUG logs with LOG_LEVEL=DEBUG.
"""

import logging

from build_a_long.pdf_extract.classifier.candidate import Candidate
from build_a_long.pdf_extract.classifier.classification_result import (
    ClassificationResult,
)
from build_a_long.pdf_extract.classifier.config import StepCountConfig
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.score import Score, Weight
from build_a_long.pdf_extract.classifier.text import (
    extract_part_count_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    StepCount,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


class _StepCountScore(Score):
    """Internal score representation for step count classification."""

    text_score: float
    """Score based on how well the text matches count patterns (0.0-1.0)."""

    font_size_score: float
    """Score based on font size being between part count and step number (0.0-1.0)."""

    config: StepCountConfig
    """Step count configuration for dynamic score calculations."""

    def score(self) -> Weight:
        """Calculate final weighted score from components.

        Combines text matching and font size matching.
        """
        return (
            self.config.text_weight * self.text_score
            + self.config.font_size_weight * self.font_size_score
        )


class StepCountClassifier(LabelClassifier):
    """Classifier for step counts (substep counts like "2x").

    These are count labels that appear inside substep callout boxes,
    indicating how many times to build the sub-assembly.
    They use a font size between part counts and step numbers.
    """

    output = "step_count"
    requires = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score text blocks and create candidates."""
        page_data = result.page_data
        if not page_data.blocks:
            return

        step_count_config = self.config.step_count

        for block in page_data.blocks:
            if not isinstance(block, Text):
                continue

            # Check if text matches count pattern (e.g., "2x", "4x")
            text_score = self._score_count_text(block.text)
            if text_score == 0.0:
                continue

            # Score font size: should be >= part_count_size and <= step_number_size
            font_size_score = 0.5  # Default neutral score
            if block.font_size is not None:
                font_size_score = self._score_step_count_font_size(block.font_size)

            detail_score = _StepCountScore(
                text_score=text_score,
                font_size_score=font_size_score,
                config=step_count_config,
            )

            combined = detail_score.score()

            # Skip candidates below minimum score threshold
            if combined < step_count_config.min_score:
                log.debug(
                    "[step_count] Skipping low-score candidate: text='%s' "
                    "font_size=%.1f score=%.3f (below threshold %.3f)",
                    block.text,
                    block.font_size,
                    combined,
                    step_count_config.min_score,
                )
                continue

            result.add_candidate(
                Candidate(
                    bbox=block.bbox,
                    label="step_count",
                    score=combined,
                    score_details=detail_score,
                    source_blocks=[block],
                ),
            )
            log.debug(
                "[step_count] Candidate: text='%s' font_size=%.1f score=%.3f",
                block.text,
                block.font_size,
                combined,
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> StepCount:
        """Construct a StepCount element from a candidate.

        The candidate may include additional source blocks (e.g., text outline
        effects) beyond the primary Text block.
        """
        # Get the primary text block (first in source_blocks)
        assert len(candidate.source_blocks) >= 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Parse the count value
        value = extract_part_count_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse step count from text: '{block.text}'")

        return StepCount(count=value, bbox=block.bbox)

    def _score_count_text(self, text: str) -> float:
        """Score text based on how well it matches count patterns.

        Returns:
            1.0 if text matches count pattern (e.g., "2x"), 0.0 otherwise
        """
        if extract_part_count_value(text) is not None:
            return 1.0
        return 0.0

    def _score_step_count_font_size(self, font_size: float) -> float:
        """Score font size for step counts.

        Step counts should have a font size that is:
        - Greater than or equal to part_count_size
        - Less than or equal to step_number_size

        Returns:
            1.0 if font size is in the expected range
            0.5 if we don't have hints to compare against
            0.0 if font size is clearly outside the range
        """
        hints = self.config.font_size_hints
        part_count_size = hints.part_count_size
        step_number_size = hints.step_number_size

        # If we don't have both hints, give a neutral score
        if part_count_size is None or step_number_size is None:
            return 0.5

        # Check if font size is in the expected range
        # Allow some tolerance (within 1pt)
        tolerance = 1.0

        if font_size < part_count_size - tolerance:
            # Too small - likely a part count or something smaller
            return 0.0

        if font_size > step_number_size + tolerance:
            # Too large - likely a step number or larger element
            return 0.0

        # Font size is in the expected range
        # Give higher score if it's strictly between the two sizes
        if font_size > part_count_size + tolerance:
            # Clearly larger than part count - good indicator
            return 1.0

        # Font size is close to part count size - less confident
        return 0.7
