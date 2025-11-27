"""
Step number classifier.
"""

from dataclasses import dataclass

from build_a_long.pdf_extract.classifier.classification_result import (
    Candidate,
    ClassificationResult,
    ClassifierConfig,
)
from build_a_long.pdf_extract.classifier.label_classifier import (
    LabelClassifier,
)
from build_a_long.pdf_extract.classifier.text_extractors import (
    extract_step_number_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    StepNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text


@dataclass
class _StepNumberScore:
    """Internal score representation for step number classification."""

    text_score: float
    """Score based on how well the text matches step number patterns (0.0-1.0)."""

    font_size_score: float
    """Score based on font size match to expected step number size (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components.

        Combines text matching and font size matching with text weighted more heavily.
        """
        # Determine font size weight based on whether hints are available
        font_size_weight = config.step_number_font_size_weight
        if config.font_size_hints.step_number_size is None:
            # No hint available, zero out the font size weight
            font_size_weight = 0.0

        # Sum the weighted components
        score = (
            config.step_number_text_weight * self.text_score
            + font_size_weight * self.font_size_score
        )
        # Normalize by the sum of weights to keep score in [0, 1]
        total_weight = config.step_number_text_weight + font_size_weight
        return score / total_weight if total_weight > 0 else 0.0


@dataclass(frozen=True)
class StepNumberClassifier(LabelClassifier):
    """Classifier for step numbers."""

    output = "step_number"
    requires = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score text blocks and create candidates.

        This method:
        1. Iterates through all text blocks on the page
        2. Skips blocks in the bottom 10% (where page numbers appear)
        3. Calculates component scores based on text pattern, font size.
        """
        page_data = result.page_data
        if not page_data.blocks:
            return

        # Get page bbox and height for bottom band check
        page_bbox = page_data.bbox
        assert page_bbox is not None
        page_height = page_bbox.height

        for block in page_data.blocks:
            if not isinstance(block, Text):
                continue

            # Skip blocks in the bottom 10% of the page where page numbers
            # typically appear
            # TODO Maybe we don't need this anymore since we have page number labeling?
            block_center_y = (block.bbox.y0 + block.bbox.y1) / 2
            bottom_threshold = page_bbox.y1 - (page_height * 0.1)
            if block_center_y >= bottom_threshold:
                continue

            text_score = self._score_step_number_text(block.text)
            if text_score == 0.0:
                continue

            # Get expected font size from hints and score font size match
            font_size_score = self._score_font_size(
                block, self.config.font_size_hints.step_number_size
            )

            # Store detailed score object
            detail_score = _StepNumberScore(
                text_score=text_score,
                font_size_score=font_size_score,
            )

            # Create candidate
            result.add_candidate(
                Candidate(
                    bbox=block.bbox,
                    label="step_number",
                    score=detail_score.combined_score(self.config),
                    score_details=detail_score,
                    source_blocks=[block],
                ),
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> StepNumber:
        """Construct a StepNumber element from a single candidate.

        Args:
            candidate: The winning candidate to construct
            result: Classification result for context

        Returns:
            StepNumber: The constructed step number element

        Raises:
            ValueError: If construction fails (parse error, etc.)
        """
        # Get the source text block
        assert len(candidate.source_blocks) == 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Parse the step number value
        # TODO use the value as determined in the score.
        value = extract_step_number_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse step number from text: '{block.text}'")

        # Successfully constructed
        return StepNumber(value=value, bbox=block.bbox)

    def _score_step_number_text(self, text: str) -> float:
        """Score text based on how well it matches step number patterns.

        Returns:
            1.0 if text matches step number pattern, 0.0 otherwise
        """
        # Use the extraction function to validate format
        if extract_step_number_value(text) is not None:
            return 1.0
        return 0.0
