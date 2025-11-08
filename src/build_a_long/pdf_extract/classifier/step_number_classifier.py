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
from build_a_long.pdf_extract.extractor import PageData
from build_a_long.pdf_extract.extractor.lego_page_elements import StepNumber
from build_a_long.pdf_extract.extractor.page_blocks import Text


@dataclass
class _StepNumberScore:
    """Internal score representation for step number classification."""

    text_score: float
    """Score based on how well the text matches step number patterns (0.0-1.0)."""

    size_score: float
    """Score based on element height relative to page number height (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components."""
        # Sum the weighted components
        score = (
            config.step_number_text_weight * self.text_score
            + config.step_number_size_weight * self.size_score
        )
        # Normalize by the sum of weights to keep score in [0, 1]
        total_weight = config.step_number_text_weight + config.step_number_size_weight
        return score / total_weight if total_weight > 0 else 0.0


class StepNumberClassifier(LabelClassifier):
    """Classifier for step numbers."""

    outputs = {"step_number"}
    requires = {"page_number"}

    def _score_step_number_size(
        self, element: Text, page_num_height: float | None
    ) -> float:
        """Score based on element height relative to page number height.

        Returns 0.0 if element is not significantly taller than page number,
        scaling up to 1.0 as element gets taller.
        """
        if not page_num_height or page_num_height <= 0.0:
            return 0.0

        h = element.bbox.height
        if h <= page_num_height * 1.1:
            return 0.0

        ratio_over = (h / page_num_height) - 1.0
        return max(0.0, min(1.0, ratio_over / 0.5))

    def evaluate(
        self,
        page_data: PageData,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for step numbers.

        This method scores each text element, attempts to construct StepNumber objects,
        and stores all candidates with their scores and any failure reasons.
        """
        if not page_data.blocks:
            return

        page_num_height: float | None = None
        # Find the page_number block to use for size comparison
        labeled_blocks = result.get_labeled_blocks()
        for block in page_data.blocks:
            if labeled_blocks.get(block) == "page_number":
                page_num_height = block.bbox.height
                break

        # Get page bbox and height for bottom band check
        page_bbox = page_data.bbox
        assert page_bbox is not None
        page_height = page_bbox.height

        for block in page_data.blocks:
            if not isinstance(block, Text):
                continue

            # Skip blocks in the bottom 10% of the page where page numbers typically appear
            block_center_y = (block.bbox.y0 + block.bbox.y1) / 2
            bottom_threshold = page_bbox.y1 - (page_height * 0.1)
            if block_center_y >= bottom_threshold:
                continue

            text_score = self._score_step_number_text(block.text)
            if text_score == 0.0:
                continue

            size_score = self._score_step_number_size(block, page_num_height)

            # If we have a page number for size comparison, require the block to be
            # taller than the page number (size_score > 0). This prevents small
            # numeric text from being classified as step numbers.
            if page_num_height and size_score == 0.0:
                continue

            # Store detailed score object
            detail_score = _StepNumberScore(
                text_score=text_score,
                size_score=size_score,
            )

            # Try to construct (parse step number value)
            value = extract_step_number_value(block.text)
            constructed_elem = None
            failure_reason = None

            if value is not None:
                constructed_elem = StepNumber(
                    value=value,
                    bbox=block.bbox,
                )
            else:
                failure_reason = (
                    f"Could not parse step number from text: '{block.text}'"
                )

            # Add candidate
            result.add_candidate(
                "step_number",
                Candidate(
                    bbox=block.bbox,
                    label="step_number",
                    score=detail_score.combined_score(self.config),
                    score_details=detail_score,
                    constructed=constructed_elem,
                    source_block=block,
                    failure_reason=failure_reason,
                    is_winner=False,  # Will be set by classify()
                ),
            )

    def classify(self, page_data: PageData, result: ClassificationResult) -> None:
        """Select winning step numbers from pre-built candidates."""
        # Get pre-built candidates
        candidate_list = result.get_candidates("step_number")

        # Find the page number block to avoid classifying it as a step number
        page_number_blocks = result.get_blocks_by_label("page_number")
        page_number_block = page_number_blocks[0] if page_number_blocks else None

        # Mark winners (all successfully constructed candidates that aren't the page number
        # and meet the confidence threshold)
        for candidate in candidate_list:
            if candidate.source_block is page_number_block:
                # Don't classify page number as step number
                candidate.failure_reason = "Block is already labeled as page_number"
                continue

            if candidate.score < self.config.min_confidence_threshold:
                candidate.failure_reason = (
                    f"Score {candidate.score:.2f} below threshold "
                    f"{self.config.min_confidence_threshold}"
                )
                continue

            if candidate.constructed is None:
                # Already has failure_reason from calculate_scores
                continue

            # Check if this candidate has been removed due to overlap with a
            # previous winner
            if candidate.source_block is not None and result.is_removed(
                candidate.source_block
            ):
                continue

            # This is a winner!
            assert isinstance(candidate.constructed, StepNumber)
            result.mark_winner(candidate, candidate.constructed)
            # Note: Do NOT remove child bboxes - parts lists might be nearby
            self.classifier._remove_similar_bboxes(
                page_data, candidate.source_block, result
            )

    def _score_step_number_text(self, text: str) -> float:
        """Score text based on how well it matches step number patterns.

        Returns:
            1.0 if text matches step number pattern, 0.0 otherwise
        """
        # Use the extraction function to validate format
        if extract_step_number_value(text) is not None:
            return 1.0
        return 0.0
