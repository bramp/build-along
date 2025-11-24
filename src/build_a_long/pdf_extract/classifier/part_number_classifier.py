"""
Part number (element ID) classifier.

Purpose
-------
Detect LEGO part numbers (element IDs) - typically 6-7 digit numbers that appear
on catalog pages below part counts.

Debugging
---------
Enable DEBUG logs with LOG_LEVEL=DEBUG.
"""

import logging
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
    extract_element_id,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElements,
    PartNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


# Empirical distribution of element ID lengths (fraction of observed IDs)
# Source: project-wide histogram
LENGTH_DISTRIBUTION: dict[int, float] = {
    4: 0.0002,
    5: 0.0050,
    6: 0.0498,
    7: 0.9447,
    8: 0.0003,
}


@dataclass
class _PartNumberScore:
    """Internal score representation for part number classification."""

    text_score: float
    """Score based on text matching pattern and length distribution (0.0-1.0).
    
    This combines binary pattern matching (valid element ID format) with the
    empirical length distribution so uncommon lengths are down-weighted.
    """

    font_size_score: float
    """Score based on font size match to expected catalog element ID size (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components.

        Combines text matching and font size matching with text weighted more heavily.
        """
        # For part numbers, reuse the part-count weights
        text_weight = config.part_count_text_weight
        font_size_weight = config.part_count_font_size_weight

        # If catalog element ID hint is not available, zero out font weight
        if config.font_size_hints.catalog_element_id_size is None:
            font_size_weight = 0.0

        # Sum the weighted components
        score = text_weight * self.text_score + font_size_weight * self.font_size_score

        # Normalize by the sum of weights to keep score in [0, 1]
        total_weight = text_weight + font_size_weight
        return score / total_weight if total_weight > 0 else 0.0


@dataclass(frozen=True)
class PartNumberClassifier(LabelClassifier):
    """Classifier for LEGO part numbers (element IDs)."""

    outputs = frozenset({"part_number"})
    requires = frozenset()

    def score(self, result: ClassificationResult) -> None:
        """Score text blocks and create candidates WITHOUT construction."""
        page_data = result.page_data
        if not page_data.blocks:
            return

        for block in page_data.blocks:
            if not isinstance(block, Text):
                continue

            # Try to extract element ID (which also validates the format)
            element_id = extract_element_id(block.text)

            # Score based on pattern match and length distribution
            if element_id is not None:
                num_digits = len(element_id)
                # Base score (0.5) + distribution bonus (scaled to 0.0-0.5 range)
                distribution_score = LENGTH_DISTRIBUTION.get(num_digits, 0.0)
                text_score = 0.5 + (0.5 * distribution_score)
            else:
                text_score = 0.0

            font_size_score = self._score_font_size(
                block, self.config.font_size_hints.catalog_element_id_size
            )

            # Store detailed score object
            detail_score = _PartNumberScore(
                text_score=text_score,
                font_size_score=font_size_score,
            )

            combined = detail_score.combined_score(self.config)

            # Skip candidates below minimum score threshold
            if combined < self.config.part_number_min_score:
                log.debug(
                    "[part_number] Skipping low-score candidate: text='%s' "
                    "score=%.3f (below threshold %.3f)",
                    block.text,
                    combined,
                    self.config.part_number_min_score,
                )
                continue

            # Create candidate WITHOUT construction
            result.add_candidate(
                "part_number",
                Candidate(
                    bbox=block.bbox,
                    label="part_number",
                    score=combined,
                    score_details=detail_score,
                    constructed=None,
                    source_blocks=[block],
                    failure_reason=None,
                ),
            )

    def construct(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a PartNumber element from a winning candidate."""
        # Get the source text block
        assert len(candidate.source_blocks) == 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Extract and validate element ID
        element_id = extract_element_id(block.text)
        if element_id is None:
            raise ValueError(f"Text doesn't match part number pattern: '{block.text}'")

        # Successfully constructed
        return PartNumber(element_id=element_id, bbox=block.bbox)

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for part numbers.

        DEPRECATED: Calls score() + construct() for backward compatibility.
        """
        self.score(result)
        self._construct_all_candidates(result, "part_number")
