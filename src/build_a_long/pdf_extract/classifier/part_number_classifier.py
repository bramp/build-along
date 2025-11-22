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
from build_a_long.pdf_extract.extractor.lego_page_elements import PartNumber
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

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for part numbers.

        This method scores each text element, attempts to construct PartNumber objects,
        and stores all candidates with their scores and any failure reasons.
        """
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

            # Construct PartNumber element if extraction succeeded
            constructed_elem = None
            failure_reason = None

            if element_id is None:
                failure_reason = (
                    f"Text doesn't match part number pattern: '{block.text}'"
                )
            else:
                constructed_elem = PartNumber(
                    element_id=element_id,
                    bbox=block.bbox,
                )

            # Add candidate
            result.add_candidate(
                "part_number",
                Candidate(
                    bbox=block.bbox,
                    label="part_number",
                    score=combined,
                    score_details=detail_score,
                    constructed=constructed_elem,
                    source_block=block,
                    failure_reason=failure_reason,
                ),
            )
