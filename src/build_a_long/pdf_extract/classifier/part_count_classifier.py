"""
Part count classifier.

Purpose
-------
Detect part-count text like "2x", "3X", or "5×".

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
    extract_part_count_value,
)
from build_a_long.pdf_extract.extractor.lego_page_elements import PartCount
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


@dataclass
class _PartCountScore:
    """Internal score representation for part count classification."""

    text_score: float
    """Score based on how well the text matches part count patterns (0.0-1.0)."""

    font_size_score: float
    """Score based on font size match to expected part count size (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components.

        Combines text matching and font size matching with text weighted more heavily.
        """
        # Determine font size weight based on whether hints are available
        font_size_weight = config.part_count_font_size_weight

        # If neither instruction nor catalog hints are available, zero out weight
        hints = config.font_size_hints
        if hints.part_count_size is None and hints.catalog_part_count_size is None:
            font_size_weight = 0.0

        # Sum the weighted components
        score = (
            config.part_count_text_weight * self.text_score
            + font_size_weight * self.font_size_score
        )
        # Normalize by the sum of weights to keep score in [0, 1]
        total_weight = config.part_count_text_weight + font_size_weight
        return score / total_weight if total_weight > 0 else 0.0


@dataclass(frozen=True)
class PartCountClassifier(LabelClassifier):
    """Classifier for part counts."""

    outputs = frozenset({"part_count"})
    requires = frozenset()

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for part counts.

        This method scores each text element, attempts to construct PartCount objects,
        and stores all candidates with their scores and any failure reasons.
        """
        page_data = result.page_data
        if not page_data.blocks:
            return

        for block in page_data.blocks:
            if not isinstance(block, Text):
                continue

            # Try matching against both instruction and catalog part count font sizes
            text_score = self._score_part_count_text(block.text)

            # Score against instruction part count size
            instruction_font_score = self._score_font_size(
                block, self.config.font_size_hints.part_count_size
            )

            # Score against catalog part count size
            catalog_font_score = self._score_font_size(
                block, self.config.font_size_hints.catalog_part_count_size
            )

            # Use the better matching font size
            font_size_score = max(instruction_font_score, catalog_font_score)

            # Determine which hint matched best
            matched_hint = None
            if font_size_score > 0:
                if instruction_font_score > catalog_font_score:
                    matched_hint = "part_count"
                else:
                    matched_hint = "catalog_part_count"

            # Store detailed score object
            detail_score = _PartCountScore(
                text_score=text_score,
                font_size_score=font_size_score,
            )

            combined = detail_score.combined_score(self.config)

            # Skip candidates below minimum score threshold
            if combined < self.config.part_count_min_score:
                log.debug(
                    "[part_count] Skipping low-score candidate: text='%s' "
                    "score=%.3f (below threshold %.3f)",
                    block.text,
                    combined,
                    self.config.part_count_min_score,
                )
                continue

            # Try to construct (parse part count value)
            value = extract_part_count_value(block.text)
            constructed_elem = None
            failure_reason = None

            if text_score == 0.0:
                failure_reason = (
                    f"Text doesn't match part count pattern: '{block.text}'"
                )
            elif value is None:
                failure_reason = f"Could not parse part count from text: '{block.text}'"
            else:
                constructed_elem = PartCount(
                    count=value,
                    bbox=block.bbox,
                    matched_hint=matched_hint,
                )

            # Add candidate
            result.add_candidate(
                "part_count",
                Candidate(
                    bbox=block.bbox,
                    label="part_count",
                    score=combined,
                    score_details=detail_score,
                    constructed=constructed_elem,
                    source_blocks=[block],
                    failure_reason=failure_reason,
                ),
            )

    def _score_part_count_text(self, text: str) -> float:
        """Score text based on how well it matches part count patterns.

        Returns:
            1.0 if text matches part count pattern, 0.0 otherwise
        """
        # Use the extraction function to validate format
        if extract_part_count_value(text) is not None:
            return 1.0
        return 0.0
