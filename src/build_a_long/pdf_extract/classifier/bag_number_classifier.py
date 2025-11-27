"""
Bag number classifier.

Purpose
-------
Identify bag numbers in LEGO instructions. These are typically large text
numbers (1, 2, 3, etc.) that appear in the top-left area of a page,
surrounded by a cluster of images forming a "New Bag" visual element.

Heuristic
---------
- Look for Text elements containing single digits or small integers
- Typically larger font size than step numbers
- Located in the top portion of the page (upper 40%)
- Often left-aligned or centered within the left portion of the page
- Usually surrounded by multiple Image/Drawing blocks forming a bag icon

Debugging
---------
Enable with `LOG_LEVEL=DEBUG` for structured logs.
"""

from __future__ import annotations

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
    extract_bag_number_value,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    BagNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


@dataclass
class _BagNumberScore:
    """Internal score representation for bag number classification."""

    text_score: float
    """Score based on how well the text matches bag number patterns (0.0-1.0)."""

    position_score: float
    """Score based on position in the page (top-left preferred) (0.0-1.0)."""

    font_size_score: float
    """Score based on font size (larger is better) (0.0-1.0)."""

    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components.

        Combines text matching, position, and font size with position and text
        weighted more heavily.
        """
        # Weight: 40% text, 40% position, 20% font size
        score = (
            0.4 * self.text_score
            + 0.4 * self.position_score
            + 0.2 * self.font_size_score
        )
        return score


@dataclass(frozen=True)
class BagNumberClassifier(LabelClassifier):
    """Classifier for bag numbers."""

    output = "bag_number"
    requires = frozenset()

    def _score(self, result: ClassificationResult) -> None:
        """Score text blocks and create candidates.

        This method iterates through all text blocks on the page and
        calculates component scores based on text pattern, position, and font size.
        """
        page_data = result.page_data
        if not page_data.blocks:
            return

        page_bbox = page_data.bbox
        assert page_bbox is not None

        for block in page_data.blocks:
            if not isinstance(block, Text):
                continue

            text_score = self._score_bag_number_text(block.text)
            if text_score == 0.0:
                continue

            position_score = self._score_position(block.bbox, page_bbox)
            if position_score == 0.0:
                continue

            font_size_score = self._score_font_size_relative(block)
            if font_size_score == 0.0:
                continue

            # Store detailed score object
            detail_score = _BagNumberScore(
                text_score=text_score,
                position_score=position_score,
                font_size_score=font_size_score,
            )

            result.add_candidate(
                Candidate(
                    bbox=block.bbox,
                    label="bag_number",
                    score=detail_score.combined_score(self.config),
                    score_details=detail_score,
                    source_blocks=[block],
                ),
            )

    def build(self, candidate: Candidate, result: ClassificationResult) -> BagNumber:
        """Construct a BagNumber element from a single candidate.

        Args:
            candidate: The winning candidate to construct
            result: Classification result for context

        Returns:
            BagNumber: The constructed bag number element

        Raises:
            ValueError: If construction fails (parse error, etc.)
        """
        # Get the source text block
        assert len(candidate.source_blocks) == 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Parse the bag number value
        value = extract_bag_number_value(block.text)
        if value is None:
            raise ValueError(f"Could not parse bag number from text: '{block.text}'")

        # Successfully constructed
        return BagNumber(value=value, bbox=block.bbox)

    def _score_bag_number_text(self, text: str) -> float:
        """Score text based on how well it matches bag number patterns.

        Returns:
            1.0 if text matches bag number pattern, 0.0 otherwise
        """
        # Use the extraction function to validate format
        if extract_bag_number_value(text) is not None:
            return 1.0
        return 0.0

    def _score_position(self, bbox: BBox, page_bbox: BBox) -> float:
        """Score based on position in the page.

        Bag numbers typically appear in the upper portion (top 40%) and
        left portion (left 50%) of the page.

        Returns:
            Score from 0.0 to 1.0
        """
        # Check vertical position (should be in top 40% of page)
        center_y = (bbox.y0 + bbox.y1) / 2
        vertical_ratio = (center_y - page_bbox.y0) / page_bbox.height

        if vertical_ratio > 0.4:
            # Too far down the page
            return 0.0

        # Score higher for positions closer to the top
        # At y=0 (top): score = 1.0
        # At y=0.4 (40% down): score = 0.0
        vertical_score = 1.0 - (vertical_ratio / 0.4)

        # Check horizontal position (prefer left half)
        center_x = (bbox.x0 + bbox.x1) / 2
        horizontal_ratio = (center_x - page_bbox.x0) / page_bbox.width

        # Favor left side, but don't completely exclude right side
        horizontal_score = 1.0 if horizontal_ratio <= 0.5 else 0.3

        # Combine scores (70% vertical, 30% horizontal)
        return 0.7 * vertical_score + 0.3 * horizontal_score

    def _score_font_size_relative(self, element: Text) -> float:
        """Score based on font size.

        Bag numbers are typically large text (around 40-60 points).
        Step numbers are typically smaller (around 16-26 points).

        Returns:
            Score from 0.0 to 1.0
        """
        font_size = element.font_size
        if font_size is None:
            return 0.0  # Require font size for bag numbers

        # Bag numbers must be quite large (at least 35 points)
        # This helps distinguish from step numbers which are typically 16-26 points
        if font_size < 35:
            # Too small - likely a step number or part count
            return 0.0
        elif font_size <= 60:
            # Good range for bag numbers
            return 1.0
        else:
            # Very large is okay but not preferred
            return max(0.5, 1.0 - (font_size - 60) / 60.0)
