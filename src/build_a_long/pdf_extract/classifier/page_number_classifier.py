"""
Page number classifier.
"""

import logging
import math
import re
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
    extract_page_number_value,
)
from build_a_long.pdf_extract.extractor.bbox import BBox
from build_a_long.pdf_extract.extractor.lego_page_elements import (
    LegoPageElements,
    PageNumber,
)
from build_a_long.pdf_extract.extractor.page_blocks import Text

log = logging.getLogger(__name__)


@dataclass
class _PageNumberScore:
    """Internal score representation for page number classification."""

    text_score: float
    """Score based on how well the text matches page number patterns (0.0-1.0)."""

    position_score: float
    """Score based on position in bottom corners of page (0.0-1.0)."""

    page_value_score: float
    """Score based on how well the text matches the expected page number
    (0.0-1.0)."""

    font_size_score: float
    """Score based on font size match to expected page number size (0.0-1.0)."""

    # TODO Test this score is always between 0.0 and 1.0
    def combined_score(self, config: ClassifierConfig) -> float:
        """Calculate final weighted score from components."""
        # Determine font size weight based on whether hints are available
        font_size_weight = config.page_number_font_size_weight
        if config.font_size_hints.page_number_size is None:
            # No hint available, zero out the font size weight
            font_size_weight = 0.0

        # Sum the weighted components
        score = (
            config.page_number_text_weight * self.text_score
            + config.page_number_position_weight * self.position_score
            + config.page_number_page_value_weight * self.page_value_score
            + font_size_weight * self.font_size_score
        )
        # Normalize by the sum of weights to keep score in [0, 1]
        total_weight = (
            config.page_number_text_weight
            + config.page_number_position_weight
            + config.page_number_page_value_weight
            + font_size_weight
        )
        return score / total_weight if total_weight > 0 else 0.0


@dataclass(frozen=True)
class PageNumberClassifier(LabelClassifier):
    """Classifier for page numbers."""

    outputs = frozenset({"page_number"})
    requires = frozenset()

    def score(self, result: ClassificationResult) -> None:
        """Legacy classifier - uses evaluate() instead of score() + construct()."""
        raise NotImplementedError(
            f"{self.__class__.__name__} uses legacy evaluate() method. "
            "Implement score() and construct() to use two-phase classification."
        )

    def construct(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Legacy classifier - uses evaluate() instead of score() + construct()."""
        raise NotImplementedError(
            f"{self.__class__.__name__} uses legacy evaluate() method. "
            "Implement score() and construct() to use two-phase classification."
        )

    def evaluate(
        self,
        result: ClassificationResult,
    ) -> None:
        """Evaluate elements and create candidates for page numbers.

        This method scores each text element, attempts to construct PageNumber objects,
        and stores all candidates with their scores and any failure reasons.
        """
        page_data = result.page_data
        page_bbox = page_data.bbox
        assert page_bbox is not None

        for block in page_data.blocks:
            # TODO Support non-text blocks - such as images of text.
            if not isinstance(block, Text):
                continue

            # Score the block
            text_score = self._score_page_number_text(block.text)
            position_score = self._score_page_number_position(block, page_bbox)
            page_value_score = self._score_page_number(
                block.text, page_data.page_number
            )
            font_size_score = self._score_font_size(
                block, self.config.font_size_hints.page_number_size
            )

            # Create detailed score object
            score = _PageNumberScore(
                text_score=text_score,
                position_score=position_score,
                page_value_score=page_value_score,
                font_size_score=font_size_score,
            )

            combined = score.combined_score(self.config)

            # Skip candidates below minimum score threshold
            if combined < self.config.page_number_min_score:
                log.debug(
                    "[page_number] Skipping low-score candidate: text='%s' "
                    "score=%.3f (below threshold %.3f)",
                    block.text,
                    combined,
                    self.config.page_number_min_score,
                )
                continue

            # Try to construct the LegoElement (parse the text)
            value = extract_page_number_value(block.text)
            constructed_elem = None
            failure_reason = None

            if text_score == 0.0:
                failure_reason = (
                    f"Text doesn't match page number pattern: '{block.text}'"
                )
            elif position_score == 0.0:
                failure_reason = "Block not in bottom 10% of page"
            elif value is None or value < 0:
                failure_reason = (
                    f"Could not parse page number from text: '{block.text}'"
                )
            else:
                # Successfully constructed
                constructed_elem = PageNumber(value=value, bbox=block.bbox)

            # Store candidate (even if construction failed, for debugging)
            result.add_candidate(
                "page_number",
                Candidate(
                    bbox=block.bbox,
                    label="page_number",
                    score=combined,
                    score_details=score,
                    constructed=constructed_elem,
                    source_blocks=[block],
                    failure_reason=failure_reason,
                ),
            )

    def _score_page_number_text(self, text: str) -> float:
        text = text.strip()
        if re.match(r"^0+\d{1,3}$", text):
            return 0.95
        if re.match(r"^\d{1,3}$", text):
            return 1.0
        return 0.0

    def _score_page_number(self, text: str, page_number: int) -> float:
        """Score how well the text matches the expected page number."""

        value = extract_page_number_value(text)
        if value is None:
            # Can't extract text.
            return 0.0

        # For every digit away from the expected page number,
        # reduce score by 10%
        diff = abs(value - page_number)
        return max(0.0, 1.0 - 0.1 * diff)

    def _is_in_bottom_band(self, element: Text, page_bbox: BBox) -> bool:
        """Check if the element is in the bottom 10% of the page height."""
        bottom_threshold = page_bbox.y1 - (page_bbox.height * 0.1)
        element_center_y = (element.bbox.y0 + element.bbox.y1) / 2
        return element_center_y >= bottom_threshold

    def _calculate_position_score(self, element: Text, page_bbox: BBox) -> float:
        """Calculate position score based on distance to bottom corners. Based
        on exp(-min_distance_to_bottom_corners / scale)"""
        element_center_x = (element.bbox.x0 + element.bbox.x1) / 2
        element_center_y = (element.bbox.y0 + element.bbox.y1) / 2
        dist_bottom_left = math.sqrt(
            (element_center_x - page_bbox.x0) ** 2
            + (element_center_y - page_bbox.y1) ** 2
        )
        dist_bottom_right = math.sqrt(
            (element_center_x - page_bbox.x1) ** 2
            + (element_center_y - page_bbox.y1) ** 2
        )
        min_dist = min(dist_bottom_left, dist_bottom_right)
        return math.exp(-min_dist / self.config.page_number_position_scale)

    def _score_page_number_position(self, element: Text, page_bbox: BBox) -> float:
        # TODO Take the hint, and increase score if near expected position (of
        # expected size).

        if not self._is_in_bottom_band(element, page_bbox):
            return 0.0

        # TODO it might be simplier to check if in left or right band.
        return self._calculate_position_score(element, page_bbox)
