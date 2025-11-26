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
        """Score text blocks and create candidates WITHOUT construction.

        This method:
        1. Iterates through all text blocks on the page
        2. Calculates component scores (text pattern, position, page value, font size)
        3. Computes combined score
        4. Creates Candidates with constructed=None for viable candidates
        5. Stores score_details for debugging and later construction
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

            # STRICTER FILTERING:
            # 1. Must be in the bottom band (position_score > 0).
            # 2. If it's a single digit (common step number), require extremely high position confidence
            #    or match to page number value.
            if position_score < 0.1:
                continue

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

            # Create candidate WITHOUT construction (constructed=None)
            # Construction happens later in construct() method
            result.add_candidate(
                "page_number",
                Candidate(
                    bbox=block.bbox,
                    label="page_number",
                    score=combined,
                    score_details=score,
                    constructed=None,  # Not constructed yet!
                    source_blocks=[block],
                    failure_reason=None,  # No failure yet, construction happens later
                ),
            )

    def construct(self, result: ClassificationResult) -> None:
        """Construct PageNumber elements from candidates.

        Only constructs the highest-scoring candidate (there should only be one
        page number per page).
        """
        candidates = result.get_candidates("page_number")
        if not candidates:
            return

        # Sort by score descending and construct only the top one
        candidates_sorted = sorted(candidates, key=lambda c: c.score, reverse=True)
        winner = candidates_sorted[0]

        try:
            elem = self.construct_candidate(winner, result)
            winner.constructed = elem
        except Exception as e:
            winner.failure_reason = str(e)

    def construct_candidate(
        self, candidate: Candidate, result: ClassificationResult
    ) -> LegoPageElements:
        """Construct a PageNumber element from a single candidate.

        This method:
        1. Extracts the text from the candidate's source block
        2. Parses the page number value
        3. Validates the extraction and score components
        4. Returns a constructed PageNumber or raises ValueError

        Args:
            candidate: The winning candidate to construct
            result: Classification result for context

        Returns:
            PageNumber: The constructed page number element

        Raises:
            ValueError: If construction fails (invalid text, parse error, etc.)
        """
        # Get the source text block
        assert len(candidate.source_blocks) == 1
        block = candidate.source_blocks[0]
        assert isinstance(block, Text)

        # Get score details for validation
        score_details = candidate.score_details
        assert isinstance(score_details, _PageNumberScore)

        # Validate score components
        if score_details.text_score == 0.0:
            raise ValueError(f"Text doesn't match page number pattern: '{block.text}'")
        if score_details.position_score == 0.0:
            raise ValueError("Block not in bottom 10% of page")

        # Parse the page number value
        value = extract_page_number_value(block.text)
        if value is None or value < 0:
            raise ValueError(f"Could not parse page number from text: '{block.text}'")

        # Successfully constructed
        return PageNumber(value=value, bbox=block.bbox)

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
